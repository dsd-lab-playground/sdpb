// lab (agent) #226: benchmark of SDPB's Cholesky_Q stage on its own.
//
// A random symmetric positive definite N x N BigFloat matrix (full-precision
// random mantissas, diagonally dominant) is distributed [MC,MR] over all
// ranks, as SDPB's Q. Then, per method, a fresh copy is factored (A = U^T U):
//   el      El::Cholesky(UPPER, Q), as SDPB (El's default panel 128)
//   mm<nb>  multimodular_cholesky_upper_dist(Q, nb): El's algorithm with the
//           trailing update by multimodular BLAS (sdpb --multimodularCholeskyQ nb)
//   mmd<nb> the same with the diagonal blocks by multimodular_cholesky_lower
// and El::cholesky::SolveAfter(UPPER, NORMAL, U, b) is timed for one right-hand
// side (SDPB's Q solve in computeSearchDirection).
// Each mm factor is compared with the el factor: max |U_mm - U_el| / max |U_el|
// over the upper triangle.
//
// usage: qchol_bench N precision method[,method...] [reps]
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "sdpb_util/Environment.hxx"

#include <El.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

namespace
{
  uint64_t splitmix(uint64_t x)
  {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
  }
  // random value in [-1, 1) with a full-precision mantissa, from (i, j)
  El::BigFloat entry(int64_t i, int64_t j, int64_t n)
  {
    if(i > j)
      std::swap(i, j);
    uint64_t s = splitmix(uint64_t(i) * uint64_t(n) + uint64_t(j) + 1);
    El::BigFloat x(0), scale(1);
    const El::BigFloat two53(9007199254740992.0);
    const int pieces = int(El::gmp::Precision() / 53) + 2;
    for(int p = 0; p < pieces; ++p)
      {
        s = splitmix(s);
        const double d = double(s >> 11) / 9007199254740992.0; // [0, 1)
        x += El::BigFloat(d) * scale;
        scale /= two53;
      }
    x = x * El::BigFloat(2) - El::BigFloat(1);
    if(i == j)
      x += El::BigFloat(double(n));
    return x;
  }
  double now()
  {
    return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
  }
  double max_over_ranks(double x)
  {
    return El::mpi::AllReduce(x, El::mpi::MAX, El::mpi::COMM_WORLD);
  }
  double mean_over_ranks(double x)
  {
    return El::mpi::AllReduce(x, El::mpi::SUM, El::mpi::COMM_WORLD)
           / El::mpi::Size(El::mpi::COMM_WORLD);
  }
}

int main(int argc, char **argv)
{
  Environment env(argc, argv);
  if(argc < 4)
    {
      if(El::mpi::Rank() == 0)
        std::fprintf(stderr,
                     "usage: qchol_bench N precision method[,...] [reps]\n");
      return 1;
    }
  const int64_t n = std::stoll(argv[1]);
  const int64_t precision = std::stoll(argv[2]);
  std::vector<std::string> methods;
  {
    std::stringstream ss(argv[3]);
    std::string m;
    while(std::getline(ss, m, ','))
      methods.push_back(m);
  }
  const int reps = argc > 4 ? std::stoi(argv[4]) : 1;
  Environment::set_precision(precision);
  const int rank = El::mpi::Rank(), size = El::mpi::Size();

  El::Grid grid(El::mpi::COMM_WORLD);
  El::DistMatrix<El::BigFloat> Q0(n, n, grid);
  double t = now();
  for(int64_t jl = 0; jl < Q0.LocalWidth(); ++jl)
    for(int64_t il = 0; il < Q0.LocalHeight(); ++il)
      Q0.SetLocal(il, jl, entry(Q0.GlobalRow(il), Q0.GlobalCol(jl), n));
  El::mpi::Barrier();
  if(rank == 0)
    std::printf("N %lld precision %lld ranks %d grid %dx%d fill %.1f s\n",
                (long long)n, (long long)precision, size, grid.Height(),
                grid.Width(), now() - t);

  El::DistMatrix<El::BigFloat> b0(n, 1, grid);
  for(int64_t il = 0; il < b0.LocalHeight(); ++il)
    for(int64_t jl = 0; jl < b0.LocalWidth(); ++jl)
      b0.SetLocal(il, jl, entry(b0.GlobalRow(il) + n, 0, 3 * n));

  El::DistMatrix<El::BigFloat> U_el(grid);
  bool have_el = false;
  for(const auto &method : methods)
    for(int r = 0; r < reps; ++r)
      {
        El::DistMatrix<El::BigFloat> U(Q0);
        Multimodular_Blas_Config config;
        Multimodular_Cholesky_Dist_Timings mt;
        El::mpi::Barrier();
        t = now();
        if(method == "el")
          El::Cholesky(El::UPPER, U);
        else if(method.rfind("mmd", 0) == 0)
          multimodular_cholesky_upper_dist(U, std::stoi(method.substr(3)),
                                           true, config, mt);
        else if(method.rfind("mm", 0) == 0)
          multimodular_cholesky_upper_dist(U, std::stoi(method.substr(2)),
                                           false, config, mt);
        else
          {
            if(rank == 0)
              std::printf("unknown method %s\n", method.c_str());
            continue;
          }
        const double t_own = now() - t;
        El::mpi::Barrier();
        const double t_chol = now() - t;

        // one Q solve, as SDPB's computeSearchDirection
        El::DistMatrix<El::BigFloat> b(b0);
        El::mpi::Barrier();
        t = now();
        El::cholesky::SolveAfter(El::UPPER, El::NORMAL, U, b);
        El::mpi::Barrier();
        const double t_solve = now() - t;

        // difference from El's factor (upper triangle)
        double rel = -1;
        if(method == "el" && !have_el)
          {
            U_el = U;
            have_el = true;
          }
        else if(have_el)
          {
            El::BigFloat dmax(0), amax(0);
            for(int64_t jl = 0; jl < U.LocalWidth(); ++jl)
              for(int64_t il = 0; il < U.LocalHeight(); ++il)
                {
                  if(U.GlobalRow(il) > U.GlobalCol(jl))
                    continue;
                  const El::BigFloat a = U_el.GetLocal(il, jl);
                  El::BigFloat d = U.GetLocal(il, jl) - a;
                  if(d < 0)
                    d = -d;
                  const El::BigFloat aa = a < 0 ? El::BigFloat(-a) : a;
                  if(d > dmax)
                    dmax = d;
                  if(aa > amax)
                    amax = aa;
                }
            dmax = El::mpi::AllReduce(dmax, El::mpi::MAX, El::mpi::COMM_WORLD);
            amax = El::mpi::AllReduce(amax, El::mpi::MAX, El::mpi::COMM_WORLD);
            rel = amax > 0 ? double(El::Log(dmax / amax) / El::Log(El::BigFloat(2)))
                           : 0;
          }
        const double own_max = max_over_ranks(t_own),
                     diag = max_over_ranks(mt.diag),
                     panel = max_over_ranks(mt.panel),
                     redist = max_over_ranks(mt.redist),
                     update_max = max_over_ranks(mt.update),
                     update_mean = mean_over_ranks(mt.update),
                     blas = mean_over_ranks(mt.mm.blas_seconds),
                     to_int = mean_over_ranks(mt.mm.to_int_seconds),
                     from_int = mean_over_ranks(mt.mm.from_int_seconds);
        if(rank == 0)
          std::printf(
            "RESULT N %lld ranks %d method %s rep %d cholesky %.3f "
            "own_max %.3f diag %.3f panel %.3f redist %.3f update_max %.3f "
            "update_mean %.3f [blas %.3f to_int %.3f from_int %.3f] "
            "solve %.3f log2_rel_diff_vs_el %.1f\n",
            (long long)n, size, method.c_str(), r, t_chol, own_max, diag,
            panel, redist, update_max, update_mean, blas, to_int, from_int,
            t_solve, rel);
        std::fflush(stdout);
      }
  return 0;
}
