// lab (agent) #235: see lab235.hxx.

// #239: mpfr.h before any FLINT header (arf_get_mpfr is declared only then)
#include <gmp.h>
#include <mpfr.h>

#include "lab235.hxx"

#include "sdpb_util/copy_matrix.hxx"
#include "sdp_solve/SDP_Solver/run/step/compute_search_direction/lower_triangular_solve.hxx"
#include "sdp_solve/lower_triangular_transpose_solve.hxx"

#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdpb_util/flint.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"

#include <flint/arb.h>
#include <flint/arf.h>
#include <flint/fmpz.h>

#include <algorithm>
#include <climits>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

// compute_Q.cxx: Q = P^T P (#239's on-demand refactor)
void syrk_Q(const Environment &env, Block_Matrix &schur_off_diagonal,
            BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
            El::DistMatrix<El::BigFloat> &Q, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity);

namespace lab235
{
  namespace
  {
    using Clock = std::chrono::steady_clock;
    double seconds_since(const Clock::time_point &t0)
    {
      return std::chrono::duration<double>(Clock::now() - t0).count();
    }

    std::vector<int> parse_ints(const char *s, std::vector<int> def)
    {
      if(s == nullptr)
        return def;
      std::vector<int> out;
      std::stringstream ss(s);
      std::string item;
      while(std::getline(ss, item, ','))
        if(!item.empty())
          out.push_back(std::stoi(item));
      return out;
    }
    int env_int(const char *name, int def)
    {
      const char *s = std::getenv(name);
      return s ? std::atoi(s) : def;
    }
    double env_double(const char *name, double def)
    {
      const char *s = std::getenv(name);
      return s ? std::atof(s) : def;
    }

    bool is_root() { return El::mpi::Rank(El::mpi::COMM_WORLD) == 0; }

    std::string fmt(double x)
    {
      std::ostringstream os;
      if(std::isfinite(x))
        os << std::setprecision(6) << x;
      else
        os << "null";
      return os.str();
    }
    std::string fmt_vec(const std::vector<double> &v)
    {
      std::ostringstream os;
      os << "[";
      for(size_t i = 0; i < v.size(); ++i)
        os << (i ? "," : "") << fmt(v[i]);
      os << "]";
      return os.str();
    }

    void write_line(const std::string &line)
    {
      auto &st = state();
      if(is_root() && st.log.is_open())
        {
          st.log << line << "\n";
          st.log.flush();
        }
    }

    El::BigFloat max_abs(const El::DistMatrix<El::BigFloat> &v)
    {
      El::BigFloat local(0);
      for(int64_t i = 0; i < v.LocalHeight(); ++i)
        for(int64_t j = 0; j < v.LocalWidth(); ++j)
          local = std::max(local, El::Abs(v.GetLocal(i, j)));
      return El::mpi::AllReduce(local, El::mpi::MAX, El::mpi::COMM_WORLD);
    }

    // Emulated storage precision: x := the sum of nd doubles closest (by
    // successive truncation) to x, i.e. ~53*nd bits (nd = 2: double-double,
    // 4: quad-double). Values outside double's exponent range are left alone
    // and counted.
    long round_out_of_range = 0;
    void round_bf(El::BigFloat &x, int nd)
    {
      if(nd <= 0 || x == El::BigFloat(0))
        return;
      long e = 0;
      mpf_get_d_2exp(&e, x.gmp_float.get_mpf_t());
      if(e > 1000 || e < -960)
        {
          ++round_out_of_range;
          return;
        }
      El::BigFloat y(x), acc(0);
      for(int i = 0; i < nd; ++i)
        {
          double d = double(y);
          if(d == 0.0)
            break;
          El::BigFloat bd(d);
          acc += bd;
          y -= bd;
        }
      x = acc;
    }
    void round_mat(El::DistMatrix<El::BigFloat> &A, int nd)
    {
      if(nd <= 0)
        return;
      for(int64_t i = 0; i < A.LocalHeight(); ++i)
        for(int64_t j = 0; j < A.LocalWidth(); ++j)
          {
            El::BigFloat v = A.GetLocal(i, j);
            round_bf(v, nd);
            A.SetLocal(i, j, v);
          }
    }

    El::BigFloat dot(const El::DistMatrix<El::BigFloat> &a,
                     const El::DistMatrix<El::BigFloat> &b)
    {
      return El::Dotu(a, b);
    }

    // #239 (step 3): the matvec in low precision. P's local entries are
    // rounded once per iteration to arb midpoints of LAB235_MATVEC_BITS bits
    // (row-major); both products are arb_approx_dot at that precision. For a
    // block spread over several ranks, P_j v's partial sums (over the rank's
    // columns) are summed over the block's row communicator in BigFloat.
    void arf_from_mpf(arf_t y, mpf_srcptr x, slong limbs)
    {
      const slong size = x->_mp_size;
      if(size == 0)
        {
          arf_zero(y);
          return;
        }
      // an mpf need not be normalized: skip zero top limbs (arf_set_mpn
      // scans for a nonzero limb without a bound)
      slong n = size < 0 ? -size : size;
      slong e = x->_mp_exp;
      while(n > 0 && x->_mp_d[n - 1] == 0)
        {
          --n;
          --e;
        }
      if(n == 0)
        {
          arf_zero(y);
          return;
        }
      const slong k = std::min(n, limbs);
      arf_set_mpn(y, x->_mp_d + (n - k), k, size < 0);
      arf_mul_2exp_si(y, y, slong(FLINT_BITS) * (e - k));
    }
    void bf_from_arf(El::BigFloat &y, const arf_t x, mpfr_t tmp)
    {
      arf_get_mpfr(tmp, x, MPFR_RNDN);
      mpfr_get_f(y.gmp_float.get_mpf_t(), tmp, MPFR_RNDN);
    }
    struct Low_Block
    {
      slong h = 0, w = 0;
      arb_ptr rm = nullptr; // h x w, row-major
      std::vector<int64_t> gcol;
      El::mpi::Comm row_comm;
      int row_comm_size = 1;
    };
    struct Low_P
    {
      int bits = 0;
      int iteration = -1;
      int run = -1;
      std::vector<Low_Block> blocks;
      double t_build = 0;
      void clear()
      {
        for(auto &b : blocks)
          if(b.rm != nullptr)
            _arb_vec_clear(b.rm, b.h * b.w);
        blocks.clear();
      }
    };
    Low_P &low_p()
    {
      static Low_P *lp = new Low_P;
      return *lp;
    }
    bool low_active = false; // apply_Q dispatches to apply_Q_low

    void build_low(const Block_Matrix &P, int bits, int iteration, int run)
    {
      auto &lp = low_p();
      if(lp.bits == bits && lp.iteration == iteration && lp.run == run)
        return;
      auto t0 = std::chrono::steady_clock::now();
      lp.clear();
      lp.bits = bits;
      lp.iteration = iteration;
      lp.run = run;
      const slong limbs = (bits + FLINT_BITS - 1) / FLINT_BITS + 1;
      lp.blocks.resize(P.blocks.size());
      for(size_t b = 0; b < P.blocks.size(); ++b)
        {
          const auto &Pb = P.blocks[b];
          auto &lb = lp.blocks[b];
          lb.h = Pb.LocalHeight();
          lb.w = Pb.LocalWidth();
          lb.rm = _arb_vec_init(lb.h * lb.w);
          for(slong i = 0; i < lb.h; ++i)
            for(slong k = 0; k < lb.w; ++k)
              arf_from_mpf(arb_midref(lb.rm + i * lb.w + k),
                           Pb.GetLocalCRef(i, k).gmp_float.get_mpf_t(), limbs);
          lb.gcol.resize(lb.w);
          for(slong k = 0; k < lb.w; ++k)
            lb.gcol[k] = Pb.GlobalCol(k);
          lb.row_comm = Pb.RowComm();
          lb.row_comm_size = El::mpi::Size(lb.row_comm);
        }
      lp.t_build
        = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    }

    void apply_Q_low(const El::DistMatrix<El::BigFloat> &v,
                     El::DistMatrix<El::BigFloat> &out)
    {
      auto &lp = low_p();
      const slong prec = lp.bits;
      const slong limbs = (prec + FLINT_BITS - 1) / FLINT_BITS + 1;
      const int64_t N = v.Height();
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> v_local(v);
      arb_ptr va = _arb_vec_init(N);
      for(int64_t k = 0; k < N; ++k)
        arf_from_mpf(arb_midref(va + k),
                     v_local.GetLocalCRef(k, 0).gmp_float.get_mpf_t(), limbs);
      El::Matrix<El::BigFloat> sum;
      El::Zeros(sum, N, 1);
      mpfr_t tmp;
      mpfr_init2(tmp, prec + 64);
      arb_t t;
      arb_init(t);
      for(auto &lb : lp.blocks)
        {
          if(lb.h == 0 || lb.w == 0)
            continue;
          arb_ptr vl = _arb_vec_init(lb.w);
          for(slong k = 0; k < lb.w; ++k)
            arb_set(vl + k, va + lb.gcol[k]);
          arb_ptr wl = _arb_vec_init(lb.h);
          for(slong i = 0; i < lb.h; ++i)
            arb_approx_dot(wl + i, nullptr, 0, lb.rm + i * lb.w, 1, vl, 1,
                           lb.w, prec);
          if(lb.row_comm_size > 1)
            {
              std::vector<El::BigFloat> buf(lb.h);
              for(slong i = 0; i < lb.h; ++i)
                bf_from_arf(buf[i], arb_midref(wl + i), tmp);
              El::mpi::AllReduce(buf.data(), int(lb.h), El::mpi::SUM,
                                 lb.row_comm);
              for(slong i = 0; i < lb.h; ++i)
                {
                  arf_from_mpf(arb_midref(wl + i),
                               buf[i].gmp_float.get_mpf_t(), limbs);
                  mag_zero(arb_radref(wl + i));
                }
            }
          for(slong k = 0; k < lb.w; ++k)
            {
              arb_approx_dot(t, nullptr, 0, lb.rm + k, lb.w, wl, 1, lb.h,
                             prec);
              El::BigFloat y;
              bf_from_arf(y, arb_midref(t), tmp);
              sum(lb.gcol[k], 0) += y;
            }
          _arb_vec_clear(wl, lb.h);
          _arb_vec_clear(vl, lb.w);
        }
      arb_clear(t);
      mpfr_clear(tmp);
      _arb_vec_clear(va, N);
      El::Zeros(out, N, 1);
      El::BigFloat zero(0);
      for(int64_t row = 0; row < N; ++row)
        if(sum(row, 0) != zero)
          out.QueueUpdate(row, 0, sum(row, 0));
      out.ProcessQueues();
    }

    // Q_new v = sum_j P_j^T (P_j v), P_j = L_j^{-1} B_j (SDPB's schur_off_diagonal)
    void apply_Q(const Block_Matrix &P, Block_Vector &work_N,
                 Block_Vector &work_P, const El::DistMatrix<El::BigFloat> &v,
                 El::DistMatrix<El::BigFloat> &out)
    {
      if(low_active)
        {
          apply_Q_low(v, out);
          return;
        }
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> v_local(v);
      for(size_t b = 0; b < P.blocks.size(); ++b)
        {
          copy_matrix(v_local, work_N.blocks[b]);
          El::Gemv(El::OrientationNS::NORMAL, El::BigFloat(1), P.blocks[b],
                   work_N.blocks[b], El::BigFloat(0), work_P.blocks[b]);
          El::Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(1),
                   P.blocks[b], work_P.blocks[b], El::BigFloat(0),
                   work_N.blocks[b]);
        }
      sum_blocks(work_N, out);
    }

    // Q_new v = sum_j B_j^T L_j^{-T} L_j^{-1} B_j v without forming P_j = L_j^{-1} B_j
    // (what an inexact SDPB would do: it skips SDPB's L^{-1} B stage too)
    void apply_Q_BL(const SDP &sdp, const Block_Diagonal_Matrix &L,
                    Block_Vector &work_N, Block_Vector &work_P,
                    const El::DistMatrix<El::BigFloat> &v,
                    El::DistMatrix<El::BigFloat> &out)
    {
      El::DistMatrix<El::BigFloat, El::STAR, El::STAR> v_local(v);
      El::DistMatrix<El::BigFloat> scratch;
      for(size_t b = 0; b < work_P.blocks.size(); ++b)
        {
          const auto &Bj = sdp.free_var_block(b, scratch);
          copy_matrix(v_local, work_N.blocks[b]);
          El::Gemv(El::OrientationNS::NORMAL, El::BigFloat(1), Bj,
                   work_N.blocks[b], El::BigFloat(0), work_P.blocks[b]);
        }
      lower_triangular_solve(L, work_P);
      lower_triangular_transpose_solve(L, work_P);
      for(size_t b = 0; b < work_P.blocks.size(); ++b)
        {
          const auto &Bj = sdp.free_var_block(b, scratch);
          El::Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(1), Bj,
                   work_P.blocks[b], El::BigFloat(0), work_N.blocks[b]);
        }
      sum_blocks(work_N, out);
    }

    // eigenvalues (ascending) of a symmetric BigFloat DistMatrix rounded to double;
    // returned on the grid's root only
    std::vector<double>
    eig_double_root(const El::DistMatrix<El::BigFloat> &G, bool &is_grid_root)
    {
      El::DistMatrix<double> Gd(G.Grid());
      Gd.AlignWith(G);
      El::Zeros(Gd, G.Height(), G.Width());
      for(int64_t i = 0; i < G.LocalHeight(); ++i)
        for(int64_t j = 0; j < G.LocalWidth(); ++j)
          Gd.SetLocal(i, j, double(G.GetLocal(i, j)));
      El::DistMatrix<double, El::CIRC, El::CIRC> Gc(Gd);
      std::vector<double> out;
      is_grid_root = (Gc.CrossRank() == Gc.Root());
      if(is_grid_root)
        {
          El::Matrix<double> A(Gc.Matrix());
          // symmetrize (rounding can leave tiny asymmetries)
          for(int64_t i = 0; i < A.Height(); ++i)
            for(int64_t j = 0; j < i; ++j)
              {
                double m = 0.5 * (A(i, j) + A(j, i));
                A(i, j) = m;
                A(j, i) = m;
              }
          El::Matrix<double> w;
          El::HermitianEig(El::UpperOrLowerNS::LOWER, A, w);
          out.resize(w.Height());
          for(int64_t i = 0; i < w.Height(); ++i)
            out[i] = w(i, 0);
          std::sort(out.begin(), out.end());
        }
      return out;
    }

    std::vector<double> summary(const std::vector<double> &w)
    {
      // n, min, q01, q10, q50, q90, q99, max
      std::vector<double> s(8, 0.0);
      if(w.empty())
        return s;
      auto q = [&](double f) {
        size_t i = std::min(w.size() - 1, size_t(f * (w.size() - 1) + 0.5));
        return w[i];
      };
      s[0] = w.size();
      s[1] = w.front();
      s[2] = q(0.01);
      s[3] = q(0.10);
      s[4] = q(0.50);
      s[5] = q(0.90);
      s[6] = q(0.99);
      s[7] = w.back();
      return s;
    }

    bool want_spectra(int it)
    {
      auto &st = state();
      return st.spectra_iters.empty()
             || std::find(st.spectra_iters.begin(), st.spectra_iters.end(), it)
                  != st.spectra_iters.end();
    }

    void Q_spectrum(const El::DistMatrix<El::BigFloat> &Q_chol,
                    const El::DistMatrix<El::BigFloat> &U_old, int s)
    {
      auto &st = state();
      auto t0 = Clock::now();
      // W = U_new U_old^{-1}; preconditioned operator U_old^{-T} Q_new U_old^{-1} = W^T W
      El::DistMatrix<El::BigFloat> W(Q_chol);
      El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, W);
      El::Trsm(El::LeftOrRightNS::RIGHT, El::UpperOrLowerNS::UPPER,
               El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
               El::BigFloat(1), U_old, W);
      El::DistMatrix<El::BigFloat> G(W.Grid());
      El::Zeros(G, W.Width(), W.Width());
      El::Gemm(El::OrientationNS::TRANSPOSE, El::OrientationNS::NORMAL,
               El::BigFloat(1), W, W, El::BigFloat(0), G);
      double t_big = seconds_since(t0);
      bool root = false;
      auto w = eig_double_root(G, root);
      if(is_root())
        {
          auto sm = summary(w);
          std::ostringstream os;
          os << "{\"kind\":\"specQ\",\"it\":" << st.iteration << ",\"stale\":" << s
             << ",\"n\":" << sm[0] << ",\"min\":" << fmt(sm[1])
             << ",\"q01\":" << fmt(sm[2]) << ",\"q10\":" << fmt(sm[3])
             << ",\"q50\":" << fmt(sm[4]) << ",\"q90\":" << fmt(sm[5])
             << ",\"q99\":" << fmt(sm[6]) << ",\"max\":" << fmt(sm[7])
             << ",\"cond\":" << fmt(sm[7] / sm[1])
             << ",\"t_bigfloat\":" << fmt(t_big)
             << ",\"t_total\":" << fmt(seconds_since(t0)) << "}";
          write_line(os.str());
          std::ofstream f(st.dir + "/eig-Q-it" + std::to_string(st.iteration)
                          + "-s" + std::to_string(s) + ".txt");
          f << std::setprecision(17);
          for(double x : w)
            f << x << "\n";
        }
    }

    void S_spectrum(const Block_Info &block_info,
                    const Block_Diagonal_Matrix &L_new,
                    const Block_Diagonal_Matrix &L_old, int s)
    {
      auto &st = state();
      auto t0 = Clock::now();
      const size_t num_blocks = block_info.dimensions.size();
      const size_t width = 8;
      std::vector<double> stats(num_blocks * width, 0.0);
      for(size_t b = 0; b < L_new.blocks.size(); ++b)
        {
          const size_t gb = block_info.block_indices.at(b);
          // V = L_old^{-1} L_new; preconditioned operator L_old^{-1} S_new L_old^{-T} = V V^T
          El::DistMatrix<El::BigFloat> V(L_new.blocks[b]);
          El::MakeTrapezoidal(El::UpperOrLowerNS::LOWER, V);
          El::Trsm(El::LeftOrRightNS::LEFT, El::UpperOrLowerNS::LOWER,
                   El::OrientationNS::NORMAL, El::UnitOrNonUnitNS::NON_UNIT,
                   El::BigFloat(1), L_old.blocks[b], V);
          El::DistMatrix<El::BigFloat> M(V.Grid());
          El::Zeros(M, V.Height(), V.Height());
          El::Gemm(El::OrientationNS::NORMAL, El::OrientationNS::TRANSPOSE,
                   El::BigFloat(1), V, V, El::BigFloat(0), M);
          bool root = false;
          auto w = eig_double_root(M, root);
          if(root)
            {
              auto sm = summary(w);
              for(size_t k = 0; k < width; ++k)
                stats[gb * width + k] = sm[k];
            }
        }
      El::mpi::Reduce(stats.data(), stats.size(), El::mpi::SUM, 0,
                      El::mpi::COMM_WORLD);
      if(is_root())
        {
          // worst block by condition number
          double worst = 0;
          size_t worst_b = 0;
          std::vector<double> mins(num_blocks), maxs(num_blocks);
          for(size_t gb = 0; gb < num_blocks; ++gb)
            {
              mins[gb] = stats[gb * width + 1];
              maxs[gb] = stats[gb * width + 7];
              double c = maxs[gb] / mins[gb];
              if(c > worst)
                {
                  worst = c;
                  worst_b = gb;
                }
            }
          std::ostringstream os;
          os << "{\"kind\":\"specS\",\"it\":" << st.iteration << ",\"stale\":" << s
             << ",\"worst_block\":" << worst_b << ",\"worst_cond\":" << fmt(worst)
             << ",\"global_min\":" << fmt(*std::min_element(mins.begin(), mins.end()))
             << ",\"global_max\":" << fmt(*std::max_element(maxs.begin(), maxs.end()))
             << ",\"min\":" << fmt_vec(mins) << ",\"max\":" << fmt_vec(maxs)
             << ",\"t\":" << fmt(seconds_since(t0)) << "}";
          write_line(os.str());
        }
    }

    struct PCG_Result
    {
      int iterations = 0;
      bool converged = false;
      bool stopped_on_abs = false;
    };

    // PCG on Q_new x = rhs with preconditioner U_old^T U_old.
    // x_exact: SDPB's solution (for errors). If stop_abs > 0, stop at
    // |r|_inf <= stop_abs and return x there.
    PCG_Result pcg(const Block_Matrix &P, Block_Vector &work_N,
                   Block_Vector &work_P, const El::DistMatrix<El::BigFloat> &U_old,
                   const El::DistMatrix<El::BigFloat> &rhs,
                   const El::DistMatrix<El::BigFloat> *x_exact,
                   const El::DistMatrix<El::BigFloat> *x0,
                   const El::BigFloat &stop_abs, El::DistMatrix<El::BigFloat> &x,
                   const std::string &label, int s, const El::BigFloat &p_norm,
                   int nd = 0, const El::DistMatrix<El::BigFloat> *Uinv = nullptr)
    {
      // preconditioner z = (U^T U)^{-1} r: two triangular solves, or with the explicit
      // inverse two matrix-vector products z = U^-1 (U^-T r)
      auto precondition = [&](const El::DistMatrix<El::BigFloat> &rr,
                              El::DistMatrix<El::BigFloat> &zz) {
        if(Uinv != nullptr)
          {
            El::DistMatrix<El::BigFloat> tt(rr.Grid());
            El::Zeros(tt, rr.Height(), 1);
            El::Zeros(zz, rr.Height(), 1);
            El::Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(1), *Uinv, rr,
                     El::BigFloat(0), tt);
            El::Gemv(El::OrientationNS::NORMAL, El::BigFloat(1), *Uinv, tt,
                     El::BigFloat(0), zz);
          }
        else
          {
            zz = rr;
            El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                                     El::OrientationNS::NORMAL, U_old, zz);
          }
      };
      auto &st = state();
      auto t0 = Clock::now();
      double t_mv = 0, t_pc = 0;
      int n_mv = 0, n_pc = 0;
      const El::BigFloat rhs_norm = max_abs(rhs);
      // x_exact: SDPB's exact solution, for the error log (nullptr in #239's
      // inexact mode, where there is none)
      const El::BigFloat xs_norm
        = x_exact != nullptr ? max_abs(*x_exact) : El::BigFloat(0);
      const int prec = El::gmp::Precision();
      const El::BigFloat floor_rel
        = El::BigFloat(1) >> std::max(1, prec - st.pcg_floor_bits);

      El::DistMatrix<El::BigFloat> r(rhs), z(rhs.Grid()), p(rhs.Grid()),
        q(rhs.Grid()), d(rhs.Grid());
      round_mat(r, nd);
      const long oor0 = round_out_of_range;
      El::Zeros(q, rhs.Height(), 1);
      if(x0 != nullptr)
        {
          x = *x0;
          auto t1 = Clock::now();
          apply_Q(P, work_N, work_P, x, q);
          t_mv += seconds_since(t1);
          ++n_mv;
          round_mat(q, nd);
          El::Axpy(El::BigFloat(-1), q, r);
          round_mat(r, nd);
        }
      else
        {
          El::Zeros(x, rhs.Height(), 1);
        }
      std::vector<double> lres, lerr, labs, alphas, betas;
      auto record = [&]() {
        El::BigFloat rn = max_abs(r);
        lres.push_back(log10_abs(rn) - log10_abs(rhs_norm));
        labs.push_back(log10_abs(rn));
        if(x_exact != nullptr)
          {
            d = x;
            El::Axpy(El::BigFloat(-1), *x_exact, d);
            El::BigFloat en = max_abs(d);
            lerr.push_back(log10_abs(en) - log10_abs(xs_norm));
          }
        return rn;
      };
      El::BigFloat rn = record();
      {
        auto t1 = Clock::now();
        precondition(r, z);
        t_pc += seconds_since(t1);
        ++n_pc;
        round_mat(z, nd);
      }
      p = z;
      El::BigFloat rz = dot(r, z);
      PCG_Result res;
      bool stopped_on_abs = false;
      for(int k = 1; k <= st.pcg_max_iters; ++k)
        {
          if(stop_abs > 0 && rn <= stop_abs)
            {
              stopped_on_abs = true;
              break;
            }
          if(rn <= floor_rel * rhs_norm)
            {
              res.converged = true;
              break;
            }
          auto t1 = Clock::now();
          apply_Q(P, work_N, work_P, p, q);
          t_mv += seconds_since(t1);
          ++n_mv;
          round_mat(q, nd);
          El::BigFloat pq = dot(p, q);
          El::BigFloat alpha = rz / pq;
          round_bf(alpha, nd);
          El::Axpy(alpha, p, x);
          El::Axpy(-alpha, q, r);
          round_mat(x, nd);
          round_mat(r, nd);
          rn = record();
          res.iterations = k;
          t1 = Clock::now();
          precondition(r, z);
          t_pc += seconds_since(t1);
          ++n_pc;
          round_mat(z, nd);
          El::BigFloat rz_new = dot(r, z);
          El::BigFloat beta = rz_new / rz;
          round_bf(beta, nd);
          rz = rz_new;
          // p = z + beta p
          El::Scale(beta, p);
          El::Axpy(El::BigFloat(1), z, p);
          round_mat(p, nd);
          alphas.push_back(double(alpha));
          betas.push_back(double(beta));
        }
      if(rn <= floor_rel * rhs_norm)
        res.converged = true;
      if(stop_abs > 0 && rn <= stop_abs)
        stopped_on_abs = true;
      res.stopped_on_abs = stopped_on_abs;
      // true residual
      apply_Q(P, work_N, work_P, x, q);
      ++n_mv;
      d = rhs;
      El::Axpy(El::BigFloat(-1), q, d);
      El::BigFloat true_res = max_abs(d);
      if(is_root())
        {
          std::ostringstream os;
          os << "{\"kind\":\"pcg\",\"mode\":\"" << label << "\",\"it\":" << st.iteration
             << ",\"phase\":" << st.phase << ",\"stale\":" << s
             << ",\"iters\":" << res.iterations
             << ",\"converged\":" << (res.converged ? "true" : "false")
             << ",\"stopped_on_abs\":" << (stopped_on_abs ? "true" : "false")
             << ",\"warm\":" << (x0 != nullptr ? "true" : "false")
             << ",\"log10_rhs\":" << fmt(log10_abs(rhs_norm))
             << ",\"log10_p\":" << fmt(log10_abs(p_norm))
             << ",\"log10_primal_error\":" << fmt(log10_abs(st.primal_error))
             << ",\"log10_stop_abs\":" << fmt(stop_abs > 0 ? log10_abs(stop_abs) : -1e9)
             << ",\"log10_xs\":" << fmt(log10_abs(xs_norm))
             << ",\"log10_true_res_rel\":"
             << fmt(log10_abs(true_res) - log10_abs(rhs_norm))
             << ",\"prec\":" << prec << ",\"doubles\":" << nd
             << ",\"out_of_range\":" << (round_out_of_range - oor0)
             << ",\"t\":" << fmt(seconds_since(t0))
             << ",\"t_matvec\":" << fmt(t_mv) << ",\"n_matvec\":" << n_mv
             << ",\"t_prec\":" << fmt(t_pc) << ",\"n_prec\":" << n_pc
             << ",\"lres\":" << fmt_vec(lres) << ",\"labs\":" << fmt_vec(labs)
             << ",\"lerr\":" << fmt_vec(lerr) << ",\"alpha\":" << fmt_vec(alphas)
             << ",\"beta\":" << fmt_vec(betas) << "}";
          write_line(os.str());
        }
      return res;
    }
  }

  namespace
  {
    // #243, defined with the v2 code below
    void refactor_done(const El::DistMatrix<El::BigFloat> &Q_chol);
  }

  State &state()
  {
    // never destroyed: its BigFloat matrices must not be freed after El::Finalize
    static State *st = new State;
    return *st;
  }

  double log10_abs(const El::BigFloat &x)
  {
    if(x == El::BigFloat(0))
      return -1e9;
    long e = 0;
    double d = mpf_get_d_2exp(&e, x.gmp_float.get_mpf_t());
    return std::log10(std::fabs(d)) + double(e) * std::log10(2.0);
  }

  void init_from_env()
  {
    auto &st = state();
    if(st.initialized)
      return;
    st.initialized = true;
    const char *dir = std::getenv("LAB235_DIR");
    if(dir == nullptr || std::string(dir).empty())
      return;
    st.enabled = true;
    st.dir = dir;
    st.stale_Q = parse_ints(std::getenv("LAB235_STALE_Q"), {1});
    st.stale_S = parse_ints(std::getenv("LAB235_STALE_S"), {1});
    st.stale_pcg = parse_ints(std::getenv("LAB235_STALE_PCG"), {1, 2, 4});
    st.spectra_iters = parse_ints(std::getenv("LAB235_SPECTRA_ITERS"), {});
    st.pcg_iters = parse_ints(std::getenv("LAB235_PCG_ITERS"), st.spectra_iters);
    st.pcg_phases = parse_ints(std::getenv("LAB235_PCG_PHASES"), {0, 1});
    st.use_max_iters = env_int("LAB235_USE_MAX", 300);
    st.pcg_doubles = parse_ints(std::getenv("LAB235_PCG_DOUBLES"), {});
    st.pcg_doubles_stale = env_int("LAB235_PCG_DOUBLES_STALE", 1);
    st.check_BL = env_int("LAB235_CHECK_BL", 1) != 0;
    st.prec_inv = env_int("LAB235_PREC_INV", 0) != 0;
    st.self_control = env_int("LAB235_SELF", 1) != 0;
    st.pcg_max_iters = env_int("LAB235_PCG_MAX", 300);
    st.pcg_floor_bits = env_int("LAB235_FLOOR_BITS", 64);
    st.warm_corrector = env_int("LAB235_WARM", 0) != 0;
    st.use_period = env_int("LAB235_USE_PERIOD", 0);
    st.use_eta = env_double("LAB235_USE_ETA", 1e-3);
    st.use_abs_log10 = env_double("LAB235_USE_ABS_LOG10", -1e9);
    st.skip_runs = env_int("LAB235_SKIP_RUNS", 0);
    st.use_thr_factor = env_double("LAB235_USE_THR", 1.0);
    st.use_rel = env_double("LAB235_USE_REL", 0.0);
    // #239 inexact solver mode
    st.inexact = env_int("LAB235_INEXACT", 0) != 0;
    st.refactor_steps = env_int("LAB235_REFACTOR_STEPS", 12);
    st.rel_min = env_double("LAB235_REL_MIN", 1e-8);
    st.rel_gamma = env_double("LAB235_REL_GAMMA", 0.1);
    st.matvec_bits = env_int("LAB235_MATVEC_BITS", 0);
    st.matvec_check = env_int("LAB235_MATVEC_CHECK", 0);
    if(st.inexact)
      {
        // only the last refactor's factor and inverse are kept; no measurements
        st.prec_inv = true;
        st.stale_Q.clear();
        st.stale_S.clear();
        st.stale_pcg.clear();
        st.pcg_doubles.clear();
        st.self_control = false;
        st.check_BL = false;
        // #243 (v2): defaults for cheap 256-bit steps (#239's plans set
        // these explicitly; they are now the defaults)
        st.matvec_bits = env_int("LAB235_MATVEC_BITS", 256);
        if(st.matvec_bits <= 0)
          st.matvec_bits = 256; // v2's PCG is arb arithmetic
        st.use_period = env_int("LAB235_USE_PERIOD", 16);
        if(st.use_period < 1)
          st.use_period = 16;
        st.refactor_steps = env_int("LAB235_REFACTOR_STEPS", 60);
        st.use_max_iters = env_int("LAB235_USE_MAX", 100);
        st.use_rel = env_double("LAB235_USE_REL", 1e-3);
        st.use_eta = env_double("LAB235_USE_ETA", 0);
        st.use_thr_factor = env_double("LAB235_USE_THR", 0);
        st.warm_corrector = env_int("LAB235_WARM", 1) != 0;
        const char *rule = std::getenv("LAB235_REFACTOR_RULE");
        st.refactor_rule
          = (rule != nullptr && std::string(rule) == "steps") ? 0 : 1;
      }
    int keep = 1;
    for(auto v : {st.stale_Q, st.stale_S, st.stale_pcg})
      for(int s : v)
        keep = std::max(keep, s);
    keep = std::max(keep, st.use_period - 1);
    st.keep = st.inexact ? 1 : env_int("LAB235_KEEP", keep);
    if(is_root())
      {
        st.log.open(st.dir + "/lab235.jsonl", std::ios::app);
        std::ostringstream os;
        os << "{\"kind\":\"config\",\"keep\":" << st.keep
           << ",\"pcg_max\":" << st.pcg_max_iters
           << ",\"floor_bits\":" << st.pcg_floor_bits
           << ",\"warm\":" << st.warm_corrector
           << ",\"use_period\":" << st.use_period
           << ",\"use_eta\":" << fmt(st.use_eta)
           << ",\"use_abs_log10\":" << fmt(st.use_abs_log10)
           << ",\"use_rel\":" << fmt(st.use_rel)
           << ",\"inexact\":" << st.inexact
           << ",\"use_max\":" << st.use_max_iters
           << ",\"refactor_steps\":" << st.refactor_steps
           << ",\"rel_min\":" << fmt(st.rel_min)
           << ",\"rel_gamma\":" << fmt(st.rel_gamma)
           << ",\"matvec_bits\":" << st.matvec_bits
           << ",\"matvec_check\":" << st.matvec_check
           << ",\"version\":2"
           << ",\"refactor_rule\":\"" << (st.refactor_rule == 1 ? "cost" : "steps") << "\""
           << ",\"prec\":" << El::gmp::Precision()
           << ",\"ranks\":" << El::mpi::Size(El::mpi::COMM_WORLD) << "}";
        write_line(os.str());
      }
  }

  bool enabled()
  {
    init_from_env();
    return state().active;
  }

  void begin_run()
  {
    init_from_env();
    auto &st = state();
    if(!st.enabled)
      return;
    ++st.run;
    st.active = st.run > st.skip_runs;
    st.iteration = 0;
    st.self_done = false;
    st.perr_ref_set = false;
    st.last_refactor = 0;
    st.need_refactor = false;
    // the previous run's matrices may live on a grid that no longer exists: leak them
    (new std::deque<El::DistMatrix<El::BigFloat>>)->swap(st.Q_hist);
    (new std::deque<Block_Diagonal_Matrix>)->swap(st.S_hist);
    (new std::deque<El::DistMatrix<El::BigFloat>>)->swap(st.Qinv_hist);
    (new std::vector<El::DistMatrix<El::BigFloat>>)->swap(st.warm);
    std::ostringstream os;
    os << "{\"kind\":\"run\",\"run\":" << st.run
       << ",\"active\":" << (st.active ? "true" : "false") << "}";
    write_line(os.str());
  }

  void begin_step(const El::BigFloat &primal_error,
                  const El::BigFloat &primal_error_threshold,
                  const El::BigFloat &primal_error_P)
  {
    if(!enabled())
      return;
    auto &st = state();
    ++st.iteration;
    st.primal_error = primal_error;
    st.primal_threshold = primal_error_threshold;
    st.phase = 0;
    st.warm.clear();
    if(!st.inexact)
      return;
    // forcing: rel_k = clamp(use_rel x (Perr_k / Perr_1)^gamma, rel_min, use_rel)
    st.perr_log10 = log10_abs(primal_error_P);
    if(!st.perr_ref_set)
      {
        st.perr_ref_log10 = st.perr_log10;
        st.perr_ref_set = true;
      }
    double lrel = std::log10(st.use_rel)
                  + st.rel_gamma * (st.perr_log10 - st.perr_ref_log10);
    lrel = std::min(lrel, std::log10(st.use_rel));
    lrel = std::max(lrel, std::log10(st.rel_min));
    st.rel_now = std::pow(10.0, lrel);
    // refactor or not. #243: rank 0's decision (the cost rule uses its
    // timings) is broadcast, so every rank takes the same branch
    int code = st.need_refactor ? (st.refactor_reason == "cost" ? 2 : 1) : 0;
    El::mpi::Broadcast(&code, 1, 0, El::mpi::COMM_WORLD);
    st.need_refactor = code != 0;
    std::string reason;
    if(st.Q_hist.empty() || st.Qinv_hist.empty())
      reason = "first";
    else if(st.need_refactor)
      reason = code == 2 ? "cost" : "slow";
    else if(st.iteration - st.last_refactor >= st.use_period)
      reason = "period";
    st.refactor_reason = reason;
    st.skip_Q = reason.empty();
    st.Q_formed = !st.skip_Q;
    st.need_refactor = false;
    st.fallback = false;
    st.steps_now.clear();
    st.t_pcg_now = 0;
    st.t_inv_now = 0;
    st.t_sum_now = 0;
    st.t_rows_U = 0;
    st.inverse_done = false;
    st.refactor_reason = reason;
    st.t_iter0 = std::chrono::duration<double>(
                   Clock::now().time_since_epoch())
                   .count();
    if(st.Q_formed)
      st.last_refactor = st.iteration;
  }

  void set_step_context(const Environment &env,
                        BigInt_Shared_Memory_Syrk_Context &syrk_ctx,
                        Timers &timers, El::Matrix<int32_t> &block_timings_ms,
                        Verbosity verbosity, El::DistMatrix<El::BigFloat> &Q,
                        Block_Matrix &P)
  {
    if(!enabled())
      return;
    auto &st = state();
    st.env = &env;
    st.syrk_ctx = &syrk_ctx;
    st.timers = &timers;
    st.block_timings_ms = &block_timings_ms;
    st.verbosity = verbosity;
    st.Q = &Q;
    st.P = &P;
  }

  bool skip_Q()
  {
    if(!enabled())
      return false;
    const auto &st = state();
    return st.inexact && st.skip_Q;
  }

  const El::DistMatrix<El::BigFloat> &
  cond_factor(const El::DistMatrix<El::BigFloat> &Q)
  {
    if(skip_Q() && !state().Q_hist.empty())
      return state().Q_hist.front();
    return Q;
  }

  void after_factor(const SDP &sdp, const Block_Info &block_info,
                    const Block_Diagonal_Matrix &schur_complement_cholesky,
                    const El::DistMatrix<El::BigFloat> &Q_chol)
  {
    if(!enabled())
      return;
    auto &st = state();
    st.sdp = &sdp;
    st.L_now = &schur_complement_cholesky;
    if(!want_spectra(st.iteration))
      return;
    if(st.self_control && !st.self_done && !st.Q_hist.empty())
      {
        // control: the new factor against itself must give exactly 1
        El::DistMatrix<El::BigFloat> U(Q_chol);
        El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, U);
        Q_spectrum(Q_chol, U, 0);
        Block_Diagonal_Matrix L(schur_complement_cholesky);
        for(auto &blk : L.blocks)
          El::MakeTrapezoidal(El::UpperOrLowerNS::LOWER, blk);
        S_spectrum(block_info, schur_complement_cholesky, L, 0);
        st.self_done = true;
      }
    for(int s : st.stale_Q)
      if(s >= 1 && size_t(s) <= st.Q_hist.size())
        Q_spectrum(Q_chol, st.Q_hist[s - 1], s);
    for(int s : st.stale_S)
      if(s >= 1 && size_t(s) <= st.S_hist.size())
        S_spectrum(block_info, schur_complement_cholesky, st.S_hist[s - 1],
                   s);
  }

  void end_step(const Block_Diagonal_Matrix &schur_complement_cholesky,
                const El::DistMatrix<El::BigFloat> &Q_chol)
  {
    if(!enabled())
      return;
    auto &st = state();
    if(st.inexact)
      {
        // #243: U^-1 is formed in after_cholesky (or by the fallback)
        if(st.Q_formed && !st.inverse_done)
          refactor_done(Q_chol);
        double T = std::chrono::duration<double>(
                     Clock::now().time_since_epoch())
                     .count()
                   - st.t_iter0;
        double avg = -1;
        if(is_root())
          {
            // the cost rule (rank 0's clock; broadcast in begin_step)
            if(st.Q_formed)
              {
                if(!st.fallback)
                  st.t_ref_last = T;
                st.cycle_pcg = 0;
              }
            else
              {
                st.base_est = T - st.t_pcg_now;
                st.cycle_pcg += st.t_pcg_now;
                if(st.refactor_rule == 1 && st.t_ref_last > 0)
                  {
                    const int stale = st.iteration - st.last_refactor;
                    st.dR_now = std::max(0.0, st.t_ref_last - st.base_est);
                    avg = (st.dR_now + st.cycle_pcg) / (stale + 1);
                    if(!st.need_refactor && st.t_pcg_now >= avg)
                      {
                        st.need_refactor = true;
                        st.refactor_reason = "cost";
                      }
                  }
              }
          }
        if(is_root())
          {
            std::ostringstream os;
            os << "{\"kind\":\"istep\",\"it\":" << st.iteration
               << ",\"refactor\":" << (st.Q_formed ? "true" : "false")
               << ",\"reason\":\"" << (st.fallback ? "fallback" : st.refactor_reason)
               << "\",\"stale\":" << (st.Q_formed ? 0 : st.iteration - st.last_refactor)
               << ",\"rel\":" << fmt(st.rel_now)
               << ",\"log10_Perr\":" << fmt(st.perr_log10)
               << ",\"steps\":[";
            for(size_t i = 0; i < st.steps_now.size(); ++i)
              os << (i ? "," : "") << st.steps_now[i];
            os << "],\"t_pcg\":" << fmt(st.t_pcg_now)
               << ",\"t_inv\":" << fmt(st.t_inv_now)
               << ",\"t_rows_U\":" << fmt(st.t_rows_U)
               << ",\"t_sums\":" << fmt(st.t_sum_now)
               << ",\"t_iter\":" << fmt(T)
               << ",\"dR\":" << fmt(st.dR_now)
               << ",\"cycle_pcg\":" << fmt(st.cycle_pcg)
               << ",\"avg\":" << fmt(avg)
               << ",\"next_refactor\":" << (st.need_refactor ? "true" : "false")
               << ",\"refactors\":" << st.refactors
               << ",\"fallbacks\":" << st.fallbacks
               << ",\"pcg_solves\":" << st.pcg_solves
               << ",\"pcg_steps\":" << st.pcg_steps << "}";
            write_line(os.str());
          }
        return;
      }
    // Deep copies constructed in place: moving an El::DistMatrix into the deque
    // left it pointing at freed memory (smoke test: iteration 3 "equal" to 1).
    st.Q_hist.emplace_front(Q_chol);
    El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, st.Q_hist.front());
    while(st.Q_hist.size() > st.keep)
      st.Q_hist.pop_back();
    if(st.prec_inv)
      {
        auto t0 = Clock::now();
        st.Qinv_hist.emplace_front(st.Q_hist.front());
        El::TriangularInverse(El::UpperOrLowerNS::UPPER,
                              El::UnitOrNonUnitNS::NON_UNIT, st.Qinv_hist.front());
        El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, st.Qinv_hist.front());
        while(st.Qinv_hist.size() > st.keep)
          st.Qinv_hist.pop_back();
        double t = seconds_since(t0);
        if(is_root())
          write_line("{\"kind\":\"triinv\",\"it\":" + std::to_string(st.iteration)
                     + ",\"t\":" + fmt(t) + "}");
      }
    st.S_hist.emplace_front(schur_complement_cholesky);
    for(auto &blk : st.S_hist.front().blocks)
      El::MakeTrapezoidal(El::UpperOrLowerNS::LOWER, blk);
    while(st.S_hist.size() > st.keep)
      st.S_hist.pop_back();
  }

  void sum_blocks(const Block_Vector &v, El::DistMatrix<El::BigFloat> &out)
  {
    const int64_t height = out.Height();
    El::Matrix<El::BigFloat> sum;
    El::Zeros(sum, height, 1);
    for(const auto &blk : v.blocks)
      for(int64_t row = 0; row < blk.LocalHeight(); ++row)
        {
          int64_t global_row(blk.GlobalRow(row));
          for(int64_t column = 0; column < blk.LocalWidth(); ++column)
            sum(global_row, blk.GlobalCol(column)) += blk.GetLocal(row, column);
        }
    El::Zeros(out, height, 1);
    El::BigFloat zero(0);
    for(int64_t row = 0; row < height; ++row)
      if(sum(row, 0) != zero)
        out.QueueUpdate(row, 0, sum(row, 0));
    out.ProcessQueues();
  }

  void solve_hook(const Block_Matrix &P, Block_Vector &work_N,
                  const Block_Vector &work_P_template,
                  const El::DistMatrix<El::BigFloat> &Q_chol,
                  const El::DistMatrix<El::BigFloat> &rhs,
                  const El::DistMatrix<El::BigFloat> &p,
                  El::DistMatrix<El::BigFloat> &dy_dist)
  {
    if(!enabled())
      return;
    auto &st = state();
    if(st.inexact) // #239: refactor iterations of the inexact mode solve exactly
      return;
    Block_Vector work_P(work_P_template);
    const El::DistMatrix<El::BigFloat> x_exact(dy_dist);
    const El::BigFloat p_norm = max_abs(p);
    if(st.warm.empty())
      st.warm.resize(st.keep + 1);
    // measurement runs
    const bool measure_here
      = (st.pcg_iters.empty()
         || std::find(st.pcg_iters.begin(), st.pcg_iters.end(), st.iteration)
              != st.pcg_iters.end())
        && std::find(st.pcg_phases.begin(), st.pcg_phases.end(), st.phase)
             != st.pcg_phases.end();
    for(int s : st.stale_pcg)
      {
        if(!measure_here || s < 1 || size_t(s) > st.Q_hist.size())
          continue;
        El::DistMatrix<El::BigFloat> x(rhs.Grid());
        const El::DistMatrix<El::BigFloat> *x0 = nullptr;
        if(st.warm_corrector && st.phase == 1 && st.warm[s].Height() > 0)
          x0 = &st.warm[s];
        pcg(P, work_N, work_P, st.Q_hist[s - 1], rhs, &x_exact, x0,
            El::BigFloat(0), x, "measure", s, p_norm);
        if(st.phase == 0)
          st.warm[s] = x;
      }
    // stale-1 PCG with the explicit-inverse preconditioner (same operator, other application)
    if(measure_here && st.prec_inv && !st.Qinv_hist.empty())
      {
        El::DistMatrix<El::BigFloat> x(rhs.Grid());
        pcg(P, work_N, work_P, st.Q_hist[0], rhs, &x_exact, nullptr,
            El::BigFloat(0), x, "measure-inv", 1, p_norm, 0, &st.Qinv_hist[0]);
      }
    // the two matvec forms on v = rhs: agreement and time
    if(measure_here && st.check_BL && st.sdp != nullptr && st.L_now != nullptr)
      {
        El::DistMatrix<El::BigFloat> q1(rhs.Grid()), q2(rhs.Grid()), dq(rhs.Grid());
        El::Zeros(q1, rhs.Height(), 1);
        El::Zeros(q2, rhs.Height(), 1);
        El::mpi::Barrier(El::mpi::COMM_WORLD);
        auto t1 = Clock::now();
        apply_Q(P, work_N, work_P, rhs, q1);
        El::mpi::Barrier(El::mpi::COMM_WORLD);
        double tP = seconds_since(t1);
        t1 = Clock::now();
        apply_Q_BL(*st.sdp, *st.L_now, work_N, work_P, rhs, q2);
        El::mpi::Barrier(El::mpi::COMM_WORLD);
        double tBL = seconds_since(t1);
        dq = q1;
        El::Axpy(El::BigFloat(-1), q2, dq);
        double rel = log10_abs(max_abs(dq)) - log10_abs(max_abs(q1));
        if(is_root())
          {
            std::ostringstream os;
            os << "{\"kind\":\"matvec\",\"it\":" << st.iteration << ",\"phase\":" << st.phase
               << ",\"t_P\":" << fmt(tP) << ",\"t_BL\":" << fmt(tBL)
               << ",\"log10_rel_diff\":" << fmt(rel) << "}";
            write_line(os.str());
          }
      }
    // reduced-precision runs: P, U_old, rhs and every PCG vector rounded to nd doubles
    if(measure_here && !st.pcg_doubles.empty()
       && st.pcg_doubles_stale >= 1
       && size_t(st.pcg_doubles_stale) <= st.Q_hist.size())
      {
        const int s = st.pcg_doubles_stale;
        for(int nd : st.pcg_doubles)
          {
            if(nd <= 0)
              continue;
            const long oor0 = round_out_of_range;
            Block_Matrix Pr(P);
            for(auto &blk : Pr.blocks)
              round_mat(blk, nd);
            El::DistMatrix<El::BigFloat> Ur(st.Q_hist[s - 1]);
            round_mat(Ur, nd);
            El::DistMatrix<El::BigFloat> rhs_r(rhs);
            round_mat(rhs_r, nd);
            const long oor_inputs = round_out_of_range - oor0;
            if(is_root())
              write_line("{\"kind\":\"round_inputs\",\"it\":"
                         + std::to_string(st.iteration) + ",\"doubles\":"
                         + std::to_string(nd) + ",\"out_of_range_local\":"
                         + std::to_string(oor_inputs) + "}");
            El::DistMatrix<El::BigFloat> x(rhs.Grid());
            pcg(Pr, work_N, work_P, Ur, rhs_r, &x_exact, nullptr,
                El::BigFloat(0), x, "doubles", s, p_norm, nd);
          }
      }
    // control: PCG preconditioned by the exact current factor must converge in 1 step
    if(st.self_control && st.iteration <= 2 && st.phase == 0)
      {
        El::DistMatrix<El::BigFloat> x(rhs.Grid());
        int saved = st.pcg_max_iters;
        st.pcg_max_iters = 3;
        pcg(P, work_N, work_P, Q_chol, rhs, &x_exact, nullptr, El::BigFloat(0),
            x, "self", 0, p_norm);
        st.pcg_max_iters = saved;
      }
    // use mode: refactor every use_period iterations, PCG in between
    if(st.use_period > 1)
      {
        int s = (st.iteration - 1) % st.use_period;
        if(s >= 1 && size_t(s) <= st.Q_hist.size())
          {
            // stop when |r|_inf <= eta * max(|p|_inf, abs floor)
            El::BigFloat floor_abs(0);
            if(st.use_abs_log10 > -1e8)
              {
                double e = st.use_abs_log10 * std::log2(10.0);
                floor_abs = El::BigFloat(1);
                long ie = long(std::floor(e));
                if(ie >= 0)
                  floor_abs <<= ie;
                else
                  floor_abs >>= -ie;
              }
            // SDPB's primal error is max(|P|, |p|); the CG residual adds to p
            El::BigFloat floor_thr
              = El::BigFloat(st.use_thr_factor) * st.primal_threshold;
            El::BigFloat stop_abs
              = El::BigFloat(st.use_eta)
                * std::max(std::max(st.primal_error, p_norm),
                           std::max(floor_abs, floor_thr));
            // a relative forcing term (standard inexact Newton): |r| <= use_rel |rhs|
            if(st.use_rel > 0)
              stop_abs = std::max(stop_abs, El::BigFloat(st.use_rel) * max_abs(rhs));
            El::DistMatrix<El::BigFloat> x(rhs.Grid());
            const El::DistMatrix<El::BigFloat> *x0 = nullptr;
            if(st.warm_corrector && st.phase == 1 && st.warm[0].Height() > 0)
              x0 = &st.warm[0];
            int saved = st.pcg_max_iters;
            st.pcg_max_iters = st.use_max_iters;
            const El::DistMatrix<El::BigFloat> *Uinv
              = (st.prec_inv && size_t(s) <= st.Qinv_hist.size())
                  ? &st.Qinv_hist[s - 1]
                  : nullptr;
            auto res = pcg(P, work_N, work_P, st.Q_hist[s - 1], rhs, &x_exact,
                           x0, stop_abs, x, "use", s, p_norm, 0, Uinv);
            st.pcg_max_iters = saved;
            if(st.phase == 0)
              st.warm[0] = x;
            // not reached within pcg_max_iters: keep SDPB's exact dy (logged)
            if(res.stopped_on_abs)
              dy_dist = x;
            else if(is_root())
              write_line("{\"kind\":\"use_fallback\",\"it\":"
                         + std::to_string(st.iteration) + ",\"phase\":"
                         + std::to_string(st.phase) + "}");
          }
      }
  }
  // ---------------------------------------------------------------------
  // #243: v2 of the inexact mode (see lab235.hxx)
  namespace
  {
    constexpr int DIGIT_BITS = 48; // int64 digits: sums of < 2^15 ranks fit

    // bits [pos, pos + nb) of |m| (nb <= 64)
    uint64_t mpz_bits(const mpz_t m, slong pos, int nb)
    {
      const slong li = pos / 64, off = pos % 64;
      uint64_t v = uint64_t(mpz_getlimbn(m, li)) >> off;
      if(off + nb > 64)
        v |= uint64_t(mpz_getlimbn(m, li + 1)) << (64 - off);
      return nb == 64 ? v : (v & ((uint64_t(1) << nb) - 1));
    }

    // out[j] = sum over the ranks of COMM_WORLD of v[j], exact once each
    // rank's v[j] is rounded to a multiple of 2^(E_j - bits), E_j the
    // smallest e with |v[j]| < 2^e on every rank (per-entry fixed point,
    // signed int64 digits of 48 bits, MPI_SUM). Integer sums do not depend
    // on the reduction order, so every rank gets the same out. out may be v.
    double t_exact_sum = 0;
    void exact_sum(arb_srcptr v, slong n, slong bits, arb_ptr out,
                   MPI_Comm comm = MPI_COMM_WORLD)
    {
      const auto t0 = std::chrono::steady_clock::now();
      std::vector<int> e(n);
      for(slong j = 0; j < n; ++j)
        e[j] = arf_is_zero(arb_midref(v + j))
                 ? INT_MIN
                 : int(arf_abs_bound_lt_2exp_si(arb_midref(v + j)));
      MPI_Allreduce(MPI_IN_PLACE, e.data(), int(n), MPI_INT, MPI_MAX, comm);
      const slong K = bits / DIGIT_BITS + 1;
      std::vector<int64_t> d(size_t(n * K), 0);
      fmpz_t z;
      fmpz_init(z);
      mpz_t m;
      mpz_init(m);
      for(slong j = 0; j < n; ++j)
        {
          if(e[j] == INT_MIN || arf_is_zero(arb_midref(v + j)))
            continue;
          // z ~ v[j] / 2^(E_j - bits), |z| <= 2^bits
          arf_get_fmpz_fixed_si(z, arb_midref(v + j), slong(e[j]) - bits);
          fmpz_get_mpz(m, z);
          const int sgn = mpz_sgn(m);
          mpz_abs(m, m);
          for(slong k = 0; k < K; ++k)
            d[j * K + k] = sgn * int64_t(mpz_bits(m, k * DIGIT_BITS, DIGIT_BITS));
        }
      MPI_Allreduce(MPI_IN_PLACE, d.data(), int(n * K), MPI_INT64_T, MPI_SUM,
                    comm);
      for(slong j = 0; j < n; ++j)
        {
          if(e[j] == INT_MIN)
            {
              arb_zero(out + j);
              continue;
            }
          fmpz_zero(z);
          for(slong k = K - 1; k >= 0; --k)
            {
              fmpz_mul_2exp(z, z, DIGIT_BITS);
              const int64_t dk = d[j * K + k];
              if(dk >= 0)
                fmpz_add_ui(z, z, ulong(dk));
              else
                fmpz_sub_ui(z, z, ulong(-dk));
            }
          arf_set_fmpz(arb_midref(out + j), z);
          arf_mul_2exp_si(arb_midref(out + j), arb_midref(out + j),
                          slong(e[j]) - bits);
          mag_zero(arb_radref(out + j));
        }
      mpz_clear(m);
      fmpz_clear(z);
      t_exact_sum += std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - t0)
                       .count();
    }

    // rows of a [VC,STAR] matrix held by this rank, rounded to arb
    // midpoints (truncated to bits / 64 + 1 limbs), row-major, all columns
    struct Rows
    {
      slong h = 0, w = 0;
      arb_ptr rm = nullptr;
      std::vector<int64_t> grow;
      void clear()
      {
        if(rm != nullptr)
          _arb_vec_clear(rm, h * w);
        rm = nullptr;
        h = w = 0;
        grow.clear();
      }
    };
    void rows_from(const El::DistMatrix<El::BigFloat, El::VC, El::STAR> &A,
                   int bits, Rows &out)
    {
      out.clear();
      out.h = A.LocalHeight();
      out.w = A.Width();
      if(out.h == 0 || out.w == 0)
        {
          out.h = 0;
          return;
        }
      const slong limbs = (bits + FLINT_BITS - 1) / FLINT_BITS + 1;
      out.rm = _arb_vec_init(out.h * out.w);
      out.grow.resize(out.h);
      for(slong i = 0; i < out.h; ++i)
        {
          out.grow[i] = A.GlobalRow(i);
          for(slong k = 0; k < out.w; ++k)
            arf_from_mpf(arb_midref(out.rm + i * out.w + k),
                         A.GetLocalCRef(i, k).gmp_float.get_mpf_t(), limbs);
        }
    }

    struct V2
    {
      int p_iteration = -1, p_run = -1;
      std::vector<Rows> P; // per local block
      Rows U;              // rows of U^-1
      slong N = 0;
      arb_ptr warm = nullptr; // predictor's solution (corrector's start)
      bool warm_set = false;
    };
    V2 &v2()
    {
      static V2 *p = new V2;
      return *p;
    }

    void build_P_rows(const Block_Matrix &P, int bits)
    {
      auto &st = state();
      auto &w = v2();
      if(w.p_iteration == st.iteration && w.p_run == st.run)
        return;
      const auto t0 = Clock::now();
      for(auto &r : w.P)
        r.clear();
      w.P.resize(P.blocks.size());
      for(size_t b = 0; b < P.blocks.size(); ++b)
        {
          // collective on the block's grid (all its ranks hold the same
          // blocks, in the same order)
          El::DistMatrix<El::BigFloat, El::VC, El::STAR> rows(P.blocks[b]);
          rows_from(rows, bits, w.P[b]);
        }
      w.p_iteration = st.iteration;
      w.p_run = st.run;
      st.t_rows_P = seconds_since(t0);
    }

    // out = Q v = sum_j P_j^T (P_j v), v and out the same on every rank
    double t_matvec_local = 0;
    void matvec_v2(arb_srcptr v, arb_ptr out, slong N, slong prec)
    {
      const auto t0 = Clock::now();
      auto &w = v2();
      arb_ptr part = _arb_vec_init(N);
      arb_t t;
      arb_init(t);
      for(auto &R : w.P)
        {
          if(R.h == 0)
            continue;
          arb_ptr y = _arb_vec_init(R.h);
          for(slong i = 0; i < R.h; ++i)
            arb_approx_dot(y + i, nullptr, 0, R.rm + i * R.w, 1, v, 1, R.w,
                           prec);
          for(slong k = 0; k < R.w; ++k)
            {
              arb_approx_dot(t, part + k, 0, R.rm + k, R.w, y, 1, R.h, prec);
              arb_swap(t, part + k);
            }
          _arb_vec_clear(y, R.h);
        }
      arb_clear(t);
      t_matvec_local += seconds_since(t0);
      exact_sum(part, N, prec + 32, out);
      _arb_vec_clear(part, N);
    }

    // z = U^-1 U^-T r (U^-1 upper), r and z the same on every rank
    void precondition_v2(arb_srcptr r, arb_ptr z, slong N, slong prec)
    {
      auto &U = v2().U;
      arb_ptr part = _arb_vec_init(N);
      // t = U^-T r: t_k = sum_{i <= k} (U^-1)_{ik} r_i
      for(slong ii = 0; ii < U.h; ++ii)
        {
          const slong i = U.grow[ii];
          if(!arf_is_zero(arb_midref(r + i)))
            _arb_vec_scalar_addmul(part + i, U.rm + ii * U.w + i, N - i,
                                   r + i, prec);
        }
      for(slong k = 0; k < N; ++k)
        mag_zero(arb_radref(part + k));
      arb_ptr t = _arb_vec_init(N);
      exact_sum(part, N, prec + 32, t);
      // z_i = sum_{k >= i} (U^-1)_{ik} t_k, this rank's rows
      _arb_vec_zero(part, N);
      for(slong ii = 0; ii < U.h; ++ii)
        {
          const slong i = U.grow[ii];
          arb_approx_dot(part + i, nullptr, 0, U.rm + ii * U.w + i, 1, t + i,
                         1, N - i, prec);
        }
      exact_sum(part, N, prec + 32, z);
      _arb_vec_clear(t, N);
      _arb_vec_clear(part, N);
    }

    void arb_vec_from_bf(const El::Matrix<El::BigFloat> &a, arb_ptr out)
    {
      for(int64_t i = 0; i < a.Height(); ++i)
        {
          arf_from_mpf(arb_midref(out + i),
                       a.CRef(i, 0).gmp_float.get_mpf_t(), 1 << 20);
          mag_zero(arb_radref(out + i));
        }
    }
    void bf_from_arb_vec(arb_srcptr a, El::Matrix<El::BigFloat> &out)
    {
      mpfr_t tmp;
      mpfr_init2(tmp, El::gmp::Precision() + 64);
      for(int64_t i = 0; i < out.Height(); ++i)
        bf_from_arf(out(i, 0), arb_midref(a + i), tmp);
      mpfr_clear(tmp);
    }
    void max_abs_arb(arb_srcptr a, slong n, arf_t m)
    {
      arf_zero(m);
      for(slong i = 0; i < n; ++i)
        if(arf_cmpabs(arb_midref(a + i), m) > 0)
          arf_abs(m, arb_midref(a + i));
    }
    double log10_arf(const arf_t x)
    {
      if(arf_is_zero(x))
        return -1e9;
      return arf_get_d(x, ARF_RND_NEAR) == 0
               ? double(arf_abs_bound_lt_2exp_si(x)) * std::log10(2.0)
               : std::log10(std::fabs(arf_get_d(x, ARF_RND_NEAR)));
    }

    // U (Q = U^T U) just factored on its grid: keep it (cond numbers) and
    // form U^-1 and its rows
    void refactor_done(const El::DistMatrix<El::BigFloat> &Q_chol)
    {
      auto &st = state();
      auto t0 = Clock::now();
      Scoped_Timer inv_timer(*st.timers, "lab_triangular_inverse");
      st.Q_hist.clear();
      st.Qinv_hist.clear();
      st.Q_hist.emplace_front(Q_chol);
      if(Q_chol.Participating())
        El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, st.Q_hist.front());
      st.Qinv_hist.emplace_front(st.Q_hist.front());
      if(Q_chol.Participating())
        {
          El::TriangularInverse(El::UpperOrLowerNS::UPPER,
                                El::UnitOrNonUnitNS::NON_UNIT,
                                st.Qinv_hist.front());
          El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, st.Qinv_hist.front());
        }
      st.t_inv_now = seconds_since(t0);
      auto t1 = Clock::now();
      {
        Scoped_Timer rows_timer(*st.timers, "lab_Uinv_rows");
        v2().U.clear();
        if(Q_chol.Participating())
          {
            El::DistMatrix<El::BigFloat, El::VC, El::STAR> rows(
              st.Qinv_hist.front());
            rows_from(rows, st.matvec_bits, v2().U);
          }
      }
      st.t_rows_U = seconds_since(t1);
      st.inverse_done = true;
      ++st.refactors;
    }
  }

  namespace
  {
    // #243 check (LAB235_MATVEC_CHECK=2): Q v = sum_j B_j^T S_j^-1 B_j v at
    // `prec` bits without P_j = L_j^-1 B_j: rows of B_j (cached per run),
    // the full L_j on every rank of the block (per iteration), B_j v summed
    // exactly over the block's ranks, two triangular solves, B_j^T w.
    std::vector<Rows> B_rows;
    int B_rows_run = -1;
    void matvec_BL(const SDP &sdp, const Block_Diagonal_Matrix &L,
                   arb_srcptr v, arb_ptr out, slong N, slong prec)
    {
      auto &st = state();
      if(B_rows_run != st.run)
        {
          for(auto &r : B_rows)
            r.clear();
          B_rows.assign(L.blocks.size(), Rows());
          for(size_t b = 0; b < L.blocks.size(); ++b)
            {
              El::DistMatrix<El::BigFloat> scratch;
              const auto &Bj = sdp.free_var_block(b, scratch);
              El::DistMatrix<El::BigFloat, El::VC, El::STAR> rows(Bj);
              rows_from(rows, int(prec), B_rows[b]);
            }
          B_rows_run = st.run;
        }
      arb_ptr part = _arb_vec_init(N);
      arb_t t;
      arb_init(t);
      const slong limbs = (prec + FLINT_BITS - 1) / FLINT_BITS + 1;
      for(size_t b = 0; b < L.blocks.size(); ++b)
        {
          const auto &Lb = L.blocks[b];
          El::DistMatrix<El::BigFloat, El::STAR, El::STAR> Lss(Lb);
          const slong m = Lss.Height();
          arb_ptr La = _arb_vec_init(m * m);
          for(slong i = 0; i < m; ++i)
            for(slong k = 0; k <= i; ++k)
              arf_from_mpf(arb_midref(La + i * m + k),
                           Lss.GetLocalCRef(i, k).gmp_float.get_mpf_t(), limbs);
          const Rows &BR = B_rows[b];
          arb_ptr y = _arb_vec_init(m);
          for(slong ii = 0; ii < BR.h; ++ii)
            arb_approx_dot(y + BR.grow[ii], nullptr, 0, BR.rm + ii * BR.w, 1,
                           v, 1, BR.w, prec);
          exact_sum(y, m, prec + 32, y, Lb.Grid().Comm().comm);
          // L u = y, then L^T w = u (w in y)
          for(slong i = 0; i < m; ++i)
            {
              arb_approx_dot(t, y + i, 1, La + i * m, 1, y, 1, i, prec);
              arb_div(y + i, t, La + i * m + i, prec);
              mag_zero(arb_radref(y + i));
            }
          for(slong i = m - 1; i >= 0; --i)
            {
              arb_approx_dot(t, y + i, 1, La + (i + 1) * m + i, m, y + i + 1,
                             1, m - 1 - i, prec);
              arb_div(y + i, t, La + i * m + i, prec);
              mag_zero(arb_radref(y + i));
            }
          for(slong ii = 0; ii < BR.h; ++ii)
            _arb_vec_scalar_addmul(part, BR.rm + ii * BR.w, BR.w,
                                   y + BR.grow[ii], prec);
          _arb_vec_clear(y, m);
          _arb_vec_clear(La, m * m);
        }
      arb_clear(t);
      for(slong k = 0; k < N; ++k)
        mag_zero(arb_radref(part + k));
      exact_sum(part, N, prec + 32, out);
      _arb_vec_clear(part, N);
    }
  }

  void set_cholesky_panel(int64_t panel)
  {
    if(enabled())
      state().mm_panel = panel;
  }

  void after_cholesky(const El::DistMatrix<El::BigFloat> &Q_chol)
  {
    if(!enabled())
      return;
    auto &st = state();
    if(!st.inexact || !st.Q_formed)
      return;
    refactor_done(Q_chol);
  }

  void inexact_solve_v2(const Block_Matrix &P, El::Matrix<El::BigFloat> &dy)
  {
    auto &st = state();
    auto &w = v2();
    auto t0 = Clock::now();
    const slong N = dy.Height();
    const slong prec = st.matvec_bits;
    const slong wprec = El::gmp::Precision();
    if(w.N != N)
      {
        if(w.warm != nullptr)
          _arb_vec_clear(w.warm, w.N);
        w.warm = _arb_vec_init(N);
        w.N = N;
        w.warm_set = false;
      }
    if(st.phase == 0)
      w.warm_set = false;
    t_exact_sum = 0;
    t_matvec_local = 0;
    // the right-hand side, summed exactly at working precision + 64 bits
    arb_ptr rhs = _arb_vec_init(N);
    arb_vec_from_bf(dy, rhs);
    exact_sum(rhs, N, wprec + 64, rhs);
    El::Matrix<El::BigFloat> rhs_bf(N, 1);
    bf_from_arb_vec(rhs, rhs_bf);
    PCG_Result res;
    double t_mv = 0, t_pc = 0;
    int n_mv = 0, n_pc = 0;
    arf_t rn, rhs_norm, stop;
    arf_init(rn);
    arf_init(rhs_norm);
    arf_init(stop);
    double lres_end = 0;
    {
      Scoped_Timer pcg_timer(*st.timers, "lab_pcg");
      {
        Scoped_Timer build_timer(*st.timers, "low_build");
        build_P_rows(P, st.matvec_bits);
      }
      arb_ptr x = _arb_vec_init(N), r = _arb_vec_init(N),
              z = _arb_vec_init(N), p = _arb_vec_init(N),
              q = _arb_vec_init(N);
      for(slong i = 0; i < N; ++i)
        arb_set_round(r + i, rhs + i, prec);
      max_abs_arb(rhs, N, rhs_norm);
      // |r|_inf <= max(rel_k |rhs|_inf, eta x max(primal error, thr floor))
      {
        El::BigFloat s
          = El::BigFloat(st.use_eta)
            * std::max(st.primal_error,
                       El::BigFloat(st.use_thr_factor) * st.primal_threshold);
        arf_from_mpf(stop, s.gmp_float.get_mpf_t(), 1 << 20);
        arf_t s2;
        arf_init(s2);
        arf_set_d(s2, st.rel_now);
        arf_mul(s2, s2, rhs_norm, 64, ARF_RND_NEAR);
        if(arf_cmp(s2, stop) > 0)
          arf_set(stop, s2);
        arf_clear(s2);
      }
      if(st.matvec_check && st.phase == 0)
        {
          // #243 check: the 256-bit matvec and preconditioner against
          // working precision on v = rhs (the matvec with P's rows at full
          // precision; the preconditioner by El::Gemv with U^-1 on its grid)
          const slong wp = wprec + 64;
          std::vector<Rows> saved;
          saved.swap(w.P);
          w.p_iteration = -1;
          build_P_rows(P, int(wprec + 64));
          arb_ptr q_full = _arb_vec_init(N), q_low = _arb_vec_init(N),
                  d = _arb_vec_init(N);
          auto tc = Clock::now();
          matvec_v2(rhs, q_full, N, wp);
          const double t_full = seconds_since(tc);
          for(auto &r : w.P)
            r.clear();
          w.P.swap(saved);
          w.p_iteration = -1;
          build_P_rows(P, st.matvec_bits);
          tc = Clock::now();
          matvec_v2(r, q_low, N, prec);
          const double t_low = seconds_since(tc);
          _arb_vec_sub(d, q_low, q_full, N, wp);
          arf_t a, b;
          arf_init(a);
          arf_init(b);
          max_abs_arb(d, N, a);
          max_abs_arb(q_full, N, b);
          const double mv_rel = log10_arf(a) - log10_arf(b);
          arf_t b0;
          arf_init(b0);
          arf_set(b0, b);
          // preconditioner
          arb_ptr z_low = _arb_vec_init(N);
          precondition_v2(r, z_low, N, prec);
          double pc_rel = 0;
          {
            const auto &Uinv = st.Qinv_hist.front();
            El::Matrix<El::BigFloat> r_bf(N, 1), z_bf(N, 1);
            bf_from_arb_vec(r, r_bf);
            El::DistMatrix<El::BigFloat> rr(Uinv.Grid()), tt(Uinv.Grid()),
              zz(Uinv.Grid());
            if(Uinv.Participating())
              {
                rr.Resize(N, 1);
                copy_matrix(r_bf, rr);
                El::Zeros(tt, N, 1);
                El::Zeros(zz, N, 1);
                El::Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(1), Uinv,
                         rr, El::BigFloat(0), tt);
                El::Gemv(El::OrientationNS::NORMAL, El::BigFloat(1), Uinv, tt,
                         El::BigFloat(0), zz);
                El::DistMatrix<El::BigFloat, El::STAR, El::STAR> zs(zz);
                z_bf = zs.Matrix();
              }
            // rank 0 is on every Q grid (#143): its z is the reference
            El::mpi::Broadcast(z_bf.Buffer(), int(N), 0, El::mpi::COMM_WORLD);
            arb_ptr z_full = _arb_vec_init(N);
            arb_vec_from_bf(z_bf, z_full);
            _arb_vec_sub(d, z_low, z_full, N, wp);
            max_abs_arb(d, N, a);
            max_abs_arb(z_full, N, b);
            pc_rel = log10_arf(a) - log10_arf(b);
            _arb_vec_clear(z_full, N);
          }
          double bl_rel = 0, t_bl = 0, t_p = 0;
          if(st.matvec_check >= 2 && st.sdp != nullptr && st.L_now != nullptr)
            {
              arb_ptr q_bl = _arb_vec_init(N);
              El::mpi::Barrier(El::mpi::COMM_WORLD);
              tc = Clock::now();
              matvec_BL(*st.sdp, *st.L_now, r, q_bl, N, prec);
              El::mpi::Barrier(El::mpi::COMM_WORLD);
              t_bl = seconds_since(tc);
              tc = Clock::now();
              matvec_v2(r, q_low, N, prec);
              El::mpi::Barrier(El::mpi::COMM_WORLD);
              t_p = seconds_since(tc);
              _arb_vec_sub(d, q_bl, q_full, N, wp);
              max_abs_arb(d, N, a);
              bl_rel = log10_arf(a) - log10_arf(b0);
              _arb_vec_clear(q_bl, N);
            }
          if(is_root())
            write_line("{\"kind\":\"check2\",\"it\":" + std::to_string(st.iteration)
                       + ",\"BL_log10_rel\":" + fmt(bl_rel) + ",\"t_BL\":" + fmt(t_bl)
                       + ",\"t_P\":" + fmt(t_p)
                       + ",\"bits\":" + std::to_string(prec)
                       + ",\"matvec_log10_rel\":" + fmt(mv_rel)
                       + ",\"prec_log10_rel\":" + fmt(pc_rel)
                       + ",\"t_full\":" + fmt(t_full) + ",\"t_low\":" + fmt(t_low) + "}");
          arf_clear(a);
          arf_clear(b);
          arf_clear(b0);
          _arb_vec_clear(z_low, N);
          _arb_vec_clear(q_full, N);
          _arb_vec_clear(q_low, N);
          _arb_vec_clear(d, N);
        }
      const bool warm = st.warm_corrector && st.phase == 1 && w.warm_set;
      if(warm)
        {
          _arb_vec_set(x, w.warm, N);
          auto t1 = Clock::now();
          matvec_v2(x, q, N, prec);
          t_mv += seconds_since(t1);
          ++n_mv;
          _arb_vec_sub(r, r, q, N, prec);
        }
      auto zero_radii = [N](arb_ptr a) {
        for(slong i = 0; i < N; ++i)
          mag_zero(arb_radref(a + i));
      };
      auto dot = [N, prec](arb_t d, arb_srcptr a, arb_srcptr b) {
        arb_approx_dot(d, nullptr, 0, a, 1, b, 1, N, prec);
      };
      {
        auto t1 = Clock::now();
        precondition_v2(r, z, N, prec);
        t_pc += seconds_since(t1);
        ++n_pc;
      }
      _arb_vec_set(p, z, N);
      arb_t rz, rz_new, pq, alpha, beta;
      arb_init(rz);
      arb_init(rz_new);
      arb_init(pq);
      arb_init(alpha);
      arb_init(beta);
      dot(rz, r, z);
      max_abs_arb(r, N, rn);
      bool stopped = false;
      for(int k = 1; k <= st.use_max_iters; ++k)
        {
          if(arf_cmp(rn, stop) <= 0)
            {
              stopped = true;
              break;
            }
          auto t1 = Clock::now();
          matvec_v2(p, q, N, prec);
          t_mv += seconds_since(t1);
          ++n_mv;
          dot(pq, p, q);
          arb_div(alpha, rz, pq, prec);
          mag_zero(arb_radref(alpha));
          _arb_vec_scalar_addmul(x, p, N, alpha, prec);
          arb_neg(alpha, alpha);
          _arb_vec_scalar_addmul(r, q, N, alpha, prec);
          zero_radii(x);
          zero_radii(r);
          max_abs_arb(r, N, rn);
          res.iterations = k;
          t1 = Clock::now();
          precondition_v2(r, z, N, prec);
          t_pc += seconds_since(t1);
          ++n_pc;
          dot(rz_new, r, z);
          arb_div(beta, rz_new, rz, prec);
          mag_zero(arb_radref(beta));
          arb_swap(rz, rz_new);
          // p = z + beta p
          _arb_vec_scalar_mul(p, p, N, beta, prec);
          _arb_vec_add(p, p, z, N, prec);
          zero_radii(p);
        }
      if(arf_cmp(rn, stop) <= 0)
        stopped = true;
      res.stopped_on_abs = stopped;
      res.converged = stopped;
      lres_end = log10_arf(rn) - log10_arf(rhs_norm);
      if(stopped)
        {
          bf_from_arb_vec(x, dy);
          if(st.phase == 0)
            {
              _arb_vec_set(w.warm, x, N);
              w.warm_set = true;
            }
        }
      arb_clear(rz);
      arb_clear(rz_new);
      arb_clear(pq);
      arb_clear(alpha);
      arb_clear(beta);
      _arb_vec_clear(x, N);
      _arb_vec_clear(r, N);
      _arb_vec_clear(z, N);
      _arb_vec_clear(p, N);
      _arb_vec_clear(q, N);
    }
    st.steps_now.push_back(res.iterations);
    ++st.pcg_solves;
    st.pcg_steps += res.iterations;
    if(res.iterations > st.refactor_steps)
      {
        st.need_refactor = true;
        st.refactor_reason = "slow";
      }
    if(is_root())
      {
        std::ostringstream os;
        os << "{\"kind\":\"pcg\",\"mode\":\"inexact2\",\"it\":" << st.iteration
           << ",\"phase\":" << st.phase
           << ",\"stale\":" << (st.iteration - st.last_refactor)
           << ",\"iters\":" << res.iterations
           << ",\"converged\":" << (res.converged ? "true" : "false")
           << ",\"log10_rhs\":" << fmt(log10_arf(rhs_norm))
           << ",\"log10_stop\":" << fmt(log10_arf(stop))
           << ",\"log10_res_rel\":" << fmt(lres_end)
           << ",\"rel\":" << fmt(st.rel_now) << ",\"bits\":" << prec
           << ",\"t\":" << fmt(seconds_since(t0)) << ",\"t_matvec\":" << fmt(t_mv)
           << ",\"t_matvec_local\":" << fmt(t_matvec_local)
           << ",\"n_matvec\":" << n_mv << ",\"t_prec\":" << fmt(t_pc)
           << ",\"n_prec\":" << n_pc << ",\"t_sums\":" << fmt(t_exact_sum)
           << ",\"t_rows_P\":" << fmt(st.t_rows_P) << "}";
        write_line(os.str());
      }
    arf_clear(rn);
    arf_clear(rhs_norm);
    arf_clear(stop);
    _arb_vec_clear(rhs, N);
    if(!res.converged)
      {
        // on-demand refactor: form and factor Q now (default grid), exact
        // solve for the rest of this iteration; the next iterations use the
        // new factor
        Scoped_Timer fallback_timer(*st.timers, "lab_fallback_refactor");
        syrk_Q(*st.env, *st.P, *st.syrk_ctx, *st.Q, *st.timers,
               *st.block_timings_ms, st.verbosity);
        {
          Scoped_Timer cholesky_timer(*st.timers, "Cholesky_Q");
          if(st.mm_panel > 0)
            {
              Multimodular_Blas_Config config;
              Multimodular_Cholesky_Dist_Timings t;
              multimodular_cholesky_upper_dist(*st.Q, int(st.mm_panel), false,
                                               config, t, st.timers);
            }
          else
            El::Cholesky(El::UpperOrLowerNS::UPPER, *st.Q);
        }
        st.skip_Q = false;
        st.Q_formed = true;
        st.fallback = true;
        st.need_refactor = false;
        st.last_refactor = st.iteration;
        ++st.fallbacks;
        refactor_done(*st.Q);
        El::DistMatrix<El::BigFloat> d(st.Q->Grid());
        d.Resize(N, 1);
        copy_matrix(rhs_bf, d);
        El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                                 El::OrientationNS::NORMAL, *st.Q, d);
        El::DistMatrix<El::BigFloat, El::STAR, El::STAR> d_ss(d);
        dy = d_ss.Matrix();
        w.warm_set = false;
        if(is_root())
          write_line("{\"kind\":\"fallback\",\"it\":"
                     + std::to_string(st.iteration) + ",\"phase\":"
                     + std::to_string(st.phase) + "}");
      }
    st.t_pcg_now += seconds_since(t0);
    st.t_sum_now += t_exact_sum;
  }
}
