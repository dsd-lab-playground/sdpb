// lab (agent) #235: see lab235.hxx.

#include "lab235.hxx"

#include "sdpb_util/copy_matrix.hxx"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>

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

    El::BigFloat dot(const El::DistMatrix<El::BigFloat> &a,
                     const El::DistMatrix<El::BigFloat> &b)
    {
      return El::Dotu(a, b);
    }

    // Q_new v = sum_j P_j^T (P_j v), P_j = L_j^{-1} B_j (SDPB's schur_off_diagonal)
    void apply_Q(const Block_Matrix &P, Block_Vector &work_N,
                 Block_Vector &work_P, const El::DistMatrix<El::BigFloat> &v,
                 El::DistMatrix<El::BigFloat> &out)
    {
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
                   const El::DistMatrix<El::BigFloat> &x_exact,
                   const El::DistMatrix<El::BigFloat> *x0,
                   const El::BigFloat &stop_abs, El::DistMatrix<El::BigFloat> &x,
                   const std::string &label, int s, const El::BigFloat &p_norm)
    {
      auto &st = state();
      auto t0 = Clock::now();
      double t_mv = 0, t_pc = 0;
      int n_mv = 0, n_pc = 0;
      const El::BigFloat rhs_norm = max_abs(rhs);
      const El::BigFloat xs_norm = max_abs(x_exact);
      const int prec = El::gmp::Precision();
      const El::BigFloat floor_rel
        = El::BigFloat(1) >> std::max(1, prec - st.pcg_floor_bits);

      El::DistMatrix<El::BigFloat> r(rhs), z(rhs.Grid()), p(rhs.Grid()),
        q(rhs.Grid()), d(rhs.Grid());
      El::Zeros(q, rhs.Height(), 1);
      if(x0 != nullptr)
        {
          x = *x0;
          auto t1 = Clock::now();
          apply_Q(P, work_N, work_P, x, q);
          t_mv += seconds_since(t1);
          ++n_mv;
          El::Axpy(El::BigFloat(-1), q, r);
        }
      else
        {
          El::Zeros(x, rhs.Height(), 1);
        }
      std::vector<double> lres, lerr, labs, alphas, betas;
      auto record = [&]() {
        El::BigFloat rn = max_abs(r);
        d = x;
        El::Axpy(El::BigFloat(-1), x_exact, d);
        El::BigFloat en = max_abs(d);
        lres.push_back(log10_abs(rn) - log10_abs(rhs_norm));
        labs.push_back(log10_abs(rn));
        lerr.push_back(log10_abs(en) - log10_abs(xs_norm));
        return rn;
      };
      El::BigFloat rn = record();
      z = r;
      {
        auto t1 = Clock::now();
        El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                                 El::OrientationNS::NORMAL, U_old, z);
        t_pc += seconds_since(t1);
        ++n_pc;
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
          El::BigFloat pq = dot(p, q);
          El::BigFloat alpha = rz / pq;
          El::Axpy(alpha, p, x);
          El::Axpy(-alpha, q, r);
          rn = record();
          res.iterations = k;
          z = r;
          t1 = Clock::now();
          El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                                   El::OrientationNS::NORMAL, U_old, z);
          t_pc += seconds_since(t1);
          ++n_pc;
          El::BigFloat rz_new = dot(r, z);
          El::BigFloat beta = rz_new / rz;
          rz = rz_new;
          // p = z + beta p
          El::Scale(beta, p);
          El::Axpy(El::BigFloat(1), z, p);
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
             << ",\"prec\":" << prec << ",\"t\":" << fmt(seconds_since(t0))
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
    st.self_control = env_int("LAB235_SELF", 1) != 0;
    st.pcg_max_iters = env_int("LAB235_PCG_MAX", 300);
    st.pcg_floor_bits = env_int("LAB235_FLOOR_BITS", 64);
    st.warm_corrector = env_int("LAB235_WARM", 0) != 0;
    st.use_period = env_int("LAB235_USE_PERIOD", 0);
    st.use_eta = env_double("LAB235_USE_ETA", 1e-3);
    st.use_abs_log10 = env_double("LAB235_USE_ABS_LOG10", -1e9);
    st.skip_runs = env_int("LAB235_SKIP_RUNS", 0);
    st.use_thr_factor = env_double("LAB235_USE_THR", 1.0);
    int keep = 1;
    for(auto v : {st.stale_Q, st.stale_S, st.stale_pcg})
      for(int s : v)
        keep = std::max(keep, s);
    keep = std::max(keep, st.use_period - 1);
    st.keep = env_int("LAB235_KEEP", keep);
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
    // the previous run's matrices may live on a grid that no longer exists: leak them
    (new std::deque<El::DistMatrix<El::BigFloat>>)->swap(st.Q_hist);
    (new std::deque<Block_Diagonal_Matrix>)->swap(st.S_hist);
    (new std::vector<El::DistMatrix<El::BigFloat>>)->swap(st.warm);
    std::ostringstream os;
    os << "{\"kind\":\"run\",\"run\":" << st.run
       << ",\"active\":" << (st.active ? "true" : "false") << "}";
    write_line(os.str());
  }

  void begin_step(const El::BigFloat &primal_error,
                  const El::BigFloat &primal_error_threshold)
  {
    if(!enabled())
      return;
    auto &st = state();
    ++st.iteration;
    st.primal_error = primal_error;
    st.primal_threshold = primal_error_threshold;
    st.phase = 0;
    st.warm.clear();
  }

  void after_factor(const Block_Info &block_info,
                    const Block_Diagonal_Matrix &schur_complement_cholesky,
                    const El::DistMatrix<El::BigFloat> &Q_chol)
  {
    if(!enabled())
      return;
    auto &st = state();
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
    // Deep copies constructed in place: moving an El::DistMatrix into the deque
    // left it pointing at freed memory (smoke test: iteration 3 "equal" to 1).
    st.Q_hist.emplace_front(Q_chol);
    El::MakeTrapezoidal(El::UpperOrLowerNS::UPPER, st.Q_hist.front());
    while(st.Q_hist.size() > st.keep)
      st.Q_hist.pop_back();
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
        pcg(P, work_N, work_P, st.Q_hist[s - 1], rhs, x_exact, x0,
            El::BigFloat(0), x, "measure", s, p_norm);
        if(st.phase == 0)
          st.warm[s] = x;
      }
    // control: PCG preconditioned by the exact current factor must converge in 1 step
    if(st.self_control && st.iteration <= 2 && st.phase == 0)
      {
        El::DistMatrix<El::BigFloat> x(rhs.Grid());
        int saved = st.pcg_max_iters;
        st.pcg_max_iters = 3;
        pcg(P, work_N, work_P, Q_chol, rhs, x_exact, nullptr, El::BigFloat(0),
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
            El::DistMatrix<El::BigFloat> x(rhs.Grid());
            const El::DistMatrix<El::BigFloat> *x0 = nullptr;
            if(st.warm_corrector && st.phase == 1 && st.warm[0].Height() > 0)
              x0 = &st.warm[0];
            int saved = st.pcg_max_iters;
            st.pcg_max_iters = st.use_max_iters;
            auto res = pcg(P, work_N, work_P, st.Q_hist[s - 1], rhs, x_exact,
                           x0, stop_abs, x, "use", s, p_norm);
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
}
