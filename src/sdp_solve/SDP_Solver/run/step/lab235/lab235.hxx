#pragma once

// lab (agent) #235: measurement hooks for an inexact interior-point step.
//
// Everything here is off unless LAB235_DIR is set. When it is on, SDPB still
// computes its exact search direction every iteration; in addition it
//  (1) keeps the Cholesky factors of Q (upper, Q = U^T U) and of the Schur
//      blocks S_j (lower, S_j = L_j L_j^T) from the last LAB235_KEEP iterations;
//  (2) computes the spectra of the stale-preconditioned operators
//      U_old^{-T} Q_new U_old^{-1}  and  L_old^{-1} S_new L_old^{-T}
//      in working precision (as W^T W with W = U_new U_old^{-1}, and V V^T with
//      V = L_old^{-1} L_new), then takes eigenvalues of the rounded double
//      matrices (accurate to ~1e-16 of the largest eigenvalue);
//  (3) in each Schur solve (predictor and corrector), runs preconditioned CG on
//      Q_new dy = rhs, with Q_new v = sum_j P_j^T (P_j v), P_j = L_j^{-1} B_j
//      (SDPB's schur_off_diagonal), preconditioned by a stale U_old, in working
//      precision, and logs residuals and errors against SDPB's exact dy;
//  (4) optionally (LAB235_USE_PERIOD=m) replaces the exact dy by the PCG one on
//      iterations that are not multiples of m, the preconditioner being the
//      factor of the last multiple of m ("refactor every m iterations").
// Output: JSON lines in $LAB235_DIR/lab235.jsonl (rank 0), eigenvalues in
// $LAB235_DIR/eig-Q-it<k>-s<s>.txt.

#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/Block_Matrix.hxx"
#include "sdp_solve/Block_Vector.hxx"

#include <El.hpp>

#include <deque>
#include <fstream>
#include <string>
#include <vector>

namespace lab235
{
  struct State
  {
    bool initialized = false;
    bool enabled = false;
    std::string dir;
    size_t keep = 0;                 // factors kept (max stale level)
    std::vector<int> stale_Q;        // stale levels for Q spectra
    std::vector<int> stale_S;        // stale levels for S spectra
    std::vector<int> stale_pcg;      // stale levels for PCG runs
    std::vector<int> spectra_iters;  // iterations with spectra (empty: all)
    bool self_control = true;        // U_new against itself, once
    bool self_done = false;
    int pcg_max_iters = 300;
    int pcg_floor_bits = 64;         // stop when |r| < 2^(-(prec - floor_bits)) |rhs|
    bool warm_corrector = false;     // corrector PCG starts from predictor's
    int use_period = 0;              // 0: exact dy always
    double use_eta = 0;              // stop when |r|_inf <= eta * max(primal error, thr floor, abs)
    double use_abs_log10 = -1e9;     // log10 of the absolute floor
    int run = 0;                     // SDP_Solver::run calls (the timing pass is run 1)
    int skip_runs = 0;               // runs without instrumentation (LAB235_SKIP_RUNS)
    bool active = false;             // enabled and run > skip_runs
    int iteration = 0;               // steps taken in this run
    El::BigFloat primal_error;       // SDPB's max(|P|, |p|) at the start of the step
    El::BigFloat primal_threshold;   // --primalErrorThreshold
    double use_thr_factor = 0;       // use mode floor: LAB235_USE_THR x primal_threshold
    int phase = 0;                   // 0 predictor, 1 corrector
    std::deque<El::DistMatrix<El::BigFloat>> Q_hist; // newest first
    std::deque<Block_Diagonal_Matrix> S_hist;        // newest first
    std::vector<El::DistMatrix<El::BigFloat>> warm;  // per stale level
    std::ofstream log;
  };

  State &state();
  void init_from_env();
  bool enabled();
  void begin_run();
  void begin_step(const El::BigFloat &primal_error,
                  const El::BigFloat &primal_error_threshold);
  void after_factor(const Block_Info &block_info,
                    const Block_Diagonal_Matrix &schur_complement_cholesky,
                    const El::DistMatrix<El::BigFloat> &Q_chol);
  void end_step(const Block_Diagonal_Matrix &schur_complement_cholesky,
                const El::DistMatrix<El::BigFloat> &Q_chol);
  // Called in solve_schur_complement_equation after SDPB's exact solve.
  // rhs: the right-hand side of Q dy = rhs; p: the primal residue b - B^T x;
  // dy_dist: in, SDPB's exact solution; out, possibly the PCG solution.
  void solve_hook(const Block_Matrix &P, Block_Vector &work_N,
                  const Block_Vector &work_P,
                  const El::DistMatrix<El::BigFloat> &Q_chol,
                  const El::DistMatrix<El::BigFloat> &rhs,
                  const El::DistMatrix<El::BigFloat> &p,
                  El::DistMatrix<El::BigFloat> &dy_dist);
  // Sum the per-block partial vectors (as SDPB does for dy) into a global vector
  void sum_blocks(const Block_Vector &v, El::DistMatrix<El::BigFloat> &out);
  double log10_abs(const El::BigFloat &x);
}
