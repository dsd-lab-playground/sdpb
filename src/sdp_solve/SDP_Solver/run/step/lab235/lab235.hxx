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
//
// lab (agent) #239: (5) LAB235_INEXACT=1 is a real inexact solver mode, not a
// measurement: Q = sum_j P_j^T P_j is formed (syrk) and factored (Cholesky_Q)
// only on refactor iterations, and U^-1 is formed there once
// (El::TriangularInverse). On the other iterations SDPB skips the syrk and
// Cholesky_Q altogether and solves Q dy = rhs (predictor and corrector) by PCG
// with the matvec sum_j P_j^T (P_j v) and the preconditioner U^-1 U^-T of the
// last refactor. A refactor happens on the run's first iteration, when
// LAB235_USE_PERIOD iterations have passed since the last one, after a solve
// that took more than LAB235_REFACTOR_STEPS PCG steps, and on demand inside a
// solve whose PCG does not reach its target within LAB235_USE_MAX steps (Q is
// then formed and factored there and SDPB's exact solve is used for the rest
// of the step). Forcing: stop at
//   |r|_inf <= max(rel_k |rhs|_inf, eta x max(primal error, thr floor)),
//   rel_k = clamp(LAB235_USE_REL x (Perr_k / Perr_1)^LAB235_REL_GAMMA,
//                 LAB235_REL_MIN, LAB235_USE_REL),
// Perr_k the matrix primal residual |P| at the start of iteration k (not |p|,
// which the inexact solves themselves pollute) and Perr_1 its value at the
// run's first iteration: the forcing tightens as the primal error falls.
// Timers: lab_pcg (in computeSearchDirection), lab_triangular_inverse,
// lab_fallback_refactor; one JSON line per iteration ("istep").

#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdp_solve/Block_Info.hxx"
#include "sdp_solve/Block_Matrix.hxx"
#include "sdp_solve/Block_Vector.hxx"
#include "sdp_solve/SDP.hxx"
#include "sdpb_util/Environment.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdpb_util/Verbosity.hxx"

class BigInt_Shared_Memory_Syrk_Context;

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
    std::vector<int> pcg_iters;      // iterations with measurement PCG (empty: all)
    std::vector<int> pcg_phases;     // phases with measurement PCG (0 predictor, 1 corrector)
    int use_max_iters = 300;         // PCG cap in use mode
    std::vector<int> pcg_doubles;    // reduced-precision PCG runs: storage as sums of nd doubles
    int pcg_doubles_stale = 1;       // their stale level
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
    double use_rel = 0;              // LAB235_USE_REL: also stop at |r|_inf <= use_rel x |rhs|_inf
    int phase = 0;                   // 0 predictor, 1 corrector
    std::deque<El::DistMatrix<El::BigFloat>> Q_hist; // newest first
    std::deque<Block_Diagonal_Matrix> S_hist;        // newest first
    bool prec_inv = false;                           // LAB235_PREC_INV: preconditioner as U^-1 (U^-T r)
    std::deque<El::DistMatrix<El::BigFloat>> Qinv_hist; // U^-1 of Q_hist's factors
    std::vector<El::DistMatrix<El::BigFloat>> warm;  // per stale level
    bool check_BL = false;           // LAB235_CHECK_BL: Q v via B and L as well (timing + agreement)
    const SDP *sdp = nullptr;                          // set in after_factor for this step
    const Block_Diagonal_Matrix *L_now = nullptr;      // this step's Schur Cholesky factors
    std::ofstream log;

    // #239 inexact solver mode (LAB235_INEXACT=1)
    bool inexact = false;
    int refactor_steps = 12;         // LAB235_REFACTOR_STEPS
    double rel_min = 1e-8;           // LAB235_REL_MIN
    double rel_gamma = 0.1;          // LAB235_REL_GAMMA
    bool perr_ref_set = false;
    double perr_ref_log10 = 0;       // log10 |P| at the run's first iteration
    double perr_log10 = 0;           // log10 |P| at the start of this iteration
    double rel_now = 0;              // rel_k of this iteration
    bool skip_Q = false;             // this iteration: Q not formed (PCG solves)
    bool Q_formed = true;            // this iteration: Q formed and factored
    int last_refactor = 0;           // iteration of the last refactor
    bool need_refactor = false;      // set by a slow solve: refactor next iteration
    std::string refactor_reason;
    bool fallback = false;           // this iteration: refactor on demand inside a solve
    std::vector<int> steps_now;      // PCG steps of this iteration's solves
    double t_pcg_now = 0, t_inv_now = 0;
    int refactors = 0, fallbacks = 0, pcg_solves = 0, pcg_steps = 0;
    // context for an on-demand refactor (set in SDP_Solver::step)
    const Environment *env = nullptr;
    BigInt_Shared_Memory_Syrk_Context *syrk_ctx = nullptr;
    Timers *timers = nullptr;
    El::Matrix<int32_t> *block_timings_ms = nullptr;
    Verbosity verbosity = Verbosity::regular;
    El::DistMatrix<El::BigFloat> *Q = nullptr;
    Block_Matrix *P = nullptr;
    // #239 step 3: PCG matvec in LAB235_MATVEC_BITS-bit arb arithmetic (0: working precision);
    // LAB235_MATVEC_CHECK=1: per iteration, both matvecs on the predictor rhs (agreement, time);
    // =2: the same, but the PCG keeps the working-precision matvec (measurement only)
    int matvec_bits = 0;
    int matvec_check = 0;

    // #243 (v2 of the inexact mode; LAB235_INEXACT=1 now means v2):
    //  - the PCG works on vectors held identically by every rank, in arb
    //    arithmetic at LAB235_MATVEC_BITS (default 256) bits;
    //  - its matvec uses rows of P_j (all N columns) held by one rank each
    //    ([VC,STAR] on the block's grid, rounded to 256 bits once per
    //    iteration), so no per-block reductions; its preconditioner uses
    //    rows of U^-1 held the same way (built once per refactor);
    //  - every sum over ranks (rhs, matvec, the two preconditioner products)
    //    is an exact fixed-point integer AllReduce, so all ranks get the same
    //    bits and take the same branches;
    //  - U^-1 is formed where Q was factored, before Q moves to #143's solve
    //    grid (after_cholesky), so --qSolveRanks and the multimodular
    //    Cholesky_Q work unchanged on refactor iterations;
    //  - refactor rule LAB235_REFACTOR_RULE=cost (default): rank 0 measures
    //    the extra cost of a refactor iteration (dR) and the PCG time of each
    //    other iteration, and asks for a refactor when the last iteration's
    //    PCG time reaches the cycle's running average extra cost per
    //    iteration, (dR + PCG time since the refactor) / (iterations since
    //    it, refactor included); the decision is broadcast. Safety nets: a
    //    period cap (LAB235_USE_PERIOD, default 16), a solve over
    //    LAB235_REFACTOR_STEPS (default 60) steps, and the on-demand refactor
    //    when a solve misses its target within LAB235_USE_MAX (default 100).
    //    LAB235_REFACTOR_RULE=steps: #239's rule (period and step threshold).
    int refactor_rule = 1;           // 0: #239's steps/period rule, 1: cost rule
    int64_t mm_panel = 0;            // --multimodularCholeskyQ (for an on-demand refactor)
    bool inverse_done = false;       // this iteration: U^-1 formed (after_cholesky or fallback)
    double t_iter0 = 0;              // rank 0: start of this iteration (begin_step)
    double t_ref_last = -1;          // rank 0: time of the last refactor iteration
    double base_est = -1;            // rank 0: time of the last PCG iteration minus its PCG time
    double cycle_pcg = 0;            // rank 0: PCG time since the last refactor
    double dR_now = -1;              // rank 0: the dR estimate used at the last decision
    double t_rows_P = 0, t_rows_U = 0, t_sum_now = 0;
  };

  State &state();
  void init_from_env();
  bool enabled();
  void begin_run();
  void begin_step(const El::BigFloat &primal_error,
                  const El::BigFloat &primal_error_threshold,
                  const El::BigFloat &primal_error_P);
  // #239: context for an on-demand refactor, set once Q and P exist
  void set_step_context(const Environment &env,
                        BigInt_Shared_Memory_Syrk_Context &syrk_ctx,
                        Timers &timers, El::Matrix<int32_t> &block_timings_ms,
                        Verbosity verbosity, El::DistMatrix<El::BigFloat> &Q,
                        Block_Matrix &P);
  // #239: true when this iteration's Schur solves are PCG (Q not formed)
  bool skip_Q();
  // #239: the factor whose diagonal gives SDPB's printed Q condition number
  const El::DistMatrix<El::BigFloat> &
  cond_factor(const El::DistMatrix<El::BigFloat> &Q);
  // #243: --multimodularCholeskyQ's panel, for an on-demand refactor
  void set_cholesky_panel(int64_t panel);
  // #243: called right after Cholesky_Q on a refactor iteration, on the grid
  // where Q was factored: U^-1 and its 256-bit rows for the preconditioner
  void after_cholesky(const El::DistMatrix<El::BigFloat> &Q_chol);
  // #243: inexact Schur solve (v2). dy: on entry this rank's partial sums of
  // the right-hand side (N x 1); on exit the solution, the same on all ranks
  void inexact_solve_v2(const Block_Matrix &P, El::Matrix<El::BigFloat> &dy);
  void after_factor(const SDP &sdp, const Block_Info &block_info,
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
