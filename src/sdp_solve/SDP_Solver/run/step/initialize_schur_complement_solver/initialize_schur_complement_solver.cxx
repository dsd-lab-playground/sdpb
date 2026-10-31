#include "sdp_solve/SDP.hxx"
#include "sdp_solve/Block_Diagonal_Matrix.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/BigInt_Shared_Memory_Syrk_Context.hxx"
#include "sdp_solve/memory_estimates.hxx"
#include "sdp_solve/SDP_Solver/run/multimodular/multimodular_blas.hxx"
#include "sdp_solve/SDP_Solver/run/step/Q_Grids.hxx"
#include "../lab235/lab235.hxx"

#include <cstdlib>

// Compute the quantities needed to solve the Schur complement
// equation
//
// {{S, -B}, {B^T, 0}} . {dx, dy} = {r, s}
//
// (where S = SchurComplement, B = FreeVarMatrix), using the method
// described in the manual:
//
// - Compute S using BilinearPairingsXInv and BilinearPairingsY.
//
// - Compute the Cholesky decomposition S' = L' L'^T.
//
// - Form B' = (B U) and compute
//
//   - SchurOffDiagonal = L'^{-1} B
//   - L'^{-1} U
//   - Q = (L'^{-1} B')^T (L'^{-1} B') - {{0, 0}, {0, 1}}
//
// - Compute the LU decomposition of Q.
//
// This data is sufficient to efficiently solve the above equation for
// a given r,s.
//
// Inputs:
// - BilinearPairingsXInv, BilinearPairingsY (these are members of
//   SDPSolver, but we include them as arguments to emphasize that
//   they must be computed first)
// Workspace (members of SDPSolver which are modified by this method
// and not used later):
// - SchurComplement
// Outputs (members of SDPSolver which are modified by this method and
// used later):
// - SchurComplementCholesky
// - SchurOffDiagonal
//

void compute_schur_complement(
  const Block_Info &block_info,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  Block_Diagonal_Matrix &schur_complement, Timers &timers);

void compute_Q(const Environment &env, const SDP &sdp,
               const Block_Info &block_info,
               const Block_Diagonal_Matrix &schur_complement,
               Block_Matrix &schur_off_diagonal,
               Block_Diagonal_Matrix &schur_complement_cholesky,
               BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
               El::DistMatrix<El::BigFloat> &Q, Timers &timers,
               El::Matrix<int32_t> &block_timings_ms, bool multimodular_trsm,
               Verbosity verbosity);

void initialize_schur_complement_solver(
  const Environment &env, const Block_Info &block_info, const SDP &sdp,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  const El::Grid &group_grid, Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
  El::DistMatrix<El::BigFloat> &Q, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, bool multimodular_trsm,
  int64_t multimodular_cholesky_Q, Verbosity verbosity,
  const Q_Grids *q_grids);

void initialize_schur_complement_solver(
  const Environment &env, const Block_Info &block_info, const SDP &sdp,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  const El::Grid &group_grid, Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
  El::DistMatrix<El::BigFloat> &Q, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const bool multimodular_trsm,
  const int64_t multimodular_cholesky_Q, const Verbosity verbosity)
{
  initialize_schur_complement_solver(
    env, block_info, sdp, A_X_inv, A_Y, group_grid, schur_complement_cholesky,
    schur_off_diagonal, bigint_syrk_context, Q, timers, block_timings_ms,
    multimodular_trsm, multimodular_cholesky_Q, verbosity, nullptr);
}

// q_grids (lab #143, may be nullptr): factor Q on q_grids->cholesky_grid()
// and leave the factor on q_grids->solve_grid(). On return Q is a matrix on
// that grid; ranks outside it hold no part of Q.
void initialize_schur_complement_solver(
  const Environment &env, const Block_Info &block_info, const SDP &sdp,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_X_inv,
  const std::array<
    std::vector<std::vector<std::vector<El::DistMatrix<El::BigFloat>>>>, 2>
    &A_Y,
  const El::Grid &group_grid, Block_Diagonal_Matrix &schur_complement_cholesky,
  Block_Matrix &schur_off_diagonal,
  BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
  El::DistMatrix<El::BigFloat> &Q, Timers &timers,
  El::Matrix<int32_t> &block_timings_ms, const bool multimodular_trsm,
  const int64_t multimodular_cholesky_Q, const Verbosity verbosity,
  const Q_Grids *q_grids)
{
  Scoped_Timer initialize_timer(timers, "initializeSchurComplementSolver");
  // The Schur complement matrix S: a Block_Diagonal_Matrix with one
  // block for each 0 <= j < J.  SchurComplement.blocks[j] has dimension
  // (d_j+1)*m_j*(m_j+1)/2
  //
  Block_Diagonal_Matrix schur_complement(
    block_info.schur_block_sizes(), block_info.block_indices,
    block_info.num_points.size(), group_grid);
  if(verbosity >= Verbosity::trace)
    {
      print_allocation_message_per_node(env, "schur_complement",
                                        get_allocated_bytes(schur_complement));
    }
  compute_schur_complement(block_info, A_X_inv, A_Y, schur_complement, timers);

  compute_Q(env, sdp, block_info, schur_complement, schur_off_diagonal,
            schur_complement_cholesky, bigint_syrk_context, Q, timers,
            block_timings_ms, multimodular_trsm, verbosity);

  // lab (agent) #239: inexact mode, not a refactor iteration: no Q (syrk
  // skipped in compute_Q), no Cholesky_Q; Q stays (unformed) on the default
  // grid
  if(lab235::skip_Q())
    return;

  // Move Q to the grid where it is factored (lab #143).
  // SDPB_Q_COPY_UPPER=1: copy the upper triangle only (all that
  // Cholesky(UPPER) and the solves read), else El's copy of all N^2 entries.
  static const bool copy_upper = [] {
    const char *env = std::getenv("SDPB_Q_COPY_UPPER");
    return env != nullptr && env[0] == '1';
  }();
  auto move_Q_to = [&Q, &timers](const El::Grid &grid, const char *name) {
    Scoped_Timer timer(timers, name);
    El::DistMatrix<El::BigFloat> Q_new(grid);
    if(copy_upper)
      copy_upper_between_grids(Q, Q_new);
    else
      Q_new = Q;
    Q.Empty();
    Q = std::move(Q_new);
  };
  if(q_grids != nullptr && q_grids->cholesky_grid() != nullptr)
    move_Q_to(*q_grids->cholesky_grid(), "Q_to_cholesky_grid");

  Scoped_Timer Cholesky_timer(timers, "Cholesky_Q");
  try
    {
      if(!Q.Participating())
        {
          // Q is on a sub-grid without this rank
        }
      else if(multimodular_cholesky_Q > 0)
        {
          // lab #226 prototype; SDPB_MMQ_DIAG=1: diagonal blocks by
          // multimodular_cholesky_lower too
          const char *diag_env = std::getenv("SDPB_MMQ_DIAG");
          const bool mm_diag = diag_env != nullptr && diag_env[0] == '1';
          Multimodular_Blas_Config config;
          Multimodular_Cholesky_Dist_Timings t;
          multimodular_cholesky_upper_dist(Q, int(multimodular_cholesky_Q),
                                           mm_diag, config, t, &timers);
        }
      else
        Cholesky(El::UpperOrLowerNS::UPPER, Q);
    }
  catch(std::exception &e)
    {
      RUNTIME_ERROR("Error when computing Cholesky(Q): ", e.what());
    }
  Cholesky_timer.stop();

  // lab (agent) #243: inexact mode, refactor iteration: U^-1 and the
  // preconditioner's 256-bit rows, on the grid where Q was factored (all
  // ranks unless --qCholeskyRanks), before Q moves to the solve grid
  lab235::after_cholesky(Q);

  if(q_grids != nullptr && q_grids->solve_grid() != nullptr)
    move_Q_to(*q_grids->solve_grid(), "Q_to_solve_grid");
}
