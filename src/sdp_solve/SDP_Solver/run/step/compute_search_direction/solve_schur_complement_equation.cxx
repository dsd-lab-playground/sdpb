#include "lower_triangular_solve.hxx"
#include "sdp_solve/SDP_Solver.hxx"
#include "sdp_solve/lower_triangular_transpose_solve.hxx"
#include "sdpb_util/copy_matrix.hxx"
#include "sdpb_util/Timers/Timers.hxx"
#include "../lab235/lab235.hxx"

#include <cstdlib>
#include <optional>

// Solve the Schur complement equation for dx, dy.
//
// - As inputs, dx and dy are the residues r_x and r_y on the
//   right-hand side of the Schur complement equation.
// - As outputs, dx and dy are overwritten with the solutions of the
//   Schur complement equation.
//
// The equation is solved using the block-decomposition described in
// the manual.
//
void solve_schur_complement_equation(
  const Block_Diagonal_Matrix &schur_complement_cholesky,
  const Block_Matrix &schur_off_diagonal,
  const El::DistMatrix<El::BigFloat> &Q, Block_Vector &dx, Block_Vector &dy,
  Timers *timers)
{
  // lab #226: sub-timers (no change to the arithmetic)
  std::optional<Scoped_Timer> phase;
  auto begin = [&](const char *name) {
    phase.reset();
    if(timers != nullptr)
      phase.emplace(*timers, name);
  };
  // dx = schur_complement^{-1}.dx
  begin("L_solve");
  lower_triangular_solve(schur_complement_cholesky, dx);
  begin("dy_sum");

  // lab (agent) #243: inexact mode, Q not formed this iteration. The rhs's
  // partial sums go to the PCG, which sums them exactly over all ranks and
  // returns dy, the same on every rank (lab235.hxx, v2).
  if(lab235::skip_Q())
    {
      El::Matrix<El::BigFloat> dy_sum;
      Zeros(dy_sum, Q.Height(), 1);
      for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
        {
          Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(-1),
               schur_off_diagonal.blocks[block], dx.blocks[block],
               El::BigFloat(1), dy.blocks[block]);
          for(int64_t row = 0; row < dy.blocks[block].LocalHeight(); ++row)
            {
              const int64_t global_row(dy.blocks[block].GlobalRow(row));
              for(int64_t column = 0;
                  column < dy.blocks[block].LocalWidth(); ++column)
                dy_sum(global_row, dy.blocks[block].GlobalCol(column))
                  += dy.blocks[block].GetLocal(row, column);
            }
        }
      phase.reset(); // lab_pcg directly under solve_schur
      lab235::inexact_solve_v2(schur_off_diagonal, dy_sum);
      begin("dx_update");
      for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
        {
          copy_matrix(dy_sum, dy.blocks[block]);
          Gemv(El::OrientationNS::NORMAL, El::BigFloat(1),
               schur_off_diagonal.blocks[block], dy.blocks[block],
               El::BigFloat(1), dx.blocks[block]);
        }
      begin("LT_solve");
      lower_triangular_transpose_solve(schur_complement_cholesky, dx);
      phase.reset();
      return;
    }

  // lab #143: if Q lives on a sub-grid of the ranks (--qSolveRanks,
  // --qCholeskyRanks), or SDPB_DY_ALLREDUCE=1, sum dy over all ranks by one
  // AllReduce, solve on Q's grid and broadcast the result from rank 0.
  // (Otherwise, as before: QueueUpdate/ProcessQueues to Q's [MC,MR] owners.)
  static const bool dy_allreduce_env = [] {
    const char *env = std::getenv("SDPB_DY_ALLREDUCE");
    return env != nullptr && env[0] == '1';
  }();
  if(dy_allreduce_env || Q.Grid() != El::Grid::Default())
    {
      El::Matrix<El::BigFloat> dy_sum;
      Zeros(dy_sum, Q.Height(), 1);
      for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
        {
          Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(-1),
               schur_off_diagonal.blocks[block], dx.blocks[block],
               El::BigFloat(1), dy.blocks[block]);
          for(int64_t row = 0; row < dy.blocks[block].LocalHeight(); ++row)
            {
              const int64_t global_row(dy.blocks[block].GlobalRow(row));
              for(int64_t column = 0;
                  column < dy.blocks[block].LocalWidth(); ++column)
                dy_sum(global_row, dy.blocks[block].GlobalCol(column))
                  += dy.blocks[block].GetLocal(row, column);
            }
        }
      begin("dy_allreduce");
      El::mpi::AllReduce(dy_sum.Buffer(), dy_sum.Height(), El::mpi::SUM,
                         El::mpi::COMM_WORLD);
      if(Q.Participating())
        {
          begin("Q_solve");
          El::DistMatrix<El::BigFloat> dy_dist(Q.Grid());
          dy_dist.Resize(Q.Height(), 1);
          copy_matrix(dy_sum, dy_dist);
          El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                                   El::OrientationNS::NORMAL, Q, dy_dist);
          begin("dy_star_star");
          El::DistMatrix<El::BigFloat, El::STAR, El::STAR> dy_star_star(
            dy_dist);
          if(El::mpi::Rank() == 0)
            dy_sum = dy_star_star.Matrix();
        }
      begin("dy_bcast");
      El::mpi::Broadcast(dy_sum.Buffer(), dy_sum.Height(), 0,
                         El::mpi::COMM_WORLD);
      begin("dx_update");
      for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
        {
          copy_matrix(dy_sum, dy.blocks[block]);
          Gemv(El::OrientationNS::NORMAL, El::BigFloat(1),
               schur_off_diagonal.blocks[block], dy.blocks[block],
               El::BigFloat(1), dx.blocks[block]);
        }
      begin("LT_solve");
      lower_triangular_transpose_solve(schur_complement_cholesky, dx);
      phase.reset();
      return;
    }

  El::DistMatrix<El::BigFloat> dy_dist;
  Zeros(dy_dist, Q.Height(), 1);
  // lab (agent) #235: p = b - B^T x is the sum of dy's blocks on entry
  // (#243: not in the inexact mode, whose refactor iterations solve exactly)
  const bool lab_measure = lab235::enabled() && !lab235::state().inexact;
  El::DistMatrix<El::BigFloat> lab_p, lab_rhs;
  if(lab_measure)
    {
      Zeros(lab_p, Q.Height(), 1);
      lab235::sum_blocks(dy, lab_p);
    }
  {
    El::Matrix<El::BigFloat> dy_sum;
    Zeros(dy_sum, Q.Height(), 1);

    for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
      {
        Gemv(El::OrientationNS::TRANSPOSE, El::BigFloat(-1),
             schur_off_diagonal.blocks[block], dx.blocks[block],
             El::BigFloat(1), dy.blocks[block]);

        // Locally sum contributions to dy
        for(int64_t row = 0; row < dy.blocks[block].LocalHeight(); ++row)
          {
            int64_t global_row(dy.blocks[block].GlobalRow(row));
            for(int64_t column = 0; column < dy.blocks[block].LocalWidth();
                ++column)
              {
                int64_t global_column(dy.blocks[block].GlobalCol(column));
                dy_sum(global_row, global_column)
                  += dy.blocks[block].GetLocal(row, column);
              }
          }
      }

    // Send out updates for dy
    begin("dy_queue");
    El::BigFloat zero(0);
    for(int64_t row = 0; row < dy_sum.Height(); ++row)
      for(int64_t column = 0; column < dy_sum.Width(); ++column)
        {
          if(dy_sum(row, column) != zero)
            {
              dy_dist.QueueUpdate(row, column, dy_sum(row, column));
            }
        }
  }
  begin("dy_process_queues");
  dy_dist.ProcessQueues();
  if(lab_measure)
    lab_rhs = dy_dist;

  // dy_dist = Q^{-1}.dy_dist
  begin("Q_solve");
  El::cholesky::SolveAfter(El::UpperOrLowerNS::UPPER,
                           El::OrientationNS::NORMAL, Q, dy_dist);
  if(lab_measure)
    lab235::solve_hook(schur_off_diagonal, dy, dx, Q, lab_rhs, lab_p,
                       dy_dist);
  begin("dy_star_star");
  El::DistMatrix<El::BigFloat, El::STAR, El::STAR> dy_local(dy_dist);
  begin("dx_update");

  // dx += schur_off_diagonal.dy
  for(size_t block = 0; block < schur_off_diagonal.blocks.size(); ++block)
    {
      copy_matrix(dy_local, dy.blocks[block]);
      Gemv(El::OrientationNS::NORMAL, El::BigFloat(1),
           schur_off_diagonal.blocks[block], dy.blocks[block], El::BigFloat(1),
           dx.blocks[block]);
    }

  // dx = schur_complement^{-T}.dx
  begin("LT_solve");
  lower_triangular_transpose_solve(schur_complement_cholesky, dx);
  phase.reset();
}
