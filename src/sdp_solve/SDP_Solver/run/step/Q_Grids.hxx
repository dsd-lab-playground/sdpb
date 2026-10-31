#pragma once

#include "sdpb_util/Environment.hxx"

#include <El.hpp>

#include <memory>

// Experimental (lab #143): process grids on a subset of the ranks for the
// global Q, the N x N matrix of the Schur complement solver.
//
// Q is computed on all ranks (compute_Q). Its Cholesky factorization and
// the two triangular solves per search direction are latency-bound: on many
// nodes they get slower as ranks are added. With --qCholeskyRanks r, Q is
// copied to a grid of r ranks and factored there; with --qSolveRanks s, the
// factor is copied to a grid of s ranks for the solves (dy is summed over
// all ranks by one AllReduce and broadcast back after the solve).
//
// The ranks of a sub-grid are the first r in (node, rank on node) order, so
// r = k * (ranks per node) gives the ranks of k whole nodes (with
// SDPB_Q_GRID_ORDER=spread: (rank on node, node) order, so r = k * (number of
// nodes) gives k ranks on every node). The grid is
// h x (r / h) with h the largest divisor of r not above sqrt(r)
// (experiment: SDPB_Q_CHOLESKY_GRID_HEIGHT / SDPB_Q_SOLVE_GRID_HEIGHT set h if
// it divides r; with r >= number of ranks they make a grid of all ranks with
// that height, e.g. h = ranks per node, so that each process column is one
// node). World rank 0 is always in both sub-grids.
class Q_Grids
{
public:
  // r <= 0 or r >= number of ranks: no sub-grid (unless a height is set).
  // solve_ranks <= 0 or == cholesky_ranks: solves on the Cholesky grid.
  Q_Grids(const Environment &env, int64_t cholesky_ranks,
          int64_t solve_ranks);

  // Grid for Cholesky(Q); nullptr: the grid Q was computed on.
  [[nodiscard]] const El::Grid *cholesky_grid() const;
  // Grid for the solves with Q; nullptr: the Cholesky grid.
  [[nodiscard]] const El::Grid *solve_grid() const;

private:
  std::unique_ptr<El::Grid> cholesky_, solve_;
};

// B := the upper triangle of A, with B's strictly lower triangle zero
// (lab #143). A and B are [MC,MR] on grids whose viewing communicator has
// the ranks of COMM_WORLD (B keeps its grid); all ranks call it. One
// AllToAll moves the N(N+1)/2 entries that Cholesky(UPPER) and the
// triangular solves read, instead of El::Copy's N^2.
void copy_upper_between_grids(const El::DistMatrix<El::BigFloat> &A,
                              El::DistMatrix<El::BigFloat> &B);
