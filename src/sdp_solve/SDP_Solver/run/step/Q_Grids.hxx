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
// r = k * (ranks per node) gives the ranks of k whole nodes. The grid is
// h x (r / h) with h the largest divisor of r not above sqrt(r).
// World rank 0 is always in both sub-grids.
class Q_Grids
{
public:
  // r <= 0 or r >= number of ranks: no sub-grid.
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
