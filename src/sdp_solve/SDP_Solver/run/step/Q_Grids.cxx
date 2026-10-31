#include "Q_Grids.hxx"

#include "sdpb_util/assert.hxx"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <cmath>
#include <vector>

namespace
{
  // Grid of the first r ranks in (node, rank on node) order; nullptr if
  // r <= 0, or r >= number of ranks and the environment variable height_env
  // is unset. height_env (experiment): the grid's height, if it divides r;
  // with r >= number of ranks it makes a grid of all ranks with that height
  // (e.g. the ranks per node: then each process column is one node).
  std::unique_ptr<El::Grid>
  make_sub_grid(const Environment &env, int64_t r, const char *height_env)
  {
    const int num_ranks = El::mpi::Size(El::mpi::COMM_WORLD);
    const char *h_env = std::getenv(height_env);
    const int custom_height = h_env != nullptr ? std::atoi(h_env) : 0;
    if(r <= 0 || (r >= num_ranks && custom_height <= 0))
      return nullptr;
    r = std::min<int64_t>(r, num_ranks);

    // (node index, rank on node) of every world rank
    const int mine[2] = {env.node_index(), env.comm_shared_mem.Rank()};
    std::vector<int> all(2 * num_ranks);
    El::mpi::AllGather(mine, 2, all.data(), 2, El::mpi::COMM_WORLD);
    // SDPB_Q_GRID_ORDER=spread (experiment): (rank on node, node) order
    // instead, so r = k * (number of nodes) takes k ranks on every node
    const char *order_env = std::getenv("SDPB_Q_GRID_ORDER");
    const bool spread
      = order_env != nullptr && std::string(order_env) == "spread";
    std::vector<int> ranks(num_ranks);
    for(int i = 0; i < num_ranks; ++i)
      ranks[i] = i;
    std::stable_sort(ranks.begin(), ranks.end(), [&](int a, int b) {
      const int ka = spread ? 1 : 0, kb = spread ? 0 : 1;
      return std::make_pair(all[2 * a + ka], all[2 * a + kb])
             < std::make_pair(all[2 * b + ka], all[2 * b + kb]);
    });
    ranks.resize(r);
    ASSERT_EQUAL(ranks.at(0), 0, "world rank 0 must lead the Q sub-grid");

    int height = int(std::sqrt(double(r)));
    while(r % height != 0)
      --height;
    if(custom_height > 0 && r % custom_height == 0)
      height = custom_height;
    else if(custom_height > 0 && El::mpi::Rank() == 0)
      El::Output("Warning: ", height_env, "=", custom_height,
                 " does not divide ", r, "; using height ", height);

    El::mpi::Group world_group, owners;
    El::mpi::CommGroup(El::mpi::COMM_WORLD, world_group);
    El::mpi::Incl(world_group, int(r), ranks.data(), owners);
    auto grid = std::make_unique<El::Grid>(El::mpi::COMM_WORLD, owners,
                                           height);
    El::mpi::Free(owners);
    El::mpi::Free(world_group);
    return grid;
  }
}

Q_Grids::Q_Grids(const Environment &env, const int64_t cholesky_ranks,
                 const int64_t solve_ranks)
    : cholesky_(
        make_sub_grid(env, cholesky_ranks, "SDPB_Q_CHOLESKY_GRID_HEIGHT"))
{
  if(solve_ranks > 0 && solve_ranks != cholesky_ranks)
    solve_ = make_sub_grid(env, solve_ranks, "SDPB_Q_SOLVE_GRID_HEIGHT");
  if(El::mpi::Rank() == 0 && (cholesky_ || solve_))
    {
      auto describe = [](const El::Grid *g) {
        return g == nullptr ? std::string("all ranks")
                            : std::to_string(g->Size()) + " ranks ("
                                + std::to_string(g->Height()) + " x "
                                + std::to_string(g->Width()) + ")";
      };
      El::Output("Q sub-grids (lab #143): Cholesky(Q) on ",
                 describe(cholesky_grid()), ", Q solves on ",
                 solve_ ? describe(solve_.get()) : std::string("the same"));
    }
}

const El::Grid *Q_Grids::cholesky_grid() const
{
  return cholesky_.get();
}

const El::Grid *Q_Grids::solve_grid() const
{
  return solve_.get();
}

void copy_upper_between_grids(const El::DistMatrix<El::BigFloat> &A,
                              El::DistMatrix<El::BigFloat> &B)
{
  const El::Int n = A.Height();
  ASSERT_EQUAL(n, A.Width());
  B.Resize(n, n);
  El::Zero(B);
  const int num_ranks = El::mpi::Size(El::mpi::COMM_WORLD);

  // World rank owning (i, j) in M
  auto owners = [n](const El::DistMatrix<El::BigFloat> &M) {
    std::vector<int> row(n), col(n);
    for(El::Int i = 0; i < n; ++i)
      {
        row[i] = M.RowOwner(i);
        col[i] = M.ColOwner(i);
      }
    return std::make_pair(row, col);
  };
  const auto owners_A = owners(A), owners_B = owners(B);
  const std::vector<int> &row_A = owners_A.first, &col_A = owners_A.second;
  const std::vector<int> &row_B = owners_B.first, &col_B = owners_B.second;
  auto owner = [](const El::DistMatrix<El::BigFloat> &M, const int r,
                  const int c) {
    return M.Grid().VCToViewing(r + c * M.ColStride());
  };

  // Both sides visit the shared entries in the same order: local columns,
  // then local rows (global (j, i) increasing), upper triangle only.
  std::vector<int> send_counts(num_ranks, 0), recv_counts(num_ranks, 0);
  auto for_each_upper = [](const El::DistMatrix<El::BigFloat> &M,
                           const auto &f) {
    for(El::Int jLoc = 0; jLoc < M.LocalWidth(); ++jLoc)
      {
        const El::Int j = M.GlobalCol(jLoc);
        for(El::Int iLoc = 0; iLoc < M.LocalHeight(); ++iLoc)
          {
            const El::Int i = M.GlobalRow(iLoc);
            if(i > j)
              break;
            f(iLoc, jLoc, i, j);
          }
      }
  };
  if(A.Participating())
    for_each_upper(A, [&](El::Int, El::Int, El::Int i, El::Int j) {
      ++send_counts[owner(B, row_B[i], col_B[j])];
    });
  if(B.Participating())
    for_each_upper(B, [&](El::Int, El::Int, El::Int i, El::Int j) {
      ++recv_counts[owner(A, row_A[i], col_A[j])];
    });
  std::vector<int> send_displs(num_ranks, 0), recv_displs(num_ranks, 0);
  for(int r = 1; r < num_ranks; ++r)
    {
      send_displs[r] = send_displs[r - 1] + send_counts[r - 1];
      recv_displs[r] = recv_displs[r - 1] + recv_counts[r - 1];
    }
  const int send_total = send_displs.back() + send_counts.back();
  const int recv_total = recv_displs.back() + recv_counts.back();

  std::vector<El::BigFloat> send_buf(send_total), recv_buf(recv_total);
  {
    std::vector<int> pos(send_displs);
    if(A.Participating())
      for_each_upper(A, [&](El::Int iLoc, El::Int jLoc, El::Int i,
                            El::Int j) {
        send_buf[pos[owner(B, row_B[i], col_B[j])]++]
          = A.GetLocalCRef(iLoc, jLoc);
      });
  }
  El::mpi::AllToAll(send_buf.data(), send_counts.data(), send_displs.data(),
                    recv_buf.data(), recv_counts.data(), recv_displs.data(),
                    El::mpi::COMM_WORLD);
  {
    std::vector<int> pos(recv_displs);
    if(B.Participating())
      for_each_upper(B, [&](El::Int iLoc, El::Int jLoc, El::Int i,
                            El::Int j) {
        B.SetLocal(iLoc, jLoc, recv_buf[pos[owner(A, row_A[i], col_A[j])]++]);
      });
  }
}
