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
  // r <= 0 or r >= number of ranks.
  std::unique_ptr<El::Grid> make_sub_grid(const Environment &env, int64_t r)
  {
    const int num_ranks = El::mpi::Size(El::mpi::COMM_WORLD);
    if(r <= 0 || r >= num_ranks)
      return nullptr;

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
    : cholesky_(make_sub_grid(env, cholesky_ranks))
{
  if(solve_ranks > 0 && solve_ranks != cholesky_ranks)
    solve_ = make_sub_grid(env, solve_ranks);
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
