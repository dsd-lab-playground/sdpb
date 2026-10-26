#include "Compute_S_Context.hxx"
#include "lab_guard_bits.hxx"
#include "sdp_solve/Block_Matrix/Block_Matrix.hxx"
#include "sdp_solve/SDP_Solver/run/bigint_syrk/Matrix_Normalizer.hxx"
#include "sdpa_solve/SDP.hxx"
#include "sdpa_solve/memory_estimates.hxx"
#include "sdpb_util/split_range.hxx"
#include "sdpb_util/Timers/Timers.hxx"

#include <cstdlib>
#include <string>

// Q = P^T P
// See sdp_solve/SDP_Solver/run/step/initialize_schur_complement_solver/compute_Q.cxx
// TODO rename
void syrk_Q(const Environment &env, Block_Matrix &P,
            BigInt_Shared_Memory_Syrk_Context &bigint_syrk_context,
            El::DistMatrix<El::BigFloat> &Q, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity);

namespace Sdpb::Sdpa
{
  // compute P = (vec(G_1) vec(G_2) vec(G_3) ... )
  // G_p = L_X_Inv F_p L_Y
  // P is split vertically into blocks
  Block_Matrix initialize_P(const Environment &env, const SDP &sdp,
                            const Block_Info &block_info, const El::Grid &grid,
                            const Block_Diagonal_Matrix &X_cholesky,
                            const Block_Diagonal_Matrix &Y_cholesky,
                            Initialize_P_Context &ctx, Timers &timers,
                            El::Matrix<int32_t> &block_timings_ms,
                            const Verbosity verbosity)
  {
    // TODO add timers, update block_timings_ms
    Scoped_Timer timer(timers, "initialize_P");
    const size_t primal_dimension = block_info.primal_dimension;

    // Set matrix dimensions for output Block_Matrix P
    // Each block is a (dim*dim) x primal_dimension matrix
    std::vector<size_t> P_block_heights;
    P_block_heights.reserve(block_info.num_blocks());
    for(const auto &dim : block_info.block_dimensions)
      {
        P_block_heights.emplace_back(dim * dim);
      }
    const size_t P_width = primal_dimension;
    Block_Matrix result(P_block_heights, P_width, block_info.block_indices,
                        grid);
    if(verbosity >= Verbosity::trace)
      {
        print_allocation_message_per_node(env, "P",
                                          get_allocated_bytes(result));
      }

    const auto bits = El::gmp::Precision();
    constexpr auto uplo = El::LOWER;
    constexpr auto diag = El::NON_UNIT;
    constexpr auto orientation = El::NORMAL;

    // Prepare L_X_inv and L_Y
    Scoped_Timer L_timer(timers, "L");
    auto L_X_inv = X_cholesky;
    for(auto &block : L_X_inv.blocks)
      {
        // lab (bfss #77): LAB_L_INV_EXTRA_BITS (experimental, lab_guard_bits.hxx)
        if(lab::l_inv_extra_bits() > 0
           && lab::lower_triangular_inverse_extra(block,
                                                  lab::l_inv_extra_bits()))
          continue;
        El::TriangularInverse(uplo, diag, block);
      }
    auto L_Y = Y_cholesky;
    if(verbosity >= Verbosity::trace)
      {
        print_allocation_message_per_node(env, "L_X_inv",
                                          get_allocated_bytes(L_X_inv));
        print_allocation_message_per_node(env, "L_Y",
                                          get_allocated_bytes(L_Y));
      }

    // lab (bfss #77): LAB_GUARD_BITS=<g>: power-of-2 fixed point with g
    // guard bits instead of the 2-norm normalization (lab_guard_bits.hxx)
    const int guard = lab::guard_bits();
    if(guard > 0)
      {
        const int fp_bits = bits + guard;
        std::vector<std::vector<int>> L_X_inv_rows, L_Y_cols;
        for(auto &block : L_X_inv.blocks)
          {
            L_X_inv_rows.push_back(
              lab::pow2_shifts(block, true, true, fp_bits));
            lab::apply_shifts(block, L_X_inv_rows.back(), {}, true, 1);
          }
        for(auto &block : L_Y.blocks)
          {
            L_Y_cols.push_back(lab::pow2_shifts(block, false, true, fp_bits));
            lab::apply_shifts(block, {}, L_Y_cols.back(), true, 1);
          }
        ctx.compute_residues(L_X_inv, ctx.L_X_inv_residues, block_timings_ms);
        ctx.compute_residues(L_Y, ctx.L_Y_residues, block_timings_ms);
        L_timer.stop();
        Scoped_Timer G_timer(timers, "G");
        for(const auto &p_range :
            split_range(El::IR(0, primal_dimension), ctx.cfg.split_factor))
          {
            const size_t p_begin = p_range.beg;
            const size_t p_end = p_range.end;
            Scoped_Timer p_timer(timers,
                                 El::BuildString("p_", p_begin, "_", p_end));
            std::vector<Block_Diagonal_Matrix> G;
            G.reserve(p_end - p_begin);
            G.insert(G.end(), sdp.sdp_blocks_F.begin() + p_begin,
                     sdp.sdp_blocks_F.begin() + p_end);
            for(const auto side : {El::LEFT, El::RIGHT})
              {
                // left: L_X_inv rows x G columns; right: G rows x L_Y columns
                std::vector<std::vector<std::vector<int>>> G_shifts(G.size());
                {
                  Scoped_Timer norm_timer(timers, "normalize_G");
                  for(size_t v = 0; v < G.size(); ++v)
                    for(auto &block : G[v].blocks)
                      {
                        const bool rows = side == El::RIGHT;
                        G_shifts[v].push_back(
                          lab::pow2_shifts(block, rows, false, fp_bits));
                        if(rows)
                          lab::apply_shifts(block, G_shifts[v].back(), {},
                                            false, 1);
                        else
                          lab::apply_shifts(block, {}, G_shifts[v].back(),
                                            false, 1);
                      }
                }
                {
                  Scoped_Timer compute_residues_timer(timers,
                                                      "compute_residues");
                  auto &G_residues = ctx.G_residues(
                    p_end - p_begin,
                    side == El::LEFT ? El::HORIZONTAL : El::VERTICAL);
                  ctx.compute_residues(G, G_residues, block_timings_ms);
                  compute_residues_timer.stop();
                  ctx.trmm(side, uplo, orientation, diag,
                           side == El::LEFT ? ctx.L_X_inv_residues
                                            : ctx.L_Y_residues,
                           G_residues, G, verbosity, timers,
                           block_timings_ms);
                }
                Scoped_Timer remove_normalization_timer(
                  timers, "remove_normalization");
                for(size_t v = 0; v < G.size(); ++v)
                  for(size_t b = 0; b < G[v].blocks.size(); ++b)
                    {
                      auto &block = G[v].blocks[b];
                      if(side == El::LEFT)
                        lab::apply_shifts(block, L_X_inv_rows[b],
                                          G_shifts[v][b], false, -1);
                      else
                        lab::apply_shifts(block, G_shifts[v][b], L_Y_cols[b],
                                          false, -1);
                    }
              }
            Scoped_Timer reshape_timer(timers, "reshape");
            for(size_t block = 0; block < block_info.num_blocks_local();
                ++block)
              {
                auto &P_block = result.blocks.at(block);
                for(size_t p = p_begin; p < p_end; ++p)
                  {
                    const auto &G_p_block = G.at(p - p_begin).blocks.at(block);
                    const int vec_height = P_block.Height();
                    auto vec_G_p_view
                      = El::View(P_block, El::Range<int>(0, vec_height),
                                 El::Range<int>(p, p + 1));
                    El::Reshape(vec_height, 1, G_p_block, vec_G_p_view);
                  }
              }
          }
        return result;
      }

    // Make L_X_inv and L_Y bigint matrices
    const auto L_X_inv_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::ROWS>(L_X_inv, bits,
                                                             uplo);
    const auto L_Y_normalizer
      = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(L_Y, bits,
                                                                uplo);

    ctx.compute_residues(L_X_inv, ctx.L_X_inv_residues, block_timings_ms);
    ctx.compute_residues(L_Y, ctx.L_Y_residues, block_timings_ms);

    L_timer.stop();

    // Finished L_X_inv and L_Y

    // Compute G_p = L_X_inv F_p L_Y and copy to P
    Scoped_Timer G_timer(timers, "G");
    for(const auto &p_range :
        split_range(El::IR(0, primal_dimension), ctx.cfg.split_factor))
      {
        const size_t p_begin = p_range.beg;
        const size_t p_end = p_range.end;
        Scoped_Timer p_timer(timers,
                             El::BuildString("p_", p_begin, "_", p_end));
        std::vector<Block_Diagonal_Matrix> G;
        G.reserve(p_end - p_begin);
        const auto F_begin = sdp.sdp_blocks_F.begin() + p_begin;
        const auto F_end = sdp.sdp_blocks_F.begin() + p_end;
        G.insert(G.end(), F_begin, F_end);
        if(verbosity >= Verbosity::trace && p_begin == 0)
          {
            print_allocation_message_per_node(env, "G",
                                              get_allocated_bytes(G));
          }
        // G := L_X_inv G
        {
          Scoped_Timer norm_timer(timers, "normalize_G");
          const auto G_normalizer
            = normalize_and_shift<Matrix_Normalization_Kind::COLUMNS>(
              G, bits, std::nullopt);
          norm_timer.stop();

          Scoped_Timer compute_residues_timer(timers, "compute_residues");
          auto &G_residues = ctx.G_residues(p_end - p_begin, El::HORIZONTAL);
          ctx.compute_residues(G, G_residues, block_timings_ms);
          compute_residues_timer.stop();

          constexpr auto side = El::LEFT;
          ctx.trmm(side, uplo, orientation, diag, ctx.L_X_inv_residues,
                   G_residues, G, verbosity, timers, block_timings_ms);

          Scoped_Timer remove_normalization_timer(timers,
                                                  "remove_normalization");
          restore_trmm_output(side, uplo, orientation, L_X_inv_normalizer,
                              G_normalizer, G);
        }
        // G := G L_Y
        {
          Scoped_Timer norm_timer(timers, "normalize_G");
          const auto G_normalizer
            = normalize_and_shift<Matrix_Normalization_Kind::ROWS>(
              G, bits, std::nullopt);
          norm_timer.stop();

          Scoped_Timer compute_residues_timer(timers, "compute_residues");
          auto &G_residues = ctx.G_residues(p_end - p_begin, El::VERTICAL);
          ctx.compute_residues(G, G_residues, block_timings_ms);
          compute_residues_timer.stop();

          constexpr auto side = El::RIGHT;
          ctx.trmm(side, uplo, orientation, diag, ctx.L_Y_residues, G_residues,
                   G, verbosity, timers, block_timings_ms);
          Scoped_Timer remove_normalization_timer(timers,
                                                  "remove_normalization");
          restore_trmm_output(side, uplo, orientation, L_Y_normalizer,
                              G_normalizer, G);
        }

        // Copy each G_p into a p-th column of the corresponding block of P
        Scoped_Timer reshape_timer(timers, "reshape");
        // lab (bfss #77), LAB_BATCHED_RESHAPE=1: the same updates as
        // El::Reshape below (zero, then QueueUpdate of each entry of G_p at
        // (i + j * dim, p)), but queued for all p of this chunk and sent in one
        // ProcessQueues per block instead of one per (block, p). Each entry of P
        // receives exactly one update, 0 + value, as with El::Reshape, so the
        // values are the same.
        static const bool batched_reshape = [] {
          const char *env = std::getenv("LAB_BATCHED_RESHAPE");
          return env != nullptr && std::string(env) == "1";
        }();
        if(batched_reshape)
          {
            for(size_t block = 0; block < block_info.num_blocks_local();
                 ++block)
              {
                auto &P_block = result.blocks.at(block);
                {
                  auto P_columns = El::View(
                    P_block, El::Range<int>(0, P_block.Height()),
                    El::Range<int>(p_begin, p_end));
                  El::Zero(P_columns);
                }
                size_t num_local = 0;
                for(size_t p = p_begin; p < p_end; ++p)
                  {
                    const auto &G_p_block = G.at(p - p_begin).blocks.at(block);
                    num_local += G_p_block.LocalHeight() * G_p_block.LocalWidth();
                  }
                P_block.Reserve(num_local);
                for(size_t p = p_begin; p < p_end; ++p)
                  {
                    const auto &G_p_block = G.at(p - p_begin).blocks.at(block);
                    const int dim = G_p_block.Height();
                    ASSERT_EQUAL(dim * dim, P_block.Height());
                    for(int jLoc = 0; jLoc < G_p_block.LocalWidth(); ++jLoc)
                      {
                        const int j = G_p_block.GlobalCol(jLoc);
                        for(int iLoc = 0; iLoc < G_p_block.LocalHeight();
                            ++iLoc)
                          {
                            const int i = G_p_block.GlobalRow(iLoc);
                            P_block.QueueUpdate(i + j * dim, p,
                                                G_p_block.GetLocal(iLoc, jLoc));
                          }
                      }
                  }
                P_block.ProcessQueues();
              }
            continue;
          }
        for(size_t block = 0; block < block_info.num_blocks_local(); ++block)
          {
            const auto global_block_index = block_info.block_indices.at(block);
            auto block_index_string = std::to_string(global_block_index);
            for(size_t p = p_begin; p < p_end; ++p)
              {
                const auto &G_p_block = G.at(p - p_begin).blocks.at(block);
                auto &P_block = result.blocks.at(block);
                // Write all elements of G_p into the p-th column of the resulting P matrix
                const int vec_height = P_block.Height();
                constexpr int vec_width = 1;
                const auto out_I = El::Range<int>(0, vec_height);
                const auto out_J = El::Range<int>(p, p + vec_width);

                auto vec_G_p_view = El::View(P_block, out_I, out_J);
                El::Reshape(vec_height, vec_width, G_p_block, vec_G_p_view);
              }
          }
      }

    return result;
  }

  void
  compute_S(const Environment &env, const SDP &sdp,
            const Block_Info &block_info,
            const Block_Diagonal_Matrix &X_cholesky,
            const Block_Diagonal_Matrix &Y_cholesky, const El::Grid &grid,
            Compute_S_Context &compute_S_context,
            El::DistMatrix<El::BigFloat> &S, Timers &timers,
            El::Matrix<int32_t> &block_timings_ms, const Verbosity verbosity)
  {
    Scoped_Timer timer(timers, "S");
    Block_Matrix P
      = initialize_P(env, sdp, block_info, grid, X_cholesky, Y_cholesky,
                     compute_S_context.initialize_P_context, timers,
                     block_timings_ms, verbosity);
    syrk_Q(env, P, compute_S_context.syrk_S_context, S, timers,
           block_timings_ms, verbosity);
  }
}
