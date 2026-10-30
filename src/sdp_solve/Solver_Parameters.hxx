#pragma once
// Parameters to control the behavior of an SDPSolver. See the manual
// for a detailed description of each.
//

#include <El.hpp>
#include <filesystem>
#include <boost/property_tree/ptree.hpp>
#include <boost/program_options.hpp>

#include <iostream>

struct Solver_Parameters
{
  int64_t max_iterations, max_runtime, checkpoint_interval;
  size_t max_shared_memory_bytes;
  bool find_primal_feasible, find_dual_feasible, detect_primal_feasible_jump,
    detect_dual_feasible_jump;
  // Per-block Cholesky and L^{-1}B by multimodular BLAS,
  // see sdp_solve/SDP_Solver/run/multimodular/Readme.md
  bool multimodular_trsm = false;
  // Experimental (lab #164): store the free variable matrix B compactly
  // (nonzero elements only), see sdp_solve/Sparse_Block_Matrix.hxx
  bool sparse_B = false;
  // Experimental (lab #228, tail without restarts): when a dual feasible
  // jump is detected after at most dual_jump_extra_within steps of this
  // run, keep iterating dual_jump_extra_iterations more steps before
  // terminating with DualFeasibleJumpDetected. Every iterate after the
  // jump stays dual feasible, so its functional is as valid as the one at
  // the jump, at a smaller mu. 0 (default): stop at the jump, as before.
  int64_t dual_jump_extra_iterations = 0;
  int64_t dual_jump_extra_within = std::numeric_limits<int64_t>::max();
  size_t precision;

  El::BigFloat duality_gap_threshold, primal_error_threshold,
    dual_error_threshold, initial_matrix_scale_primal,
    initial_matrix_scale_dual, feasible_centering_parameter,
    infeasible_centering_parameter, step_length_reduction, max_complementarity,
    min_primal_step, min_dual_step;

  std::filesystem::path checkpoint_in, checkpoint_out;
  Solver_Parameters() = default;
  boost::program_options::options_description options();
};

std::ostream &operator<<(std::ostream &os, const Solver_Parameters &p);

boost::property_tree::ptree to_property_tree(const Solver_Parameters &p);
