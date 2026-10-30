#include "sdp_solve/SDP_Solver_Terminate_Reason.hxx"
#include "sdp_solve/Solver_Parameters.hxx"

#include <chrono>

void compute_feasible_and_termination(
  const Solver_Parameters &parameters, const El::BigFloat &primal_error,
  const El::BigFloat &dual_error, const El::BigFloat &duality_gap,
  const El::BigFloat &primal_step_length, const El::BigFloat &dual_step_length,
  const int &iteration,
  const std::chrono::time_point<std::chrono::high_resolution_clock>
    &solver_start_time,
  bool &is_primal_and_dual_feasible,
  SDP_Solver_Terminate_Reason &terminate_reason, bool &terminate_now,
  int64_t &dual_jump_iteration)
{
  const bool is_dual_feasible(dual_error < parameters.dual_error_threshold),
    is_primal_feasible(primal_error < parameters.primal_error_threshold);
  is_primal_and_dual_feasible = (is_primal_feasible && is_dual_feasible);

  const bool is_optimal(duality_gap < parameters.duality_gap_threshold);

  // lab #228: dual_jump_iteration is 0 before any jump, the iteration of
  // the jump during the extra steps, and minus that once they were
  // abandoned because the iterate was no longer dual feasible by
  // dualErrorThreshold (not expected: the dual residues stay zero after a
  // full dual step). Then the usual rules apply for the rest of the run.
  if(dual_jump_iteration > 0 && !is_dual_feasible)
    {
      if(El::mpi::Rank() == 0)
        El::Output("lab #228: dual error ", dual_error,
                   " above dualErrorThreshold at iteration ", iteration,
                   " after the dual feasible jump at iteration ",
                   dual_jump_iteration, "; extra steps abandoned");
      dual_jump_iteration = -dual_jump_iteration;
    }

  terminate_now = true;
  if(is_primal_and_dual_feasible && is_optimal)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::PrimalDualOptimal;
    }
  else if(is_dual_feasible && parameters.find_dual_feasible)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::DualFeasible;
    }
  else if(is_primal_feasible && parameters.find_primal_feasible)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::PrimalFeasible;
    }
  else if(dual_jump_iteration > 0)
    {
      // lab #228: extra steps after a dual feasible jump (see
      // Solver_Parameters::dual_jump_extra_iterations). Stop at the end of
      // the extra steps, and also, rather than with another reason, at any
      // other limit: the iterate is dual feasible, so its functional is a
      // valid result of this run. (Loss of dual feasibility is handled
      // above: the extra steps end and the usual rules apply.)
      if(iteration
                >= dual_jump_iteration + parameters.dual_jump_extra_iterations
              || iteration > parameters.max_iterations
              || std::chrono::duration_cast<std::chrono::seconds>(
                   std::chrono::high_resolution_clock::now()
                   - solver_start_time)
                     .count()
                   >= parameters.max_runtime
              || primal_step_length < parameters.min_primal_step
              || dual_step_length < parameters.min_dual_step)
        {
          terminate_reason
            = SDP_Solver_Terminate_Reason::DualFeasibleJumpDetected;
        }
      else
        {
          terminate_now = false;
        }
    }
  else if(dual_step_length == El::BigFloat(1)
          && parameters.detect_dual_feasible_jump)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::DualFeasibleJumpDetected;
      // lab #228: a jump within dual_jump_extra_within steps of the start
      // of this run (iteration - 1 steps were taken) starts the extra steps,
      // if the iterate is dual feasible by dualErrorThreshold.
      if(parameters.dual_jump_extra_iterations > 0 && dual_jump_iteration == 0
         && iteration - 1 <= parameters.dual_jump_extra_within
         && is_dual_feasible && iteration <= parameters.max_iterations)
        {
          dual_jump_iteration = iteration;
          terminate_now = false;
          if(El::mpi::Rank() == 0)
            El::Output("lab #228: dual feasible jump after ", iteration - 1,
                       " steps; continuing ",
                       parameters.dual_jump_extra_iterations,
                       " more steps (dual error ", dual_error, ")");
        }
    }
  else if(primal_step_length == El::BigFloat(1)
          && parameters.detect_primal_feasible_jump)
    {
      terminate_reason
        = SDP_Solver_Terminate_Reason::PrimalFeasibleJumpDetected;
    }
  else if(iteration > parameters.max_iterations)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::MaxIterationsExceeded;
    }
  else if(std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::high_resolution_clock::now() - solver_start_time)
            .count()
          >= parameters.max_runtime)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::MaxRuntimeExceeded;
    }
  else if(iteration > 1 && primal_step_length < parameters.min_primal_step)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::PrimalStepTooSmall;
    }
  else if(iteration > 1 && dual_step_length < parameters.min_dual_step)
    {
      terminate_reason = SDP_Solver_Terminate_Reason::DualStepTooSmall;
    }
  else
    {
      terminate_now = false;
    }
  El::byte terminate_byte(terminate_now);
  // Time varies between cores, so follow the decision of the root.
  El::mpi::Broadcast(terminate_byte, 0, El::mpi::COMM_WORLD);
  terminate_now = static_cast<bool>(terminate_byte);
}
