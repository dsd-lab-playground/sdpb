// Resident mode: sdpb --controlDir <dir> (lab #197). See Readme.md.

#include "Resident.hxx"
#include "Control_Channel.hxx"

#include "pmp/max_normalization_index.hxx"
#include "pmp2sdp/Block_File_Format.hxx"
#include "pmp2sdp/Dual_Constraint_Group.hxx"
#include "pmp2sdp/Output_SDP/Output_SDP.hxx"
#include "pmp_read/PMP_File_Parse_Result.hxx"
#include "pmp_read/pmp_read.hxx"
#include "sdp_solve/SDP/read_block_data/Block_Data_Parse_Result.hxx"
#include "sdp_solve/SDP/read_block_data/SDP_Block_Data.hxx"
#include "sdp_solve/SDP/set_bases_blocks.hxx"
#include "sdp_solve/sdp_solve.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/ostream/set_stream_precision.hxx"

#include <El.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

// Defined elsewhere in sdpb / pmp2sdp_lib / sdp_solve
void save_solution(
  const SDP_Solver &solver,
  const SDP_Solver_Terminate_Reason &terminate_reason, const int64_t &runtime,
  const fs::path &out_directory, const Write_Solution &write_solution,
  const std::vector<size_t> &block_indices,
  const std::optional<std::vector<El::BigFloat>> &normalization,
  const Verbosity &verbosity);
void write_block_data(std::ostream &os, const Dual_Constraint_Group &group,
                      Block_File_Format format);
void write_objectives_json(std::ostream &output_stream,
                           const El::BigFloat &objective_const,
                           const std::vector<El::BigFloat> &dual_objective_b);
void write_normalization_json(std::ostream &output_stream,
                              const std::vector<El::BigFloat> &normalization);
Block_Data_Parse_Result parse_block_data(std::istream &block_stream,
                                         Block_File_Format format);
void read_objectives_from_stream(
  const El::Grid &grid, std::istream &objectives_stream,
  El::BigFloat &objective_const,
  El::DistMatrix<El::BigFloat> &dual_objective_b);
void read_objectives(const fs::path &sdp_path, const El::Grid &grid,
                     El::BigFloat &objective_const,
                     El::DistMatrix<El::BigFloat> &dual_objective_b,
                     Timers &timers);
std::optional<std::vector<El::BigFloat>>
read_normalization(const fs::path &sdp_path, Timers &timers);

namespace
{
  using Clock = std::chrono::high_resolution_clock;
  using Json_Writer = rapidjson::Writer<rapidjson::StringBuffer>;

  double seconds_since(const Clock::time_point &start)
  {
    return std::chrono::duration<double>(Clock::now() - start).count();
  }

  std::string read_file(const fs::path &path)
  {
    std::ifstream in(path, std::ios::binary);
    ASSERT(in.good(), "Cannot open ", path);
    std::stringstream ss;
    ss << in.rdbuf();
    return ss.str();
  }

  void broadcast_string(std::string &s, const int root = 0,
                        const El::mpi::Comm &comm = El::mpi::COMM_WORLD)
  {
    size_t size = s.size();
    El::mpi::Broadcast(size, root, comm);
    s.resize(size);
    if(size > 0)
      MPI_Bcast(s.data(), size, MPI_CHAR, root, comm.comm);
  }

  std::string to_string(const El::BigFloat &x)
  {
    std::ostringstream os;
    set_stream_precision(os);
    os << x;
    return os.str();
  }

  // A decimal number, or a rational p/q
  El::BigFloat parse_number(const std::string &s)
  {
    const auto slash = s.find('/');
    if(slash == std::string::npos)
      return El::BigFloat(s);
    return El::BigFloat(s.substr(0, slash))
           / El::BigFloat(s.substr(slash + 1));
  }

  std::string number_string(const rapidjson::Value &v)
  {
    if(v.IsString())
      return v.GetString();
    ASSERT(v.IsNumber(), "Expected a number or a string");
    // Exact for integers; give other values as strings
    ASSERT(v.IsInt64(), "Give non-integer numbers as strings");
    return std::to_string(v.GetInt64());
  }

  std::string get_string(const rapidjson::Value &v, const char *key,
                         const std::string &default_value)
  {
    if(!v.HasMember(key))
      return default_value;
    ASSERT(v[key].IsString(), "'", key, "' must be a string");
    return v[key].GetString();
  }

  bool get_bool(const rapidjson::Value &v, const char *key,
                const bool default_value)
  {
    if(!v.HasMember(key))
      return default_value;
    ASSERT(v[key].IsBool(), "'", key, "' must be true or false");
    return v[key].GetBool();
  }

  std::vector<std::string>
  get_string_list(const rapidjson::Value &v, const char *key)
  {
    std::vector<std::string> result;
    if(!v.HasMember(key))
      return result;
    if(v[key].IsString())
      {
        result.emplace_back(v[key].GetString());
        return result;
      }
    ASSERT(v[key].IsArray(), "'", key, "' must be a string or an array");
    for(const auto &x : v[key].GetArray())
      result.push_back(number_string(x));
    return result;
  }

  void write_bigfloat(Json_Writer &w, const El::BigFloat &x)
  {
    const auto s = to_string(x);
    w.String(s.c_str(), s.size());
  }

  // Value of option name (or short_name) in a space-separated command line
  std::optional<std::string>
  find_option(const std::vector<std::string> &tokens, const std::string &name,
              const std::string &short_name)
  {
    for(size_t i = 0; i < tokens.size(); ++i)
      {
        const auto &t = tokens.at(i);
        if(t == name || t == short_name)
          {
            ASSERT(i + 1 < tokens.size(), "Missing value for ", t);
            return tokens.at(i + 1);
          }
        if(t.rfind(name + "=", 0) == 0)
          return t.substr(name.size() + 1);
      }
    return std::nullopt;
  }

  // What the resident process needs to know about the files of an SDP
  // directory written by pmp2sdp. Read on rank 0, the same on all ranks.
  struct Sdp_Files
  {
    fs::path dir;
    size_t num_blocks = 0;
    int64_t max_num_poles = -1;
    Block_File_Format format = Block_File_Format::bin;
    // Source PMP file of each block (pmp_info.json), empty if unknown
    std::vector<fs::path> block_paths;
    // Text of objectives.json and normalization.json ("" if absent)
    std::string objectives_text, normalization_text;
    bool is_directory = true;

    explicit Sdp_Files(const fs::path &sdp_dir) : dir(sdp_dir)
    {
      std::string control, pmp_info;
      int dir_flag = 1;
      if(El::mpi::Rank() == 0)
        {
          dir_flag = fs::is_directory(sdp_dir) ? 1 : 0;
          if(dir_flag)
            {
              control = read_file(sdp_dir / "control.json");
              if(fs::exists(sdp_dir / "pmp_info.json"))
                pmp_info = read_file(sdp_dir / "pmp_info.json");
              objectives_text = read_file(sdp_dir / "objectives.json");
              if(fs::exists(sdp_dir / "normalization.json"))
                normalization_text = read_file(sdp_dir / "normalization.json");
            }
        }
      El::mpi::Broadcast(dir_flag, 0, El::mpi::COMM_WORLD);
      is_directory = dir_flag != 0;
      if(!is_directory)
        return;
      broadcast_string(control);
      broadcast_string(pmp_info);
      broadcast_string(objectives_text);
      broadcast_string(normalization_text);

      rapidjson::Document control_doc;
      control_doc.Parse(control.c_str());
      ASSERT(!control_doc.HasParseError(), "Cannot parse ",
             sdp_dir / "control.json");
      num_blocks = control_doc["num_blocks"].GetInt();
      if(control_doc.HasMember("command"))
        {
          std::vector<std::string> tokens;
          std::istringstream iss(control_doc["command"].GetString());
          std::string token;
          while(iss >> token)
            tokens.push_back(token);
          const auto poles = find_option(tokens, "--maxNumPoles", "-n");
          if(poles.has_value())
            max_num_poles = std::stoll(poles.value());
          const auto output_format
            = find_option(tokens, "--outputFormat", "-f");
          if(output_format.has_value() && output_format.value() == "json")
            format = Block_File_Format::json;
        }
      if(!pmp_info.empty())
        {
          rapidjson::Document doc;
          doc.Parse(pmp_info.c_str());
          ASSERT(!doc.HasParseError() && doc.IsArray(), "Cannot parse ",
                 sdp_dir / "pmp_info.json");
          block_paths.resize(doc.Size());
          for(const auto &entry : doc.GetArray())
            block_paths.at(entry["index"].GetInt())
              = fs::path(entry["path"].GetString());
          ASSERT_EQUAL(block_paths.size(), num_blocks,
                       sdp_dir / "pmp_info.json");
        }
    }
  };

  struct Iterate
  {
    Block_Vector x, y;
    Block_Diagonal_Matrix X, Y;
  };

  // A block whose data is c = sum_i w_i c_i, B = sum_i w_i B_i
  struct Family
  {
    size_t block_index = 0;
    int owner_root = 0;
    size_t num_components = 0;
    // A symmetric form_dim x form_dim matrix of components, given by its
    // upper triangle (row-major), if form_dim > 0
    size_t form_dim = 0;
    std::vector<std::string> paths;
    // Only on owner_root:
    std::vector<El::Matrix<El::BigFloat>> c, B;
    std::array<El::Matrix<El::BigFloat>, 2> bilinear_bases;
    // Sample scaling of the first sample point of each component
    std::vector<El::BigFloat> first_scaling;
  };

  class Resident_State
  {
  public:
    Resident_State(const Environment &env, const SDPB_Parameters &parameters,
                   const std::vector<std::string> &startup_args,
                   mp_bitcnt_t startup_precision,
                   const Block_Info &block_info);

    std::string ready_json(double load_seconds) const;
    // Execute one command and write its result (an object). Returns false if
    // the resident loop should stop.
    bool execute(const rapidjson::Value &command, Json_Writer &w);
    void checkpoint(const fs::path &dir,
                    const boost::property_tree::ptree &properties);
    const SDPB_Parameters &startup_parameters() const { return parameters; }

  private:
    const Environment &env;
    SDPB_Parameters parameters;
    std::vector<std::string> startup_args;
    mp_bitcnt_t startup_precision;
    const Block_Info &block_info;
    El::Grid grid;
    Timers load_timers;
    SDP sdp;
    SDP_Solver solver;
    bool loaded_checkpoint = false;
    std::unique_ptr<BigInt_Shared_Memory_Syrk_Context> syrk_context;
    // World rank of the root of the group that owns each block
    std::vector<int> owner_root;
    Sdp_Files files;
    // Normalization, on all ranks (sdp.normalization is on rank 0 only)
    std::optional<std::vector<El::BigFloat>> normalization;
    fs::path block_timings_path;
    std::map<std::string, Iterate> iterates;
    std::map<std::string, Family> families;
    size_t num_solves = 0;

    bool is_group_root() const { return grid.Rank() == 0; }
    size_t num_blocks() const { return block_info.dimensions.size(); }
    std::optional<size_t> local_index(size_t block_index) const;

    void cmd_solve(const rapidjson::Value &command, Json_Writer &w,
                   bool &keep_running);
    void cmd_update_sdp(const rapidjson::Value &command, Json_Writer &w);
    void cmd_update_pmp(const rapidjson::Value &command, Json_Writer &w);
    void cmd_load_family(const rapidjson::Value &command, Json_Writer &w);
    void cmd_set_coefficients(const rapidjson::Value &command,
                              Json_Writer &w);
    void cmd_negate_normalization(Json_Writer &w);
    void cmd_save_iterate(const rapidjson::Value &command, Json_Writer &w);
    void cmd_restore_iterate(const rapidjson::Value &command, Json_Writer &w);
    void cmd_status(Json_Writer &w);

    void replace_blocks(const std::vector<Dual_Constraint_Group> &groups,
                        const std::vector<size_t> &changed);
    void set_objectives_text(const std::string &text);
    El::Matrix<El::BigFloat> y_on_group_root() const;
    void write_derived(const std::vector<std::string> &names,
                       Json_Writer &w) const;
  };

  Resident_State::Resident_State(const Environment &env,
                                 const SDPB_Parameters &parameters,
                                 const std::vector<std::string> &startup_args,
                                 const mp_bitcnt_t startup_precision,
                                 const Block_Info &block_info)
      : env(env),
        parameters(parameters),
        startup_args(startup_args),
        startup_precision(startup_precision),
        block_info(block_info),
        grid(block_info.mpi_comm.value),
        load_timers(env, parameters.verbosity),
        sdp(parameters.sdp_path, block_info, grid, load_timers,
            parameters.solver.sparse_B),
        solver(parameters.solver, parameters.verbosity,
               parameters.require_initial_checkpoint, block_info, grid,
               sdp.dual_objective_b.Height()),
        files(parameters.sdp_path)
  {
    loaded_checkpoint = solver.current_generation != 0
                        || solver.backup_generation.has_value();
    owner_root.assign(num_blocks(), -1);
    if(is_group_root())
      for(const auto index : block_info.block_indices)
        owner_root.at(index) = El::mpi::Rank();
    El::mpi::AllReduce(owner_root.data(), owner_root.size(), El::mpi::MAX,
                       El::mpi::COMM_WORLD);

    // Normalization on all ranks
    {
      std::vector<El::BigFloat> n;
      int has = 0;
      if(El::mpi::Rank() == 0 && sdp.normalization.has_value())
        {
          n = sdp.normalization.value();
          has = 1;
        }
      El::mpi::Broadcast(has, 0, El::mpi::COMM_WORLD);
      if(has)
        {
          size_t size = n.size();
          El::mpi::Broadcast(size, 0, El::mpi::COMM_WORLD);
          n.resize(size);
          El::mpi::Broadcast(n.data(), size, 0, El::mpi::COMM_WORLD);
          normalization = n;
        }
    }
    if(!block_info.block_timings_filename.empty())
      block_timings_path = block_info.block_timings_filename;
    else if(fs::exists(parameters.solver.checkpoint_out / "block_timings"))
      block_timings_path = parameters.solver.checkpoint_out / "block_timings";
  }

  std::optional<size_t> Resident_State::local_index(size_t block_index) const
  {
    const auto &indices = block_info.block_indices;
    const auto it = std::find(indices.begin(), indices.end(), block_index);
    if(it == indices.end())
      return std::nullopt;
    return std::distance(indices.begin(), it);
  }

  std::string Resident_State::ready_json(const double load_seconds) const
  {
    rapidjson::StringBuffer sb;
    Json_Writer w(sb);
    w.StartObject();
    w.Key("status");
    w.String("ready");
    char host[256] = {0};
    gethostname(host, sizeof(host) - 1);
    w.Key("host");
    w.String(host);
    w.Key("pid");
    w.Int64(getpid());
    w.Key("sdpDir");
    w.String(parameters.sdp_path.string().c_str());
    w.Key("num_blocks");
    w.Uint64(num_blocks());
    w.Key("dual_dimension");
    w.Int64(sdp.dual_objective_b.Height());
    w.Key("precision");
    w.Uint64(parameters.solver.precision);
    w.Key("mpi_processes");
    w.Int(El::mpi::Size());
    w.Key("nodes");
    w.Int(env.num_nodes());
    w.Key("loaded_checkpoint");
    w.Bool(loaded_checkpoint);
    w.Key("load_seconds");
    w.Double(load_seconds);
    w.EndObject();
    return sb.GetString();
  }

  void
  Resident_State::checkpoint(const fs::path &dir,
                             const boost::property_tree::ptree &properties)
  {
    solver.save_checkpoint(dir, parameters.verbosity, properties);
    // Keep the block distribution with the checkpoint, as the file route
    // does, so that a restart from it gives the same layout
    if(El::mpi::Rank() == 0 && !block_timings_path.empty()
       && fs::exists(block_timings_path)
       && !fs::exists(dir / "block_timings"))
      fs::copy_file(block_timings_path, dir / "block_timings");
  }

  El::Matrix<El::BigFloat> Resident_State::y_on_group_root() const
  {
    // y is duplicated among the blocks: gather the first local block's copy
    // to the root of each group (collective in each group)
    const auto &y_dist = solver.y.blocks.at(0);
    El::DistMatrix<El::BigFloat, El::CIRC, El::CIRC> y_circ(y_dist);
    if(is_group_root())
      return y_circ.Matrix();
    return {};
  }

  bool Resident_State::execute(const rapidjson::Value &command, Json_Writer &w)
  {
    ASSERT(command.IsObject(), "A command must be a JSON object");
    const std::string name = get_string(command, "cmd", "");
    bool keep_running = true;
    const auto start = Clock::now();
    w.StartObject();
    w.Key("cmd");
    w.String(name.c_str());
    if(name == "solve")
      cmd_solve(command, w, keep_running);
    else if(name == "update_sdp")
      cmd_update_sdp(command, w);
    else if(name == "update_pmp")
      cmd_update_pmp(command, w);
    else if(name == "load_family")
      cmd_load_family(command, w);
    else if(name == "set_coefficients")
      cmd_set_coefficients(command, w);
    else if(name == "negate_normalization")
      cmd_negate_normalization(w);
    else if(name == "save_iterate")
      cmd_save_iterate(command, w);
    else if(name == "restore_iterate")
      cmd_restore_iterate(command, w);
    else if(name == "drop_iterate")
      {
        iterates.erase(get_string(command, "name", ""));
      }
    else if(name == "checkpoint")
      {
        const fs::path dir = get_string(
          command, "dir", parameters.solver.checkpoint_out.string());
        checkpoint(dir, to_property_tree(parameters));
        w.Key("dir");
        w.String(dir.string().c_str());
      }
    else if(name == "status")
      cmd_status(w);
    else if(name == "quit")
      {
        if(get_bool(command, "checkpoint", false))
          checkpoint(parameters.solver.checkpoint_out,
                     to_property_tree(parameters));
        keep_running = false;
      }
    else
      RUNTIME_ERROR("Unknown command: '", name, "'");
    El::mpi::Barrier(El::mpi::COMM_WORLD);
    w.Key("seconds");
    w.Double(seconds_since(start));
    w.EndObject();
    return keep_running;
  }

  // solve: run the solver from the current (or a named) iterate.
  // "args": an sdpb command line for this solve (outDir, checkpointDir,
  // thresholds, maxIterations, ...); options it lacks come from the resident
  // process's command line.
  void Resident_State::cmd_solve(const rapidjson::Value &command,
                                 Json_Writer &w, bool &keep_running)
  {
    const auto start = Clock::now();
    const auto args = get_string_list(command, "args");
    // Parse at the precision sdpb parses its command line at
    std::optional<SDPB_Parameters> parsed;
    {
      const mp_bitcnt_t working_precision = mpf_get_default_prec();
      mpf_set_default_prec(startup_precision);
      try
        {
          parsed.emplace(std::vector<std::vector<std::string>>{args,
                                                               startup_args},
                         true);
        }
      catch(...)
        {
          mpf_set_default_prec(working_precision);
          throw;
        }
      mpf_set_default_prec(working_precision);
      ASSERT_EQUAL(mpf_get_default_prec(), working_precision);
    }
    const SDPB_Parameters &p = parsed.value();
    ASSERT_EQUAL(p.solver.precision, parameters.solver.precision,
                 "solve: --precision is fixed in resident mode");
    ASSERT_EQUAL(p.solver.sparse_B, parameters.solver.sparse_B,
                 "solve: --sparseB is fixed in resident mode");
    ASSERT_EQUAL(p.solver.max_shared_memory_bytes,
                 parameters.solver.max_shared_memory_bytes,
                 "solve: --maxSharedMemory is fixed in resident mode");
    ASSERT_EQUAL(p.proc_granularity, parameters.proc_granularity);

    // Starting point
    const rapidjson::Value *start_value
      = command.HasMember("start") ? &command["start"] : nullptr;
    if(start_value != nullptr && start_value->IsString())
      {
        const std::string s = start_value->GetString();
        if(s == "initial")
          {
            solver.X.set_zero();
            solver.Y.set_zero();
            for(auto &block : solver.x.blocks)
              El::Zero(block);
            for(auto &block : solver.y.blocks)
              El::Zero(block);
            solver.X.add_diagonal(p.solver.initial_matrix_scale_primal);
            solver.Y.add_diagonal(p.solver.initial_matrix_scale_dual);
          }
        else if(s == "checkpoint")
          {
            solver.load_checkpoint(p.solver.checkpoint_in, block_info,
                                   p.verbosity, true);
          }
        else
          ASSERT(s == "current", "solve: unknown start '", s, "'");
      }
    else if(start_value != nullptr && start_value->IsObject())
      {
        const auto name = get_string(*start_value, "iterate", "");
        const auto it = iterates.find(name);
        ASSERT(it != iterates.end(), "solve: no iterate named '", name, "'");
        solver.x = it->second.x;
        solver.X = it->second.X;
        solver.y = it->second.y;
        solver.Y = it->second.Y;
      }

    if(p.verbosity >= Verbosity::regular && El::mpi::Rank() == 0)
      El::Output(boost::posix_time::second_clock::local_time(),
                 " Resident solve ", num_solves + 1, ", outDir ",
                 p.out_directory);

    Timers timers(env, p.verbosity);
    const boost::property_tree::ptree parameters_tree(to_property_tree(p));
    const auto iterations_json_path = p.out_directory / "iterations.json";
    El::Matrix<int32_t> block_timings_ms;
    const auto run_start = Clock::now();
    const SDP_Solver_Terminate_Reason reason(solver.run(
      env, p.solver, p.verbosity, parameters_tree, block_info, sdp, grid,
      run_start, iterations_json_path, timers, block_timings_ms,
      &syrk_context));
    const double run_seconds = seconds_since(run_start);
    ++num_solves;

    if(p.verbosity >= Verbosity::regular && El::mpi::Rank() == 0)
      {
        set_stream_precision(std::cout);
        std::cout << "-----" << reason << "-----\n"
                  << '\n'
                  << "primalObjective = " << solver.primal_objective << '\n'
                  << "dualObjective   = " << solver.dual_objective << '\n'
                  << "dualityGap      = " << solver.duality_gap << '\n'
                  << "primalError     = " << solver.primal_error() << '\n'
                  << "dualError       = " << solver.dual_error << '\n'
                  << '\n';
      }

    const bool sigterm
      = reason == SDP_Solver_Terminate_Reason::SIGTERM_Received;
    const auto checkpoint_start = Clock::now();
    if(sigterm || get_bool(command, "checkpoint", false))
      checkpoint(p.solver.checkpoint_out, parameters_tree);
    const double checkpoint_seconds = seconds_since(checkpoint_start);

    const auto solution_start = Clock::now();
    if(get_bool(command, "write_solution", true))
      {
        const auto runtime = static_cast<int64_t>(seconds_since(run_start));
        save_solution(solver, reason, runtime, p.out_directory,
                      p.write_solution, block_info.block_indices,
                      sdp.normalization, p.verbosity);
      }
    const double solution_seconds = seconds_since(solution_start);

    std::ostringstream reason_ss;
    reason_ss << reason;
    w.Key("terminateReason");
    w.String(reason_ss.str().c_str());
    w.Key("primalObjective");
    write_bigfloat(w, solver.primal_objective);
    w.Key("dualObjective");
    write_bigfloat(w, solver.dual_objective);
    w.Key("dualityGap");
    write_bigfloat(w, solver.duality_gap);
    w.Key("primalError");
    write_bigfloat(w, solver.primal_error());
    w.Key("dualError");
    write_bigfloat(w, solver.dual_error);
    w.Key("steps");
    w.Int64(solver.num_steps);
    w.Key("outDir");
    w.String(p.out_directory.string().c_str());
    w.Key("solve_index");
    w.Uint64(num_solves);
    w.Key("timing");
    w.StartObject();
    w.Key("setup");
    w.Double(std::chrono::duration<double>(run_start - start).count());
    w.Key("run");
    w.Double(run_seconds);
    w.Key("checkpoint");
    w.Double(checkpoint_seconds);
    w.Key("solution");
    w.Double(solution_seconds);
    w.EndObject();
    write_derived(get_string_list(command, "derived"), w);

    if(sigterm)
      keep_running = false;
  }

  // Derived quantities after a solve:
  //   "y", "z": the dual vector y, and z (y with the normalization restored);
  //   "family_values": for each family and component i, c_i - B_i.y over the
  //     block's constraint rows (for a 1x1 constant block: the functional on
  //     the component, times the positive sample scaling; "pmp_values"
  //     divides by it).
  // More can be added here (plug-in hook).
  void Resident_State::write_derived(const std::vector<std::string> &names,
                                     Json_Writer &w) const
  {
    if(names.empty())
      return;
    const auto y = y_on_group_root();
    w.Key("derived");
    w.StartObject();
    for(const auto &name : names)
      {
        if(name == "y" || name == "z")
          {
            // rank 0 is the root of its group, so it has y
            std::vector<std::string> values;
            if(El::mpi::Rank() == 0)
              {
                if(name == "y")
                  {
                    for(int i = 0; i < y.Height(); ++i)
                      values.push_back(to_string(y(i, 0)));
                  }
                else
                  {
                    ASSERT(normalization.has_value(),
                           "z needs a normalization");
                    const auto &n = normalization.value();
                    const auto max_index = max_normalization_index(n);
                    std::vector<El::BigFloat> z(y.Height() + 1);
                    for(int i = 0; i < max_index; ++i)
                      z.at(i) = y(i, 0);
                    z.at(max_index) = 0;
                    for(int i = max_index; i < y.Height(); ++i)
                      z.at(i + 1) = y(i, 0);
                    // Same arithmetic as save_solution()
                    El::Matrix<El::BigFloat> z_view, n_view;
                    z_view.LockedAttach(z.size(), 1, z.data(), z.size());
                    n_view.LockedAttach(n.size(), 1, n.data(), n.size());
                    auto nz = El::Dot(n_view, z_view);
                    z.at(max_index) = (1 - nz) / n.at(max_index);
                    for(const auto &x : z)
                      values.push_back(to_string(x));
                  }
                w.Key(name.c_str());
                w.StartArray();
                for(const auto &s : values)
                  w.String(s.c_str(), s.size());
                w.EndArray();
              }
          }
        else if(name == "family_values")
          {
            if(El::mpi::Rank() == 0)
              {
                w.Key("family_values");
                w.StartObject();
              }
            for(const auto &[family_name, family] : families)
              {
                // On the owner root: one JSON text per family
                std::string text;
                if(El::mpi::Rank() == family.owner_root)
                  {
                    rapidjson::StringBuffer sb;
                    Json_Writer fw(sb);
                    fw.StartObject();
                    fw.Key("block");
                    fw.Uint64(family.block_index);
                    std::vector<std::vector<El::BigFloat>> values;
                    for(size_t i = 0; i < family.num_components; ++i)
                      {
                        const auto &c = family.c.at(i);
                        const auto &B = family.B.at(i);
                        std::vector<El::BigFloat> v(c.Height());
                        for(int row = 0; row < c.Height(); ++row)
                          {
                            El::BigFloat By(0);
                            for(int col = 0; col < B.Width(); ++col)
                              By += B(row, col) * y(col, 0);
                            v.at(row) = c(row, 0) - By;
                          }
                        values.push_back(v);
                      }
                    fw.Key("values");
                    fw.StartArray();
                    for(const auto &v : values)
                      {
                        fw.StartArray();
                        for(const auto &x : v)
                          write_bigfloat(fw, x);
                        fw.EndArray();
                      }
                    fw.EndArray();
                    if(!values.empty() && values.front().size() == 1)
                      {
                        fw.Key("pmp_values");
                        fw.StartArray();
                        for(size_t i = 0; i < values.size(); ++i)
                          write_bigfloat(fw, values.at(i).at(0)
                                               / family.first_scaling.at(i));
                        fw.EndArray();
                      }
                    fw.EndObject();
                    text = sb.GetString();
                  }
                // Send to rank 0
                if(family.owner_root != 0)
                  {
                    if(El::mpi::Rank() == family.owner_root)
                      {
                        uint64_t size = text.size();
                        MPI_Send(&size, 1, MPI_UINT64_T, 0, 7,
                                 El::mpi::COMM_WORLD.comm);
                        MPI_Send(text.data(), size, MPI_CHAR, 0, 8,
                                 El::mpi::COMM_WORLD.comm);
                      }
                    else if(El::mpi::Rank() == 0)
                      {
                        uint64_t size = 0;
                        MPI_Recv(&size, 1, MPI_UINT64_T, family.owner_root, 7,
                                 El::mpi::COMM_WORLD.comm, MPI_STATUS_IGNORE);
                        text.resize(size);
                        MPI_Recv(text.data(), size, MPI_CHAR,
                                 family.owner_root, 8,
                                 El::mpi::COMM_WORLD.comm, MPI_STATUS_IGNORE);
                      }
                  }
                if(El::mpi::Rank() == 0)
                  {
                    w.Key(family_name.c_str());
                    w.RawValue(text.c_str(), text.size(),
                               rapidjson::kObjectType);
                  }
              }
            if(El::mpi::Rank() == 0)
              w.EndObject();
          }
        else
          RUNTIME_ERROR("Unknown derived quantity: '", name, "'");
      }
    w.EndObject();
  }

  // Replace the changed blocks by converted Dual_Constraint_Groups.
  // groups: the converted blocks held by this rank (any of them may be
  // unchanged). Each changed block must be held by exactly one rank.
  // Its data goes through the same serialization as a block_data file
  // (the SDP's --outputFormat), so the blocks in memory equal those SDPB
  // would read from pmp2sdp's output.
  void
  Resident_State::replace_blocks(const std::vector<Dual_Constraint_Group> &groups,
                                 const std::vector<size_t> &changed)
  {
    const int rank = El::mpi::Rank();
    std::vector<int> holder(num_blocks(), -1);
    std::set<size_t> changed_set(changed.begin(), changed.end());
    int bad_shape = 0;
    std::map<size_t, std::string> bytes;
    for(const auto &group : groups)
      {
        if(changed_set.count(group.block_index) == 0)
          continue;
        holder.at(group.block_index) = rank;
        if(group.dim != block_info.dimensions.at(group.block_index)
           || group.num_points != block_info.num_points.at(group.block_index))
          bad_shape = 1;
        std::ostringstream os(std::ios::binary);
        write_block_data(os, group, files.format);
        bytes[group.block_index] = os.str();
      }
    El::mpi::AllReduce(holder.data(), holder.size(), El::mpi::MAX,
                       El::mpi::COMM_WORLD);
    bad_shape = El::mpi::AllReduce(bad_shape, El::mpi::MAX,
                                   El::mpi::COMM_WORLD);
    ASSERT(bad_shape == 0, "A new block has a different size from the old "
                           "one: the block structure must stay the same");
    for(const auto index : changed)
      ASSERT(holder.at(index) >= 0, "No data for block ", index);

    // Send each block to the root of the group that owns it
    std::vector<MPI_Request> requests;
    std::vector<uint64_t> sizes;
    sizes.reserve(bytes.size());
    for(const auto &[index, data] : bytes)
      {
        const int dest = owner_root.at(index);
        if(dest == rank)
          continue;
        sizes.push_back(data.size());
        requests.emplace_back();
        MPI_Isend(&sizes.back(), 1, MPI_UINT64_T, dest, 2 * index,
                  El::mpi::COMM_WORLD.comm, &requests.back());
        requests.emplace_back();
        MPI_Isend(data.data(), data.size(), MPI_CHAR, dest, 2 * index + 1,
                  El::mpi::COMM_WORLD.comm, &requests.back());
      }
    for(const auto index : changed)
      {
        if(owner_root.at(index) != rank || holder.at(index) == rank)
          continue;
        uint64_t size = 0;
        MPI_Recv(&size, 1, MPI_UINT64_T, holder.at(index), 2 * index,
                 El::mpi::COMM_WORLD.comm, MPI_STATUS_IGNORE);
        std::string data(size, '\0');
        MPI_Recv(data.data(), size, MPI_CHAR, holder.at(index), 2 * index + 1,
                 El::mpi::COMM_WORLD.comm, MPI_STATUS_IGNORE);
        bytes[index] = std::move(data);
      }
    if(!requests.empty())
      MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);

    // Set the local changed blocks, in order (collective in each group)
    for(size_t local = 0; local < block_info.block_indices.size(); ++local)
      {
        const size_t index = block_info.block_indices.at(local);
        if(changed_set.count(index) == 0)
          continue;
        SDP_Block_Data block_data;
        if(is_group_root())
          {
            std::istringstream is(bytes.at(index), std::ios::binary);
            block_data = SDP_Block_Data(is, files.format, local, block_info);
          }
        set_sdp_block_from_root(grid, block_info, block_data, sdp,
                                parameters.solver.sparse_B);
      }
    El::mpi::Barrier(El::mpi::COMM_WORLD);
  }

  void Resident_State::set_objectives_text(const std::string &text)
  {
    std::istringstream is(text);
    read_objectives_from_stream(grid, is, sdp.objective_const,
                                sdp.dual_objective_b);
    files.objectives_text = text;
  }

  // update_pmp: like pmp2sdp --reuseSdp, in memory. "pmp": the new PMP
  // (pmp.json, .nsv, or a list of files). Files listed in the current
  // pmp_info.json keep their blocks; the others are parsed and converted by
  // pmp2sdp's code. The block structure and the normalization must not
  // change. The objective is re-read.
  void Resident_State::cmd_update_pmp(const rapidjson::Value &command,
                                      Json_Writer &w)
  {
    ASSERT(files.is_directory && !files.block_paths.empty(),
           "update_pmp needs an SDP directory with pmp_info.json");
    std::vector<fs::path> inputs;
    for(const auto &s : get_string_list(command, "pmp"))
      inputs.emplace_back(s);
    ASSERT(!inputs.empty(), "update_pmp: no 'pmp' given");

    std::map<fs::path, std::vector<size_t>> blocks_by_path;
    for(size_t index = 0; index < files.block_paths.size(); ++index)
      blocks_by_path[files.block_paths.at(index)].push_back(index);
    const std::function<std::optional<size_t>(const fs::path &)> known
      = [&blocks_by_path](const fs::path &path) -> std::optional<size_t> {
      const auto it = blocks_by_path.find(path);
      if(it == blocks_by_path.end())
        return std::nullopt;
      return it->second.size();
    };

    const auto start = Clock::now();
    Timers timers(env, Verbosity::none);
    std::vector<std::pair<fs::path, size_t>> file_counts;
    auto pmp = read_polynomial_matrix_program(env, inputs, files.max_num_poles,
                                              Verbosity::none, timers, known,
                                              &file_counts);
    ASSERT_EQUAL(pmp.num_matrices, num_blocks(),
                 "update_pmp: the number of blocks must stay the same");
    const double read_seconds = seconds_since(start);

    std::vector<fs::path> new_paths;
    std::vector<size_t> changed;
    for(const auto &[path, count] : file_counts)
      {
        const auto it = blocks_by_path.find(path);
        for(size_t k = 0; k < count; ++k)
          {
            const size_t index = new_paths.size();
            if(it == blocks_by_path.end())
              changed.push_back(index);
            else
              ASSERT_EQUAL(it->second.at(k), index, "update_pmp: block of ",
                           path, " moved; the block order must stay the same");
            new_paths.push_back(path);
          }
      }

    // The normalization must not change: every block depends on it
    std::string normalization_text;
    if(pmp.normalization.has_value())
      {
        std::ostringstream os;
        write_normalization_json(os, pmp.normalization.value());
        normalization_text = os.str();
      }
    ASSERT(normalization_text == files.normalization_text,
           "update_pmp: the normalization differs from the SDP's");

    const auto convert_start = Clock::now();
    Output_SDP output(pmp, {}, timers);
    std::ostringstream objectives_os;
    write_objectives_json(objectives_os, output.objective_const,
                          output.dual_objective_b);
    const bool objectives_changed
      = objectives_os.str() != files.objectives_text;
    if(objectives_changed)
      set_objectives_text(objectives_os.str());
    replace_blocks(output.dual_constraint_groups, changed);
    files.block_paths = new_paths;

    w.Key("changed_blocks");
    w.StartArray();
    for(const auto index : changed)
      w.Uint64(index);
    w.EndArray();
    w.Key("objectives_changed");
    w.Bool(objectives_changed);
    w.Key("read_seconds");
    w.Double(read_seconds);
    w.Key("convert_seconds");
    w.Double(seconds_since(convert_start));
  }

  // update_sdp: re-read blocks and objectives from an SDP directory written
  // by pmp2sdp (e.g. with --reuseSdp). "blocks": "changed" (default: the
  // blocks whose source file in pmp_info.json differs), "all", or a list.
  void Resident_State::cmd_update_sdp(const rapidjson::Value &command,
                                      Json_Writer &w)
  {
    const fs::path dir = get_string(command, "sdp", "");
    ASSERT(!dir.empty(), "update_sdp: no 'sdp' given");
    const auto start = Clock::now();
    Sdp_Files new_files(dir);
    ASSERT(new_files.is_directory, "update_sdp: not a directory: ", dir);
    ASSERT_EQUAL(new_files.num_blocks, num_blocks(),
                 "update_sdp: the number of blocks must stay the same");

    std::vector<size_t> changed;
    bool all = false;
    if(command.HasMember("blocks") && command["blocks"].IsArray())
      {
        for(const auto &v : command["blocks"].GetArray())
          changed.push_back(v.GetUint64());
        std::sort(changed.begin(), changed.end());
      }
    else
      {
        const auto mode = get_string(command, "blocks", "changed");
        all = mode == "all";
        ASSERT(all || mode == "changed", "update_sdp: unknown blocks '",
               mode, "'");
        if(!all)
          ASSERT(!new_files.block_paths.empty()
                   && !files.block_paths.empty(),
                 "update_sdp: blocks 'changed' needs pmp_info.json");
        for(size_t index = 0; index < num_blocks(); ++index)
          if(all
             || new_files.block_paths.at(index)
                  != files.block_paths.at(index))
            changed.push_back(index);
      }
    const bool normalization_changed
      = new_files.normalization_text != files.normalization_text;
    ASSERT(!normalization_changed || changed.size() == num_blocks(),
           "update_sdp: the normalization changed, so all blocks must be "
           "re-read (blocks: \"all\")");

    // Read the changed local blocks on the group roots
    for(size_t local = 0; local < block_info.block_indices.size(); ++local)
      {
        const size_t index = block_info.block_indices.at(local);
        if(!std::binary_search(changed.begin(), changed.end(), index))
          continue;
        SDP_Block_Data block_data;
        if(is_group_root())
          {
            fs::path block_path
              = dir / ("block_data_" + std::to_string(index) + ".bin");
            Block_File_Format format = Block_File_Format::bin;
            if(!fs::exists(block_path))
              {
                block_path.replace_extension(".json");
                format = Block_File_Format::json;
              }
            std::ifstream is(block_path, std::ios::binary);
            ASSERT(is.good(), "Cannot open ", block_path);
            block_data = SDP_Block_Data(is, format, local, block_info);
          }
        set_sdp_block_from_root(grid, block_info, block_data, sdp,
                                parameters.solver.sparse_B);
      }
    // Objectives as SDP's constructor reads them (on all ranks)
    Timers timers(env, Verbosity::none);
    read_objectives(dir, grid, sdp.objective_const, sdp.dual_objective_b,
                    timers);
    if(normalization_changed)
      {
        if(El::mpi::Rank() == 0)
          sdp.normalization = read_normalization(dir, timers);
        normalization.reset();
        std::vector<El::BigFloat> n;
        int has = 0;
        if(El::mpi::Rank() == 0 && sdp.normalization.has_value())
          {
            n = sdp.normalization.value();
            has = 1;
          }
        El::mpi::Broadcast(has, 0, El::mpi::COMM_WORLD);
        if(has)
          {
            size_t size = n.size();
            El::mpi::Broadcast(size, 0, El::mpi::COMM_WORLD);
            n.resize(size);
            El::mpi::Broadcast(n.data(), size, 0, El::mpi::COMM_WORLD);
            normalization = n;
          }
      }
    El::mpi::Barrier(El::mpi::COMM_WORLD);
    new_files.block_paths = new_files.block_paths.empty()
                              ? files.block_paths
                              : new_files.block_paths;
    files = new_files;

    w.Key("changed_blocks");
    w.StartArray();
    for(const auto index : changed)
      w.Uint64(index);
    w.EndArray();
    w.Key("normalization_changed");
    w.Bool(normalization_changed);
    w.Key("read_seconds");
    w.Double(seconds_since(start));
  }

  // load_family: block "block" becomes sum_i w_i (component i). Each
  // component is a PMP file with one matrix of the block's shape (same
  // sample points and bilinear basis), converted with the SDP's
  // normalization. "form_dim": the components are the upper triangle
  // (row-major) of a symmetric form_dim x form_dim matrix, so that
  // set_coefficients can take "lambda" (w_ab = lambda_a lambda_b, doubled
  // off the diagonal). The block keeps its current data until
  // set_coefficients.
  void Resident_State::cmd_load_family(const rapidjson::Value &command,
                                       Json_Writer &w)
  {
    const std::string name = get_string(command, "name", "");
    ASSERT(!name.empty(), "load_family: no 'name' given");
    ASSERT(command.HasMember("block") && command["block"].IsUint64(),
           "load_family: 'block' must be a block index");
    Family family;
    family.block_index = command["block"].GetUint64();
    ASSERT(family.block_index < num_blocks(), "load_family: no block ",
           family.block_index);
    family.owner_root = owner_root.at(family.block_index);
    for(const auto &s : get_string_list(command, "components"))
      family.paths.push_back(s);
    family.num_components = family.paths.size();
    ASSERT(family.num_components > 0, "load_family: no components");
    if(command.HasMember("form_dim"))
      {
        family.form_dim = command["form_dim"].GetUint64();
        ASSERT_EQUAL(family.form_dim * (family.form_dim + 1) / 2,
                     family.num_components,
                     "load_family: form_dim does not match the number of "
                     "components");
      }
    const auto start = Clock::now();
    std::string error;
    if(El::mpi::Rank() == family.owner_root)
      {
        try
          {
            ASSERT(normalization.has_value(),
                   "load_family needs a normalization");
            Timers timers(env, Verbosity::none);
            const size_t N1 = normalization->size();
            for(const auto &path : family.paths)
              {
                auto parsed = PMP_File_Parse_Result::read(
                  path, files.max_num_poles, false, false,
                  [](size_t) { return true; });
                ASSERT_EQUAL(parsed.num_matrices, 1, path);
                auto &matrix = parsed.parsed_matrices.at(0);
                ASSERT_EQUAL(matrix.polynomials(0, 0).size(), N1, path);
                family.first_scaling.push_back(matrix.sample_scalings.at(0));
                std::vector<Polynomial_Vector_Matrix> matrices;
                matrices.emplace_back(std::move(matrix));
                Polynomial_Matrix_Program pmp(
                  std::vector<El::BigFloat>(N1, El::BigFloat(0)),
                  normalization, num_blocks(), std::move(matrices),
                  {family.block_index}, {path});
                Output_SDP output(pmp, {}, timers);
                const auto &group = output.dual_constraint_groups.at(0);
                ASSERT(group.dim
                           == block_info.dimensions.at(family.block_index)
                         && group.num_points
                              == block_info.num_points.at(family.block_index),
                       "load_family: ", path,
                       " does not have the shape of block ",
                       family.block_index);
                // As if read from a block file
                std::ostringstream os(std::ios::binary);
                write_block_data(os, group, files.format);
                std::istringstream is(os.str(), std::ios::binary);
                auto data = parse_block_data(is, files.format);
                if(family.c.empty())
                  {
                    family.bilinear_bases[0] = data.bilinear_bases_even;
                    family.bilinear_bases[1] = data.bilinear_bases_odd;
                  }
                else
                  {
                    for(const size_t parity : {0, 1})
                      {
                        const auto &a = family.bilinear_bases[parity];
                        const auto &b = parity == 0 ? data.bilinear_bases_even
                                                    : data.bilinear_bases_odd;
                        bool equal = a.Height() == b.Height()
                                     && a.Width() == b.Width();
                        for(int i = 0; equal && i < a.Height(); ++i)
                          for(int j = 0; equal && j < a.Width(); ++j)
                            equal = a(i, j) == b(i, j);
                        ASSERT(equal, "load_family: ", path,
                               " has a different bilinear basis");
                      }
                  }
                El::Matrix<El::BigFloat> c(data.c.size(), 1);
                for(size_t i = 0; i < data.c.size(); ++i)
                  c(i, 0) = data.c.at(i);
                family.c.push_back(std::move(c));
                family.B.push_back(std::move(data.B));
              }
          }
        catch(std::exception &e)
          {
            error = e.what();
          }
      }
    broadcast_string(error, family.owner_root);
    ASSERT(error.empty(), error);
    families[name] = std::move(family);
    w.Key("name");
    w.String(name.c_str());
    w.Key("block");
    w.Uint64(families[name].block_index);
    w.Key("components");
    w.Uint64(families[name].num_components);
    w.Key("read_seconds");
    w.Double(seconds_since(start));
  }

  // set_coefficients: set the family's block to sum_i w_i (c_i, B_i).
  // "coefficients": w_i (decimal strings or p/q), or "lambda": the vector of
  // a symmetric family (w_ab = lambda_a lambda_b, times 2 for a < b).
  void Resident_State::cmd_set_coefficients(const rapidjson::Value &command,
                                            Json_Writer &w)
  {
    const std::string name = get_string(command, "family", "");
    const auto it = families.find(name);
    ASSERT(it != families.end(), "set_coefficients: no family '", name, "'");
    const auto &family = it->second;
    std::vector<El::BigFloat> coefficients;
    if(command.HasMember("lambda"))
      {
        ASSERT(family.form_dim > 0,
               "set_coefficients: 'lambda' needs a family with form_dim");
        std::vector<El::BigFloat> lambda;
        for(const auto &s : get_string_list(command, "lambda"))
          lambda.push_back(parse_number(s));
        ASSERT_EQUAL(lambda.size(), family.form_dim);
        for(size_t a = 0; a < family.form_dim; ++a)
          for(size_t b = a; b < family.form_dim; ++b)
            coefficients.push_back(a == b ? lambda.at(a) * lambda.at(b)
                                          : 2 * lambda.at(a) * lambda.at(b));
      }
    else
      {
        for(const auto &s : get_string_list(command, "coefficients"))
          coefficients.push_back(parse_number(s));
      }
    ASSERT_EQUAL(coefficients.size(), family.num_components,
                 "set_coefficients: wrong number of coefficients");

    const auto start = Clock::now();
    const auto local = local_index(family.block_index);
    if(local.has_value())
      {
        SDP_Block_Data block_data;
        if(is_group_root())
          {
            block_data.block_index_local = local.value();
            auto c = family.c.at(0);
            auto B = family.B.at(0);
            El::Scale(coefficients.at(0), c);
            El::Scale(coefficients.at(0), B);
            for(size_t i = 1; i < family.num_components; ++i)
              {
                El::Axpy(coefficients.at(i), family.c.at(i), c);
                El::Axpy(coefficients.at(i), family.B.at(i), B);
              }
            block_data.primal_objective_c = std::move(c);
            block_data.constraint_matrix = std::move(B);
            block_data.bilinear_bases = family.bilinear_bases;
            for(const size_t parity : {0, 1})
              {
                auto &bases_block = block_data.bases_blocks[parity];
                bases_block.Resize(
                  block_info.get_psd_matrix_block_size(family.block_index,
                                                       parity),
                  block_info.get_bilinear_pairing_block_size(
                    family.block_index, parity));
                set_bilinear_bases_block_local(
                  block_data.bilinear_bases[parity], bases_block);
              }
          }
        set_sdp_block_from_root(grid, block_info, block_data, sdp,
                                parameters.solver.sparse_B);
      }
    El::mpi::Barrier(El::mpi::COMM_WORLD);
    // The block no longer comes from a file
    if(!files.block_paths.empty())
      files.block_paths.at(family.block_index)
        = "family:" + name + ":" + std::to_string(num_solves);
    w.Key("family");
    w.String(name.c_str());
    w.Key("set_seconds");
    w.Double(seconds_since(start));
  }

  // negate_normalization: n -> -n. With the same normalization index, this
  // negates c in every block and the objective constant, and leaves B and b
  // unchanged, exactly (sign-magnitude arithmetic). For example the upper vs
  // lower bound of an OPE coefficient at a fixed point.
  void Resident_State::cmd_negate_normalization(Json_Writer &w)
  {
    ASSERT(normalization.has_value(), "negate_normalization: no normalization");
    for(auto &block : sdp.primal_objective_c.blocks)
      {
        auto &local = block.Matrix();
        for(int i = 0; i < local.Height(); ++i)
          for(int j = 0; j < local.Width(); ++j)
            local(i, j) = -local(i, j);
      }
    sdp.objective_const = -sdp.objective_const;
    for(auto &x : normalization.value())
      x = -x;
    if(El::mpi::Rank() == 0)
      sdp.normalization = normalization;
    std::ostringstream os;
    write_normalization_json(os, normalization.value());
    files.normalization_text = os.str();
    El::mpi::Barrier(El::mpi::COMM_WORLD);
    w.Key("objective_const");
    write_bigfloat(w, sdp.objective_const);
  }

  // save_iterate: "name", in memory; or with "dir", as a checkpoint there
  void Resident_State::cmd_save_iterate(const rapidjson::Value &command,
                                        Json_Writer &w)
  {
    const std::string name = get_string(command, "name", "");
    const std::string dir = get_string(command, "dir", "");
    if(!dir.empty())
      {
        // A separate checkpoint: keep the counters of the main one
        const auto current = solver.current_generation;
        const auto backup = solver.backup_generation;
        solver.current_generation = 0;
        solver.backup_generation.reset();
        checkpoint(dir, to_property_tree(parameters));
        solver.current_generation = current;
        solver.backup_generation = backup;
        w.Key("dir");
        w.String(dir.c_str());
      }
    else
      {
        ASSERT(!name.empty(), "save_iterate: give 'name' or 'dir'");
        iterates.erase(name);
        iterates.emplace(name,
                         Iterate{solver.x, solver.y, solver.X, solver.Y});
        w.Key("name");
        w.String(name.c_str());
      }
  }

  // restore_iterate: "name" (in memory), or "dir" (a checkpoint)
  void Resident_State::cmd_restore_iterate(const rapidjson::Value &command,
                                           Json_Writer &w)
  {
    const std::string name = get_string(command, "name", "");
    const std::string dir = get_string(command, "dir", "");
    if(!dir.empty())
      {
        const auto current = solver.current_generation;
        const auto backup = solver.backup_generation;
        solver.load_checkpoint(dir, block_info, parameters.verbosity, true);
        solver.current_generation = current;
        solver.backup_generation = backup;
      }
    else
      {
        const auto it = iterates.find(name);
        ASSERT(it != iterates.end(), "restore_iterate: no iterate '", name,
               "'");
        solver.x = it->second.x;
        solver.X = it->second.X;
        solver.y = it->second.y;
        solver.Y = it->second.Y;
      }
    w.Key("restored");
    w.String(dir.empty() ? name.c_str() : dir.c_str());
  }

  void Resident_State::cmd_status(Json_Writer &w)
  {
    w.Key("solves");
    w.Uint64(num_solves);
    w.Key("sdpDir");
    w.String(files.dir.string().c_str());
    w.Key("iterates");
    w.StartArray();
    for(const auto &[name, iterate] : iterates)
      w.String(name.c_str());
    w.EndArray();
    w.Key("families");
    w.StartArray();
    for(const auto &[name, family] : families)
      w.String(name.c_str());
    w.EndArray();
  }
}

void run_resident(const Environment &env, const SDPB_Parameters &parameters,
                  const std::vector<std::string> &startup_args,
                  const mp_bitcnt_t startup_precision,
                  const Block_Info &block_info)
{
  const auto start = Clock::now();
  Control_Channel channel(parameters.control_dir,
                          parameters.control_idle_timeout);
  Resident_State state(env, parameters, startup_args, startup_precision,
                       block_info);
  if(El::mpi::Rank() == 0)
    {
      channel.write_file("ready.json", state.ready_json(seconds_since(start)));
      if(parameters.verbosity >= Verbosity::regular)
        El::Output(boost::posix_time::second_clock::local_time(),
                   " Resident mode: waiting for commands in ",
                   parameters.control_dir);
    }

  bool keep_running = true;
  while(keep_running)
    {
      const auto text = channel.next_command(env);
      if(!text.has_value())
        break;
      rapidjson::Document document;
      document.Parse(text->c_str());
      rapidjson::StringBuffer sb;
      Json_Writer w(sb);
      w.StartObject();
      w.Key("seq");
      w.Uint64(channel.seq());
      if(document.HasParseError())
        {
          w.Key("error");
          w.String("cannot parse the command as JSON");
        }
      else
        {
          // One command, or {"commands": [...]} executed in order until the
          // first error
          std::vector<const rapidjson::Value *> commands;
          const bool batch
            = document.IsObject() && document.HasMember("commands");
          if(batch)
            for(const auto &c : document["commands"].GetArray())
              commands.push_back(&c);
          else
            commands.push_back(&document);
          w.Key("results");
          w.StartArray();
          for(const auto *command : commands)
            {
              rapidjson::StringBuffer csb;
              Json_Writer cw(csb);
              std::string error;
              try
                {
                  keep_running = state.execute(*command, cw);
                }
              catch(std::exception &e)
                {
                  error = e.what();
                }
              if(error.empty())
                {
                  w.RawValue(csb.GetString(), csb.GetSize(),
                             rapidjson::kObjectType);
                }
              else
                {
                  w.StartObject();
                  w.Key("error");
                  w.String(error.c_str());
                  w.EndObject();
                  if(El::mpi::Rank() == 0)
                    El::Output("Resident mode: command error: ", error);
                  break;
                }
              if(!keep_running)
                break;
            }
          w.EndArray();
        }
      w.EndObject();
      if(El::mpi::Rank() == 0)
        channel.write_result(sb.GetString());
    }

  const auto reason = channel.stop_reason();
  if(reason == Control_Channel::Stop_Reason::sigterm)
    state.checkpoint(parameters.solver.checkpoint_out,
                     to_property_tree(parameters));
  if(El::mpi::Rank() == 0)
    {
      rapidjson::StringBuffer sb;
      Json_Writer w(sb);
      w.StartObject();
      w.Key("status");
      w.String("exit");
      w.Key("reason");
      const auto reason_string
        = reason == Control_Channel::Stop_Reason::none ? std::string("quit")
                                                       : to_string(reason);
      w.String(reason_string.c_str());
      w.Key("last_seq");
      w.Uint64(channel.seq());
      w.Key("seconds");
      w.Double(seconds_since(start));
      w.EndObject();
      channel.write_file("exit.json", sb.GetString());
    }
}
