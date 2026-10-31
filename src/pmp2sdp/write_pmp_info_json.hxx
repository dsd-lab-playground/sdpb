#pragma once

#include "write_vector.hxx"
#include "pmp/PMP_Info.hxx"
#include "Reuse_SDP.hxx"
#include "sdpb_util/Boost_Float.hxx"
#include "sdpb_util/assert.hxx"
#include "sdpb_util/json/Json_Writer.hxx"

#include <El.hpp>

#include <ostream>

inline void write_pvm_info_json(Json_Writer &writer, const PVM_Info &block,
                                const size_t num_blocks)
{
  block.validate(num_blocks);
  writer.StartObject();
  writer.Key("index");
  {
    writer.Int(block.block_index);
  }
  writer.Key("path");
  {
    writer.String(block.block_path.string().c_str());
  }
  writer.Key("dim");
  {
    writer.Uint64(block.dim);
  }
  writer.Key("prefactor");
  {
    writer.StartObject();
    writer.Key("constant");
    writer.BigFloat(block.prefactor.constant);
    writer.Key("base");
    writer.BigFloat(block.prefactor.base);
    writer.Key("poles");
    writer.BigFloatArray(block.prefactor.poles);
    writer.EndObject();
  }
  writer.Key("reducedPrefactor");
  {
    writer.StartObject();
    writer.Key("constant");
    writer.BigFloat(block.reduced_prefactor.constant);
    writer.Key("base");
    writer.BigFloat(block.reduced_prefactor.base);
    writer.Key("poles");
    writer.BigFloatArray(block.reduced_prefactor.poles);
    writer.EndObject();
  }
  writer.Key("samplePoints");
  writer.BigFloatArray(block.sample_points);
  writer.Key("sampleScalings");
  writer.BigFloatArray(block.sample_scalings);
  writer.Key("reducedSampleScalings");
  writer.BigFloatArray(block.reduced_sample_scalings);
  writer.EndObject();
}

inline void write_pmp_info_json(std::ostream &output_stream,
                                const std::vector<PVM_Info> &pmp_info)
{
  ASSERT_EQUAL(El::mpi::Rank(), 0);
  rapidjson::OStreamWrapper ows(output_stream);
  Json_Writer writer(ows);

  writer.StartArray();
  for(const auto &block : pmp_info)
    write_pvm_info_json(writer, block, pmp_info.size());

  writer.EndArray();
  ASSERT(writer.IsComplete());
}

// Same output as write_pmp_info_json(), for an SDP with blocks reused from
// an old SDP: reused entries are copied from the old pmp_info.json.
inline void write_pmp_info_json(std::ostream &output_stream,
                                const std::vector<PVM_Info> &pmp_info,
                                const Reused_SDP &reuse)
{
  ASSERT_EQUAL(El::mpi::Rank(), 0);
  ASSERT_EQUAL(reuse.new_to_old.size(), pmp_info.size());
  rapidjson::OStreamWrapper ows(output_stream);
  output_stream << '[';
  for(size_t index = 0; index < pmp_info.size(); ++index)
    {
      if(index != 0)
        output_stream << ',';
      const auto &old_index = reuse.new_to_old.at(index);
      if(old_index.has_value())
        {
          output_stream << reuse.renumbered_pmp_info_entry(old_index.value(),
                                                           index);
        }
      else
        {
          ows.Flush();
          Json_Writer writer(ows);
          write_pvm_info_json(writer, pmp_info.at(index), pmp_info.size());
          ASSERT(writer.IsComplete());
          ows.Flush();
        }
    }
  output_stream << ']';
}

// Send a Boost_Float exactly: its MPFR precision and an exact hex string.
// (Converting it to El::BigFloat and back rounds it, so pmp_info.json
// depended on which rank owned each block.)
inline void send_boost_float_exact(const Boost_Float &value, const int to,
                                   const El::mpi::Comm &comm)
{
  const auto &x = value.backend().data();
  char *str = nullptr;
  ASSERT(mpfr_asprintf(&str, "%Ra", x) >= 0);
  const std::string hex(str);
  mpfr_free_str(str);
  El::mpi::Send<size_t>(mpfr_get_prec(x), to, comm);
  El::mpi::Send<size_t>(hex.size(), to, comm);
  MPI_Send(hex.data(), hex.size(), MPI_CHAR, to, 0, comm.comm);
}

inline Boost_Float
recv_boost_float_exact(const int from, const El::mpi::Comm &comm)
{
  const auto prec = El::mpi::Recv<size_t>(from, comm);
  std::string hex(El::mpi::Recv<size_t>(from, comm), '\0');
  MPI_Recv(hex.data(), hex.size(), MPI_CHAR, from, 0, comm.comm,
           MPI_STATUS_IGNORE);
  Boost_Float result;
  auto &x = result.backend().data();
  mpfr_set_prec(x, prec);
  ASSERT(mpfr_set_str(x, hex.c_str(), 0, MPFR_RNDN) == 0, "Cannot parse ",
         hex);
  return result;
}

// Send from rank='from' to rank=0
inline void synchronize_pvm_info(PVM_Info &pvm_info, const int from)
{
  const int to = 0;

  const auto comm = El::mpi::COMM_WORLD;
  const int rank = comm.Rank();

  if(rank != to && rank != from)
    return;
  if(to == from)
    return;

  // block_index
  {
    if(rank == from)
      El::mpi::Send(pvm_info.block_index, to, comm);
    if(rank == to)
      pvm_info.block_index = El::mpi::Recv<int>(from, comm);
  }

  // block_path
  {
    auto block_path = pvm_info.block_path.string();
    if(rank == from)
      {
        const size_t path_size = block_path.size();
        El::mpi::Send<size_t>(path_size, to, comm);
        MPI_Send(block_path.data(), path_size, MPI_CHAR, to, 0, comm.comm);
      }
    if(rank == to)
      {
        const auto path_size = El::mpi::Recv<size_t>(from, comm);
        block_path.resize(path_size);
        MPI_Recv(block_path.data(), path_size, MPI_CHAR, from, 0, comm.comm,
                 MPI_STATUS_IGNORE);
        pvm_info.block_path = block_path;
      }
  }

  // dim
  {
    if(rank == from)
      El::mpi::Send(pvm_info.dim, to, comm);
    if(rank == to)
      pvm_info.dim = El::mpi::Recv<size_t>(from, comm);
  }

  // prefactor, reduced_prefactor
  for(auto *damped_rational_ptr :
      {&pvm_info.prefactor, &pvm_info.reduced_prefactor})
    {
      auto &damped_rational = *damped_rational_ptr;
      if(rank == from)
        {
          send_boost_float_exact(damped_rational.constant, to, comm);
          send_boost_float_exact(damped_rational.base, to, comm);
          El::mpi::Send<size_t>(damped_rational.poles.size(), to, comm);
          for(const auto &pole : damped_rational.poles)
            send_boost_float_exact(pole, to, comm);
        }
      if(rank == to)
        {
          damped_rational.constant = recv_boost_float_exact(from, comm);
          damped_rational.base = recv_boost_float_exact(from, comm);

          const size_t num_poles = El::mpi::Recv<size_t>(from, comm);
          damped_rational.poles.resize(num_poles);
          for(auto &pole : damped_rational.poles)
            pole = recv_boost_float_exact(from, comm);
        }
    }
  for(auto vec_ptr : {&pvm_info.sample_points, &pvm_info.sample_scalings,
                      &pvm_info.reduced_sample_scalings})
    {
      auto &vec = *vec_ptr;
      if(rank == from)
        {
          El::mpi::Send<size_t>(vec.size(), to, comm);
          El::mpi::Send(vec.data(), vec.size(), to, comm);
        }
      if(rank == to)
        {
          const size_t num_points = El::mpi::Recv<size_t>(from, comm);
          vec.resize(num_points);
          El::mpi::Recv(vec.data(), vec.size(), from, comm);
        }
    }
}

// Collect all block on rank=0
inline std::vector<PVM_Info>
synchronize_pmp_info_blocks(const PMP_Info &pmp_info)
{
  std::vector<PVM_Info> all_pvm_infos(pmp_info.num_blocks);

  // Fill all_pvm_infos with all local blocks
  for(size_t local_index = 0; local_index < pmp_info.blocks.size();
      ++local_index)
    {
      const int block_index = pmp_info.blocks.at(local_index).block_index;
      ASSERT(block_index >= 0 && block_index < pmp_info.num_blocks,
             DEBUG_STRING(block_index), DEBUG_STRING(pmp_info.num_blocks));
      all_pvm_infos.at(block_index) = pmp_info.blocks.at(local_index);
    }

  // Synchronize owning ranks for each block
  const int rank = El::mpi::Rank();
  std::vector<int> block_ranks(pmp_info.num_blocks, -1);
  for(auto &block : pmp_info.blocks)
    {
      block_ranks.at(block.block_index) = rank;
    }
  El::mpi::AllReduce(block_ranks.data(), block_ranks.size(), El::mpi::MAX,
                     El::mpi::COMM_WORLD);

  // Send all blocks to rank=0
  for(size_t block_index = 0; block_index < pmp_info.num_blocks; ++block_index)
    {
      const int from = block_ranks.at(block_index);
      // Blocks reused by pmp2sdp --reuseSdp are on no rank
      if(from == -1)
        continue;
      synchronize_pvm_info(all_pvm_infos.at(block_index), from);
    }
  return all_pvm_infos;
}