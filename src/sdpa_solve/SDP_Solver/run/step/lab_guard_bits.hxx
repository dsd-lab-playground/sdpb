#pragma once

// lab (bfss #77): opt-in LAB_GUARD_BITS=<g> (g > 0) for initialize_P's trmm.
//
// sdpa-solver's fixed point divides each row of L (or column of G) by its
// 2-norm and multiplies by 2^precision, so an entry 2^-k below its row's
// norm keeps only precision - k bits (#175: 25-33 bits lost). Here, as in
// bootstrap-speed #145, each row/column is instead scaled by a power of 2,
// 2^(precision + g - E) with 2^(E-1) <= max |entry| < 2^E, which is exact
// (no division) and keeps g more bits; the trmm's primes are chosen for
// precision + g bits (Initialize_P_Config::comb). The output is scaled back
// by the inverse powers of 2. Not bitwise equal to the 2-norm version.

#include "sdp_solve/Block_Matrix/Block_Diagonal_Matrix.hxx"
#include "sdpb_util/assert.hxx"

#include <El.hpp>

#include <cstdlib>
#include <string>
#include <vector>

namespace Sdpb::Sdpa::lab
{
  inline int guard_bits()
  {
    static const int value = [] {
      const char *env = std::getenv("LAB_GUARD_BITS");
      return env == nullptr ? 0 : std::max(0, std::atoi(env));
    }();
    return value;
  }

  // 2^(E-1) <= |x| < 2^E; INT_MIN for x = 0
  inline long exponent_of(const El::BigFloat &x)
  {
    const __mpf_struct *p = x.gmp_float.get_mpf_t();
    const long size = std::abs(p->_mp_size);
    if(size == 0)
      return std::numeric_limits<int>::min();
    const mp_limb_t top = p->_mp_d[size - 1];
    return long(GMP_NUMB_BITS) * p->_mp_exp - __builtin_clzl(top);
  }

  // Per row (rows = true) or column of a DistMatrix: the shift s with
  // max |x 2^s| < 2^bits (0 for a zero row/column). Only the lower triangle
  // if lower_only. Collective over the matrix's DistComm.
  inline std::vector<int>
  pow2_shifts(const El::DistMatrix<El::BigFloat> &A, const bool rows,
              const bool lower_only, const int bits)
  {
    const int n = rows ? A.Height() : A.Width();
    std::vector<int> e(n, std::numeric_limits<int>::min());
    for(int jLoc = 0; jLoc < A.LocalWidth(); ++jLoc)
      {
        const int j = A.GlobalCol(jLoc);
        for(int iLoc = 0; iLoc < A.LocalHeight(); ++iLoc)
          {
            const int i = A.GlobalRow(iLoc);
            if(lower_only && i < j)
              continue;
            const long E = exponent_of(A.GetLocalCRef(iLoc, jLoc));
            int &slot = e[rows ? i : j];
            if(E > slot)
              slot = int(E);
          }
      }
    El::mpi::AllReduce(e.data(), n, El::mpi::MAX, A.DistComm());
    std::vector<int> shifts(n, 0);
    for(int k = 0; k < n; ++k)
      if(e[k] != std::numeric_limits<int>::min())
        shifts[k] = bits - e[k];
    return shifts;
  }

  inline void shift_by(El::BigFloat &x, const int s)
  {
    if(s >= 0)
      x <<= s;
    else
      x >>= -s;
  }

  // x_ij := x_ij 2^(row_shift_i + col_shift_j) (empty vector = no shift)
  inline void apply_shifts(El::DistMatrix<El::BigFloat> &A,
                           const std::vector<int> &row_shifts,
                           const std::vector<int> &col_shifts,
                           const bool lower_only, const int sign)
  {
    for(int jLoc = 0; jLoc < A.LocalWidth(); ++jLoc)
      {
        const int j = A.GlobalCol(jLoc);
        for(int iLoc = 0; iLoc < A.LocalHeight(); ++iLoc)
          {
            const int i = A.GlobalRow(iLoc);
            if(lower_only && i < j)
              continue;
            int s = 0;
            if(!row_shifts.empty())
              s += row_shifts[i];
            if(!col_shifts.empty())
              s += col_shifts[j];
            auto &x = A.Matrix()(iLoc, jLoc);
            if(mpf_sgn(x.gmp_float.get_mpf_t()) != 0)
              shift_by(x, sign * s);
          }
      }
  }
}
