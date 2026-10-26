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

#include <climits>
#include <cstdlib>
#include <limits>
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

  // LAB_SYRK_GUARD_BITS=<g>: the same for syrk's fixed point of P
  // (normalize_P), used by syrk_Q and Bigint_Syrk_Config::comb
  inline int syrk_guard_bits()
  {
    static const int value = [] {
      const char *env = std::getenv("LAB_SYRK_GUARD_BITS");
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

namespace Sdpb::Sdpa::lab
{
  // LAB_L_INV_EXTRA_BITS=<e> (experimental, bfss #77): for blocks held by a
  // single rank, L^-1 is computed by forward substitution in mpf at
  // precision + e bits from the working-precision L, then rounded to the
  // working precision (instead of El::TriangularInverse at the working
  // precision). A diagnostic of where S loses accuracy; blocks spread over
  // several ranks keep El::TriangularInverse.
  inline int l_inv_extra_bits()
  {
    static const int value = [] {
      const char *env = std::getenv("LAB_L_INV_EXTRA_BITS");
      return env == nullptr ? 0 : std::max(0, std::atoi(env));
    }();
    return value;
  }

  // Returns false (block untouched) if the block is not held by one rank.
  inline bool lower_triangular_inverse_extra(El::DistMatrix<El::BigFloat> &L,
                                             const int extra_bits)
  {
    const int n = L.Height();
    if(L.LocalHeight() != n || L.LocalWidth() != L.Width())
      return false;
    const mp_bitcnt_t prec = El::gmp::Precision() + extra_bits;
    auto &A = L.Matrix();
    std::vector<mpf_t> Lh(size_t(n) * n), X(size_t(n) * n);
    for(int j = 0; j < n; ++j)
      for(int i = 0; i < n; ++i)
        {
          mpf_init2(Lh[i + size_t(n) * j], prec);
          mpf_init2(X[i + size_t(n) * j], prec);
          if(i >= j)
            mpf_set(Lh[i + size_t(n) * j], A(i, j).gmp_float.get_mpf_t());
        }
    mpf_t acc, t;
    mpf_init2(acc, prec);
    mpf_init2(t, prec);
    for(int j = 0; j < n; ++j)
      {
        mpf_ui_div(X[j + size_t(n) * j], 1, Lh[j + size_t(n) * j]);
        for(int i = j + 1; i < n; ++i)
          {
            mpf_set_ui(acc, 0);
            for(int k = j; k < i; ++k)
              {
                mpf_mul(t, Lh[i + size_t(n) * k], X[k + size_t(n) * j]);
                mpf_add(acc, acc, t);
              }
            mpf_div(acc, acc, Lh[i + size_t(n) * i]);
            mpf_neg(X[i + size_t(n) * j], acc);
          }
      }
    for(int j = 0; j < n; ++j)
      for(int i = j; i < n; ++i)
        mpf_set(A(i, j).gmp_float.get_mpf_t(), X[i + size_t(n) * j]);
    mpf_clear(acc);
    mpf_clear(t);
    for(auto &v : Lh)
      mpf_clear(v);
    for(auto &v : X)
      mpf_clear(v);
    return true;
  }
}
