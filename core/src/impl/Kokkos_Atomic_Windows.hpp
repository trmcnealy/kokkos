/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#pragma once

#ifndef KOKKOS_ATOMIC_WINDOWS_HPP
#define KOKKOS_ATOMIC_WINDOWS_HPP

#if (defined(_MSC_VER) || defined(WIN32) || defined(_WIN32) || \
     defined(WIN64) || defined(_WIN64) || defined(_WINDOWS))

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <type_traits>

#include <intrin.h>

#ifdef __MINGW64__
#define _InterlockedCompareExchange128 _InterlockedCompare64Exchange128
#endif

#if !defined(KOKKOS_ATOMIC_ASSEMBLY_HPP)
#include "Kokkos_Atomic_Assembly.hpp"
#endif

namespace Kokkos {

namespace Windows {

#if defined(WIN64) || defined(_WIN64) || defined(_AMD64) || \
    defined(_AMD64_) || defined(__x86_64) || defined(__x86_64__)
#define MemoryBarrier __faststorefence
#elif defined(_IA64_)
#define MemoryBarrier __mf
#else
__forceinline void MemoryBarrier() {
  long Barrier;
  __asm {
        xchg Barrier, eax
  }
}
#endif

#pragma region SIGNED_INTERLOCKED

#define SIGNED_INTERLOCKED(TYPE, INTERLOCKED_SUF)                              \
  inline TYPE Exchange(TYPE volatile* const dest, const TYPE value) {          \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue !=                                                       \
             ::_InterlockedExchange##INTERLOCKED_SUF(dest, value));            \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T Exchange(T volatile* const dest,                                    \
                    typename std::enable_if<sizeof(T) == sizeof(TYPE) &&       \
                                                !std::is_same<T, TYPE>::value, \
                                            const T&>::type value) {           \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedExchange##INTERLOCKED_SUF(              \
                             (TYPE volatile*)dest, *(TYPE*)&value));           \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE CompareExchange(TYPE volatile* const dest, const TYPE& compare,  \
                              const TYPE& value) {                             \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(       \
                             dest, value, compare));                           \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T CompareExchange(                                                    \
      T volatile* const dest, const T& compare,                                \
      typename std::enable_if<sizeof(T) == sizeof(TYPE) &&                     \
                                  !std::is_same<T, TYPE>::value,               \
                              const T&>::type value) {                         \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue !=                                                       \
             ::_InterlockedCompareExchange##INTERLOCKED_SUF(                   \
                 (TYPE volatile*)dest, *(TYPE*)&value, *(TYPE*)&compare));     \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE ExchangeAdd(TYPE volatile* const dest, const TYPE& value) {      \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue !=                                                       \
             ::_InterlockedExchangeAdd##INTERLOCKED_SUF(dest, value));         \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T ExchangeAdd(                                                        \
      T volatile* const dest,                                                  \
      typename std::enable_if<sizeof(T) == sizeof(TYPE) &&                     \
                                  !std::is_same<T, TYPE>::value,               \
                              const T&>::type value) {                         \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedExchangeAdd##INTERLOCKED_SUF(           \
                             (TYPE volatile*)dest, *(TYPE*)&value));           \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE And(TYPE volatile* const dest, const TYPE& value) {              \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedAnd##INTERLOCKED_SUF(dest, value));     \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T And(T volatile* const dest,                                         \
               typename std::enable_if<sizeof(T) == sizeof(TYPE) &&            \
                                           !std::is_same<T, TYPE>::value,      \
                                       const T&>::type value) {                \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedAnd##INTERLOCKED_SUF(                   \
                             (TYPE volatile*)dest, *(TYPE*)&value));           \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE Or(TYPE volatile* const dest, const TYPE& value) {               \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedOr##INTERLOCKED_SUF(dest, value));      \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T Or(T volatile* const dest,                                          \
              typename std::enable_if<sizeof(T) == sizeof(TYPE) &&             \
                                          !std::is_same<T, TYPE>::value,       \
                                      const T&>::type value) {                 \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedOr##INTERLOCKED_SUF(                    \
                             (TYPE volatile*)dest, *(TYPE*)&value));           \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE Xor(TYPE volatile* const dest, const TYPE& value) {              \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedXor##INTERLOCKED_SUF(dest, value));     \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T Xor(T volatile* const dest, const TYPE& value) {                    \
    const TYPE oldValue = *(TYPE*)dest;                                        \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedXor##INTERLOCKED_SUF(                   \
                             (TYPE volatile*)dest, *(TYPE*)&value));           \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE Add(TYPE volatile* const dest, const TYPE& value) {              \
    const TYPE oldValue = *(TYPE*)dest;                                        \
    const TYPE calcVal  = (oldValue + value);                                  \
    const TYPE newVal   = calcVal;                                             \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(       \
                             dest, newVal, oldValue));                         \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  inline T Add(T volatile* const dest,                                         \
               typename std::enable_if<sizeof(T) == sizeof(TYPE) &&            \
                                           !std::is_same<T, TYPE>::value,      \
                                       const T&>::type value) {                \
    const TYPE oldValue = *(TYPE*)dest;                                        \
    const T calcVal     = (oldValue + value);                                  \
    const TYPE newVal   = *(TYPE*)&calcVal;                                    \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(       \
                             (TYPE volatile*)dest, newVal, oldValue));         \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }                                                                            \
                                                                               \
  inline TYPE Sub(TYPE volatile* const dest, const TYPE& value) {              \
    const TYPE oldValue = *(TYPE*)dest;                                        \
    const TYPE calcVal  = (oldValue - value);                                  \
    const TYPE newVal   = calcVal;                                             \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(       \
                             dest, newVal, oldValue));                         \
                                                                               \
    return oldValue;                                                           \
  }                                                                            \
  template <typename T>                                                        \
  inline T Sub(T volatile* const dest,                                         \
               typename std::enable_if<sizeof(T) == sizeof(TYPE) &&            \
                                           !std::is_same<T, TYPE>::value,      \
                                       const T&>::type value) {                \
    const TYPE oldValue = *(TYPE*)dest;                                        \
    const T calcVal     = (oldValue - value);                                  \
    const TYPE newVal   = *(TYPE*)&calcVal;                                    \
                                                                               \
    do {                                                                       \
      ;                                                                        \
    } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(       \
                             (TYPE volatile*)dest, newVal, oldValue));         \
                                                                               \
    return *(T*)&oldValue;                                                     \
  }
#pragma endregion

SIGNED_INTERLOCKED(char, 8)

SIGNED_INTERLOCKED(short, 16)

SIGNED_INTERLOCKED(long, )

SIGNED_INTERLOCKED(__int64, 64)

inline static float Add(float volatile* const dest, const float& value) {
  const volatile float oldValue = *(float*)dest;
  const float calcVal           = (oldValue + value);
  const float newVal            = calcVal;

  union V {
    long i;
    float t;
    inline V(){};
  } newval, oldval;

  newval.t = value;
  oldval.t = oldValue;

  union Vol {
    long volatile* i;
    float volatile* t;
    inline Vol(){};
  } newdest;

  newdest.t = dest;

  do {
    ;
  } while (oldval.i != ::_InterlockedExchangeAdd(newdest.i, newval.i));

  return oldValue;
}

inline static double Add(double volatile* const dest, const double& value) {
  const volatile double oldValue = *(double*)dest;
  const double calcVal           = (oldValue + value);
  const double newVal            = calcVal;

  union V {
    __int64 i;
    double t;
    inline V(){};
  } newval, oldval;

  newval.t = value;
  oldval.t = oldValue;

  union Vol {
    __int64 volatile* i;
    double volatile* t;
    inline Vol(){};
  } newdest;

  newdest.t = dest;

  do {
    ;
  } while (oldval.i != ::_InterlockedExchangeAdd64(newdest.i, newval.i));

  return oldValue;
}

#pragma region Interlocked 128

template <typename T>
inline T CompareExchange(
    T volatile* const dest, const T& compare,
    typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t), const T&>::type
        value) {
  T oldValue = *reinterpret_cast<T*>(dest);
  // void* localValue = (void*)&value;

#if defined(__MINGW64__)
  __int64 localCompare = (__int64)&compare;
#else
  __int64* localCompare = (__int64*)&compare;
#endif

  union U {
    Impl::cas128_t i;
    T t;
    inline U(){};
  } newval;

  newval.t = value;

  do {
    ;
  } while (_InterlockedCompareExchange128(
               (__int64 volatile*)dest, *(__int64*)&newval.i.upper,
               *(__int64*)&newval.i.lower, localCompare) == 0);

  return oldValue;
}

template <typename T>
inline T Add(T volatile* const dest,
             typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t),
                                     const T&>::type value) {
  const T oldValue = *reinterpret_cast<T*>(dest);
  const T newVal   = (*dest + value);

  do {
    ;
  } while (oldValue != InterlockedCompareExchange(dest, newVal, oldValue));

  return *(T*)&oldValue;
}

template <typename T>
inline T Sub(T volatile* const dest,
             typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t),
                                     const T&>::type value) {
  const T oldValue = *reinterpret_cast<T*>(dest);
  const T newVal   = (*dest - value);

  do {
    ;
  } while (oldValue != InterlockedCompareExchange(dest, newVal, oldValue));

  return *(T*)&oldValue;
}

// TODO
template <typename T>
inline T Exchange(void* volatile* dest,
                  typename std::enable_if<(sizeof(T) != 4) && (sizeof(T) != 8),
                                          void*>::type& val) {
  while (!Impl::lock_address_host_space(*dest))
    ;
  MemoryBarrier();

  const __int64 oldValue = *reinterpret_cast<__int64*>(*dest);
  *dest                  = val;

  MemoryBarrier();
  Impl::unlock_address_host_space(*dest);

  return oldValue;
}

#pragma endregion

}  // namespace Windows

}  // namespace Kokkos
#endif
#endif
