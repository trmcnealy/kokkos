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
#    define KOKKOS_ATOMIC_WINDOWS_HPP

#    if (defined(_MSC_VER) || defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(_WINDOWS))

//#        ifndef __min
//#           if __has_builtin(__builtin_min)
//#               define __min __builtin_min
//#           else
//#               define __min(a,b) (((a) < (b)) ? (a) : (b))
//#           endif
//#        endif

#        ifndef NOMINMAX
#            define NOMINMAX
#        endif

#        include <type_traits>

#        include <intrin.h>

#        ifdef __MINGW64__
#            define _InterlockedCompareExchange128 _InterlockedCompare64Exchange128
#        endif

#        if !defined(KOKKOS_ATOMIC_ASSEMBLY_HPP)
#            include "Kokkos_Atomic_Assembly.hpp"
#        endif

namespace Kokkos
{

#        pragma region SIGNED_INTERLOCKED

#        define SIGNED_INTERLOCKED(TYPE, INTERLOCKED_SUF)                                                                                                 \
            inline TYPE InterlockedExchange(TYPE volatile* const dest, const TYPE value)                                                     \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedExchange##INTERLOCKED_SUF(dest, value));                                                               \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedExchange(T volatile* const                                                                       dest,            \
                                         typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)           \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedExchange##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value));                                      \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedCompareExchange(TYPE volatile* const dest, const TYPE& compare, const TYPE& value)                   \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(dest, value, compare));                                               \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedCompareExchange(T volatile* const dest, const T& compare,                                                    \
                                                typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)    \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value, *(TYPE*)&compare));             \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedExchangeAdd(TYPE volatile* const dest, const TYPE& value)                                                 \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedExchangeAdd##INTERLOCKED_SUF(dest, value));                                                            \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedExchangeAdd(T volatile* const                                                                       dest,         \
                                            typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)        \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedExchangeAdd##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value));                                   \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedAnd(TYPE volatile* const dest, const TYPE& value)                                                         \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedAnd##INTERLOCKED_SUF(dest, value));                                                                    \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedAnd(T volatile* const                                                                       dest,                 \
                                    typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)                \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedAnd##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value));                                           \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedOr(TYPE volatile* const dest, const TYPE& value)                                                          \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedOr##INTERLOCKED_SUF(dest, value));                                                                     \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedOr(T volatile* const                                                                       dest,                  \
                                   typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)                 \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedOr##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value));                                            \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedXor(TYPE volatile* const dest, const TYPE& value)                                                         \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedXor##INTERLOCKED_SUF(dest, value));                                                                    \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedXor(T volatile* const dest, const TYPE& value)                                                               \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedXor##INTERLOCKED_SUF((TYPE volatile*)dest, *(TYPE*)&value));                                           \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedAdd(TYPE volatile* const dest, const TYPE& value)                                                         \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                const TYPE calcVal  = (oldValue + value);                                                                                                    \
                const TYPE newVal   = calcVal;                                                                                                            \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(dest, newVal, oldValue));                                             \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
                                                                                                                                                          \
            template<typename T>                                                                                                                          \
            inline T InterlockedAdd(T volatile* const                                                                       dest,                 \
                                    typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)                \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                const T    calcVal  = (oldValue + value);                                                                                                    \
                const TYPE newVal   = *(TYPE*)&calcVal;                                                                                                   \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF((TYPE volatile*)dest, newVal, oldValue));                             \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }                                                                                                                                             \
                                                                                                                                                          \
            inline TYPE InterlockedSub(TYPE volatile* const dest, const TYPE& value)                                                         \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                const TYPE calcVal  = (oldValue - value);                                                                                                    \
                const TYPE newVal   = calcVal;                                                                                                            \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF(dest, newVal, oldValue));                                             \
                                                                                                                                                          \
                return oldValue;                                                                                                                          \
            }                                                                                                                                             \
            template<typename T>                                                                                                                          \
            inline T InterlockedSub(T volatile* const                                                                       dest,                 \
                                    typename std::enable_if<sizeof(T) == sizeof(TYPE) && !std::is_same<T, TYPE>::value, const T&>::type value)                \
            {                                                                                                                                             \
                const TYPE oldValue = *(TYPE*)dest;                                                                                                       \
                const T    calcVal  = (oldValue - value);                                                                                                    \
                const TYPE newVal   = *(TYPE*)&calcVal;                                                                                                   \
                                                                                                                                                          \
                do                                                                                                                                        \
                {                                                                                                                                         \
                    ;                                                                                                                                     \
                } while (oldValue != ::_InterlockedCompareExchange##INTERLOCKED_SUF((TYPE volatile*)dest, newVal, oldValue));                             \
                                                                                                                                                          \
                return *(T*)&oldValue;                                                                                                                    \
            }
#        pragma endregion

        SIGNED_INTERLOCKED(char, 8)

        SIGNED_INTERLOCKED(short, 16)

        SIGNED_INTERLOCKED(long, )

        SIGNED_INTERLOCKED(__int64, 64)

#       pragma region Interlocked 128

    template<typename T>
    inline T InterlockedCompareExchange(T volatile* const dest,
        const T& compare,
        typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t), const T&>::type value)
    {
        T oldValue = *(T*)dest;
        // void* localValue = (void*)&value;
		
#    if defined(__MINGW64__)
        __int64 localCompare = (__int64)&compare;
#    else
        __int64* localCompare = (__int64*)&compare;
#    endif

        union U {
            Impl::cas128_t i;
            T              t;
            inline U() {};
        } newval;

        newval.t = value;

        do
        {
            ;
        } while (_InterlockedCompareExchange128((__int64 volatile*)dest, *(__int64*)& newval.i.upper, *(__int64*)& newval.i.lower, localCompare) == 0);

        return oldValue;
    }

    template<typename T>
    inline T InterlockedAdd(T volatile* const dest, typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t), const T&>::type value)
    {
        const T oldValue = *(T*)dest;
        const T newVal = (*dest + value);

        do
        {
            ;
        } while (oldValue != InterlockedCompareExchange(dest, newVal, oldValue));

        return *(T*)& oldValue;
    }

    template<typename T>
    inline T InterlockedSub(T volatile* const dest, typename std::enable_if<sizeof(T) == sizeof(Impl::cas128_t), const T&>::type value)
    {
        const T oldValue = *(T*)dest;
        const T newVal = (*dest - value);

        do
        {
            ;
        } while (oldValue != InterlockedCompareExchange(dest, newVal, oldValue));

        return *(T*)& oldValue;
    }

#        pragma endregion




#if defined(WIN64) || defined(_WIN64) || defined(_AMD64) || defined(_AMD64_) || defined(__x86_64) || defined(__x86_64__)
    #define MemoryBarrier __faststorefence
#elif defined(_IA64_)
    #define MemoryBarrier __mf
#else
__forceinline void MemoryBarrier() 
    {
        long Barrier;
        __asm {
            xchg Barrier, eax
        }
    }
#endif

    //    template<typename T>
    //    inline T atomic_compare_exchange(T volatile* const dest,
    //                                              const T& compare, _In_
    //                                              typename
    //                                              typename std::enable_if<sizeof(T) ==
    //                                              sizeof(long), const T&>::type
    //                                              value)
    //    {
    //        const long oldValue = *(long*)dest;
    //
    //        do
    //        {
    //            ;
    //        } while (oldValue != ::_InterlockedCompareExchange((long
    //        volatile*)dest, *(long*)&value, *(long*)&compare));
    //
    //        return *(T*)&oldValue;
    //    }
    //
    //    // template<typename T>
    //    // inline T atomic_compare_exchange(volatile T* const dest,
    //    //                                                 const T& compare,
    //    //                                                 typename
    //    std::enable_if_t<sizeof(T) == sizeof(long), const T&> val)
    //    //{
    //    //    union U {
    //    //        long                   i;
    //    //        T                      t;
    //    //        inline U(){};
    //    //    } tmp;
    //
    //    //    tmp.i = _InterlockedCompareExchange((long*)dest, *((long*)&val),
    //    *((long*)&compare));
    //    //    return tmp.t;
    //    //}
    //
    //    template<typename T>
    //    inline T atomic_compare_exchange(volatile T* const dest,
    //                                                     const T& compare,
    //                                                     typename
    //                                                     typename std::enable_if<sizeof(T)
    //                                                     == sizeof(long long),
    //                                                     const T&>::type val)
    //    {
    //        union U {
    //            long long              i;
    //            T                      t;
    //            inline U(){};
    //        } tmp;
    //
    //        tmp.i = _InterlockedCompareExchange64((long long*)dest, *((long
    //        long*)&val), *((long long*)&compare)); return tmp.t;
    //    }
    //
    //    template<typename T>
    //    inline T atomic_compare_exchange(volatile T* const dest,
    //                                                     const T& compare,
    //                                                     typename
    //                                                     typename std::enable_if<sizeof(T)
    //                                                     ==
    //                                                     sizeof(Impl::cas128_t),
    //                                                     const T&>::type val)
    //    {
    //        union U {
    //            Impl::cas128_t         i;
    //            T                      t;
    //            inline U(){};
    //        } tmp, newval;
    //        newval.t = val;
    //        _InterlockedCompareExchange128((long long*)dest, newval.i.upper,
    //        newval.i.lower, ((long long*)&compare)); tmp.t = dest; return tmp.t;
    //    }
    //
    //    template<typename T>
    //    inline T atomic_compare_exchange_strong(volatile T* const dest, const T&
    //    compare, const T& val)
    //    {
    //        return atomic_compare_exchange(dest, compare, val);
    //    }
    //
    //    inline void InterlockedBarrier()
    //    {
    //#    ifdef WIN64
    //        __faststorefence();
    //#    else
    //        //_mm_mfence();
    //
    //        long Barrier;
    //
    //        InterlockedOr(&Barrier, 0);
    //#    endif
    //    }
    //
    //    template<typename T>
    //    T atomic_fetch_or(volatile T* const dest, const T val)
    //    {
    //        T oldval = *dest;
    //        T assume;
    //        do
    //        {
    //            assume   = oldval;
    //            T newval = val | oldval;
    //            oldval   = atomic_compare_exchange(dest, assume, newval);
    //        } while (assume != oldval);
    //
    //        return oldval;
    //    }
    //
    //    template<typename T>
    //    T atomic_fetch_and(volatile T* const dest, const T val)
    //    {
    //        T oldval = *dest;
    //        T assume;
    //        do
    //        {
    //            assume   = oldval;
    //            T newval = val & oldval;
    //            oldval   = atomic_compare_exchange(dest, assume, newval);
    //        } while (assume != oldval);
    //
    //        return oldval;
    //    }
    //
    //    template<typename T>
    //    T atomic_fetch_add(volatile T* const dest, const T val)
    //    {
    //        T oldval = *dest;
    //        T assume;
    //        do
    //        {
    //            assume   = oldval;
    //            T newval = val + oldval;
    //            oldval   = atomic_compare_exchange(dest, assume, newval);
    //        } while (assume != oldval);
    //
    //        return oldval;
    //    }
    //
    //    template<typename T>
    //    T atomic_fetch_sub(volatile T* const dest, const T val)
    //    {
    //        T oldval = *dest;
    //        T assume;
    //        do
    //        {
    //            assume   = oldval;
    //            T newval = val - oldval;
    //            oldval   = atomic_compare_exchange(dest, assume, newval);
    //        } while (assume != oldval);
    //
    //        return oldval;
    //    }
    //
    //    template<typename T>
    //    T atomic_exchange(volatile T* const dest, const T val)
    //    {
    //        T oldval = *dest;
    //        T assume;
    //        do
    //        {
    //            assume = oldval;
    //            oldval = atomic_compare_exchange(dest, assume, val);
    //        } while (assume != oldval);
    //
    //        return oldval;
    //    }
    //
    //    // template<typename T>
    //    // void atomic_or(volatile T* const dest, const T val)
    //    //{
    //    //    atomic_fetch_or(dest, val);
    //    //}
    //
    //    // template<typename T>
    //    // void atomic_and(volatile T* const dest, const T val)
    //    //{
    //    //    atomic_fetch_and(dest, val);
    //    //}
    //
    //    // template<typename T>
    //    // void atomic_add(volatile T* const dest, const T val)
    //    //{
    //    //    atomic_fetch_add(dest, val);
    //    //}
    //
    //    // template<typename T>
    //    // void atomic_sub(volatile T* const dest, const T val)
    //    //{
    //    //    atomic_fetch_sub(dest, val);
    //    //}
    //
    //    template<typename T>
    //    void atomic_assign(volatile T* const dest, const T val)
    //    {
    //        atomic_fetch_exchange(dest, val);
    //    }
    //
    //    // template<typename T>
    //    // void atomic_increment(volatile T* const dest)
    //    //{
    //    //    T oldval = *dest;
    //    //    T assume;
    //    //    do
    //    //    {
    //    //        assume   = oldval;
    //    //        T newval = assume++;
    //    //        oldval   = atomic_compare_exchange(dest, assume, newval);
    //    //    } while (assume != oldval);
    //    //}
    //
    //    // template<typename T>
    //    // void atomic_decrement(volatile T* const dest)
    //    //{
    //    //    T oldval = *dest;
    //    //    T assume;
    //    //    do
    //    //    {
    //    //        assume   = oldval;
    //    //        T newval = assume--;
    //    //        oldval   = atomic_compare_exchange(dest, assume, newval);
    //    //    } while (assume != oldval);
    //    //}

} // namespace Kokkos
#    endif
#endif

