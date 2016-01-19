#pragma once

#include <math/vector/Vector.hpp>
#include <type_traits>

namespace xrt {

    namespace detail {

        /**
         * Converts a struct with members X, Y, Z to a vector
         */
        template<class T_Src, int32_t T_dim, typename T_Type = typename std::remove_cv<decltype(T_Src::X)>::type >
        struct ToVector;

        template<class T_Src, typename T_Type>
        struct ToVector<T_Src, 1, T_Type>
        {
            using Type = T_Type;
            using Src = T_Src;

            HDINLINE PMacc::math::Vector<Type, 1>
            operator()() const
            {
                return PMacc::math::Vector<Type, 1>(Src::X);
            }
        };

        template<class T_Src, typename T_Type>
        struct ToVector<T_Src, 2, T_Type>
        {
            using Type = T_Type;
            using Src = T_Src;

            HDINLINE PMacc::math::Vector<Type, 2>
            operator()() const
            {
                return PMacc::math::Vector<Type, 2>(Src::X, Src::Y);
            }
        };

        template<class T_Src, typename T_Type>
        struct ToVector<T_Src, 3, T_Type>
        {
            using Type = T_Type;
            using Src = T_Src;

            HDINLINE PMacc::math::Vector<Type, 3>
            operator()() const
            {
                return PMacc::math::Vector<Type, 3>(Src::X, Src::Y, Src::Z);
            }
        };

    }  // namespace detail


    /**
     * Converts a struct with members X, Y, Z to a vector
     */
    template<class T_Src, int32_t T_dim, typename T_Type = typename std::remove_cv<decltype(T_Src::X)>::type>
    using ToVector = detail::ToVector<T_Src, T_dim, T_Type>;

}  // namespace xrt
