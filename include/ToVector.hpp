/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
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
