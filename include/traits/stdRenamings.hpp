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

#include <type_traits>

namespace xrt {
namespace traits {

    //Shortcuts from C++14 standard

    template< class T >
    using result_of_t = typename std::result_of<T>::type;

    template< class T >
    using remove_reference_t = typename std::remove_reference<T>::type;

    template< class T >
    using remove_const_t = typename std::remove_const<T>::type;

    template< class T >
    using remove_cv_t = typename std::remove_cv<T>::type;

}  // namespace traits
}  // namespace xrt
