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

#include "xrtTypes.hpp"

namespace xrt {
namespace math {

    /** Rounds the given float value to the nearest integer.
     *  .5 cases are rounded towards +infinity
     *  Usable in constexpr
     */
    constexpr int floatToIntRound(const float_32 val)
    {
        return static_cast<int>(val + float_32(.5));
    }

    /** Rounds the given float value to the nearest integer.
     *  .5 cases are rounded towards +infinity
     *  Usable in constexpr
     */
    constexpr int floatToIntRound(const float_64 val)
    {
        return static_cast<int>(val + float_64(.5));
    }

}  // namespace math
}  // namespace xrt
