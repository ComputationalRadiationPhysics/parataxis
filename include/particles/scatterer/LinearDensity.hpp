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
#include "ToVector.hpp"
#include <algorithms/math.hpp>
#include <algorithms/TypeCast.hpp>

namespace xrt{
namespace particles {
namespace scatterer {

    /**
     * Scatterer that sets the direction linearly to the density
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct LinearDensity
    {
        HINLINE explicit
        LinearDensity(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Direction& dir)
        {
            dir.x() = 1;
            dir.y() = PMaccMath::tan<trigo_X>(float_X(T_Config::factorY) * density(Space::create(0)));
            dir.z() = PMaccMath::tan<trigo_X>(float_X(T_Config::factorZ) * density(Space::create(0)));
            dir = dir / PMaccMath::sqrt<sqrt_X>(PMaccMath::abs2(dir));
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
