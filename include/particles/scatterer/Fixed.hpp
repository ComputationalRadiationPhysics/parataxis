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
     * Scatterer that sets the direction to a fixed value
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct Fixed
    {
        float3_X direction_;

        HINLINE explicit
        Fixed(uint32_t)
        {
            using namespace PMacc::algorithms::precisionCast;
            float3_64 tmpDir;
            tmpDir.x() = 1;
            tmpDir.y() = PMaccMath::tan(float_64(T_Config::angleY));
            tmpDir.z() = PMaccMath::tan(float_64(T_Config::angleZ));
            direction_ = precisionCast<float_X>(tmpDir / PMaccMath::abs(tmpDir));
        }

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Direction& dir)
        {
            dir = direction_;
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
