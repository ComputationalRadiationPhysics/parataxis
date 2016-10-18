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

namespace xrt{
namespace particles {
namespace pusher {

    /**
     * Pusher used to move particles with the speed of light
     */
    struct Photon
    {
        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox&, T_Position& pos, T_Direction& dir)
        {
            // Allow higher precision in position
            using PosType = typename T_Position::type;
            for(uint32_t d=0; d<simDim; ++d)
            {
                // This assumes a unit vector for the direction, otherwise we need to normalize it here
                pos[d] += (static_cast<PosType>(dir[d]) * static_cast<PosType>(SPEED_OF_LIGHT) * static_cast<PosType>(DELTA_T)) / static_cast<PosType>(cellSize[d]);
            }
        }
    };

}  // namespace pusher
}  // namespace particles
}  // namespace xrt
