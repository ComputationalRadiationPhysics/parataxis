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
#include "particles/functors/GetWavelength.hpp"

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Returns the angular frequency (omega) of a species
     */
    template<typename T_Species>
    struct GetAngularFrequency
    {
        using Species = T_Species;

        HDINLINE float_X
        operator()() const
        {
            // k = 2 * PI / lambda
            const float_X waveNumber = 2 * float_X(PI) / GetWavelength<Species>()();
            // Assume vacuum in the medium -> w = k * v = k * c
            const float_X omega = waveNumber * SPEED_OF_LIGHT;
            return omega;
        }

    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
