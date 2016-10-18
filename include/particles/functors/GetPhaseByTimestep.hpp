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
#include "particles/functors/GetAngFrequency.hpp"

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Returns the phase for a given timestep
     */
    template<typename T_Species>
    struct GetPhaseByTimestep
    {
        using Species = T_Species;

        template<typename T_Result = float_X>
        HINLINE T_Result
        operator()(const uint32_t timestep, T_Result phi_0 = 0) const
        {
            /* phase phi = phi_0 - omega * t;
             * Note: This MUST be calculated in double precision as single precision is inexact after ~100 timesteps
             *       Double precision is enough for about 10^10 timesteps
             *       More timesteps (in SP&DP) are possible, if the product is implemented as a summation with summands reduced to 2*PI */
            static const float_64 omega = GetAngularFrequency<Species>()();
            static const float_64 phaseDiffPerTimestep = fmod(omega * DELTA_T, 2 * PI);
            // Reduce summands to range of 2*PI to avoid bit canceling
            float_64 dPhi = fmod(phaseDiffPerTimestep * static_cast<float_64>(timestep), 2 * PI);
            phi_0 = fmod(phi_0, 2 * PI);
            float_64 result = phi_0 - dPhi;
            // Keep in range of [0,2*PI)
            if(result < 0)
                result += 2*PI;
            return result;
        }
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
