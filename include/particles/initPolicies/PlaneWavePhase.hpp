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
#include "particles/functors/GetPhaseByTimestep.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Returns a phase that is only time-variant (e.g. the case with plane waves)
     */
    template<class T_Species>
    struct PlaneWavePhase
    {
        using Species = T_Species;
        float_X curPhase;

        PlaneWavePhase(float_64 phi_0, uint32_t currentStep)
        {
            // Get current phase (calculated exactly in high precision)
            curPhase = functors::GetPhaseByTimestep<Species>()(currentStep, phi_0);
        }

        DINLINE void
        init(Space localCellIdx) const
        {}

        DINLINE float_X
        operator()(uint32_t timeStep) const
        {
            return curPhase;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
