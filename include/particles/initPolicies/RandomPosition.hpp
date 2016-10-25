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

#include "parataxisTypes.hpp"
#include <random/distributions/Uniform.hpp>

namespace parataxis {
namespace particles {
namespace initPolicies {

    template<class T_Species>
    struct RandomPosition
    {
#if PARATAXIS_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;
#endif

        HINLINE RandomPosition(uint32_t /*timestep*/)
#if !PARATAXIS_USE_SLOW_RNG
                :rand(RNGProvider::createRandom<Distribution>())
#endif
        {}

        DINLINE void
        init(Space localCellIdx)
        {
            rand.init(localCellIdx);
        }

        HDINLINE void
        setCount(uint32_t /*particleCount*/)
        {}

        DINLINE position_pic::type
        operator()(uint32_t /*numPart*/)
        {
            position_pic::type result;
            result.x() = 0;
            for(uint32_t i = 1; i < simDim; ++i)
                result[i] = rand() * laserConfig::distSize[i] / cellSize[i];
            return result;
        }
    private:
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace parataxis
