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
#include <random/distributions/Uniform.hpp>

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Condition functor for scattering that depends on a probability that can be dependent
     * on the density in the particles cell.
     * The Config needs a static function calcProbability(float_X density) the returns the
     * probability for scattering in the range [0, 1]
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct OnRandom
    {
        using Config = T_Config;
#if XRT_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;
#endif

        HINLINE explicit
        OnRandom(uint32_t currentStep)
#if !XRT_USE_SLOW_RNG
                :rand(RNGProvider::createRandom<Distribution>())
#endif
        {}

        DINLINE void
        init(Space localCellIdx)
        {
            rand.init(localCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        DINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Direction& dir)
        {
            float_X probability = Config::calcProbability(density(Space::create(0)));
            return (rand() < probability);
        }

    private:
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
