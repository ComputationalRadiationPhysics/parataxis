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

namespace parataxis {
namespace particles {
namespace functors {

    namespace detail {

        template<typename T_Particle, bool hasAmplitude = HasIdentifier_t<T_Particle, amplitude<> >::value>
        struct GetAmplitude
        {
            HDINLINE float_X
            operator()(const T_Particle& particle) const
            {
                return particle[amplitude_];
            }
        };

        template<typename T_Particle>
        struct GetAmplitude<T_Particle, false>
        {
            // TODO: Use T_Particle directly after PIConGPU PR #1604
            using Amplitude = GetResolvedFlag_t<typename T_Particle::FrameType, amplitude<> >;

            HDINLINE float_X
            operator()(const T_Particle&) const
            {
                return Amplitude::getValue();
            }
        };

    }  // namespace detail

    /**
     * Return a functor that returns the amplitude of a particle
     */
    template<typename T_Particle>
    using GetAmplitude = detail::GetAmplitude<T_Particle>;

}  // namespace functors
}  // namespace particles
}  // namespace parataxis
