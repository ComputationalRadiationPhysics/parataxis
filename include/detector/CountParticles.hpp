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
#include "particles/functors/GetAmplitude.hpp"
#include <basicOperations.hpp>

namespace parataxis {
namespace detector {

    struct DetectorConfig;

    /**
     * Functor that can be used as an AccumPolicy for \see PhotonDetector
     * It simply counts the number of particles for each cell
     */
    template<class T_Species = bmpl::_1>
    class CountParticles
    {
    public:
        using Type = float_64;

        struct OutputTransformer
        {
            HDINLINE uint64_t
            operator()(const Type val) const
            {
                // Round to nearest uint64_t
                return static_cast<uint64_t>(val + 0.5);
            }
        };

        explicit CountParticles(uint32_t curTimestep, const DetectorConfig& detector)
        {}

        template<typename T_DetectorBox, typename T_Particle >
        DINLINE void
        operator()(T_DetectorBox detectorBox, const Space2D& targetCellIdx, const T_Particle& particle, const Space& globalCellIdx) const
        {
            Type& oldVal = detectorBox(targetCellIdx);
            const auto amplitude = particles::functors::GetAmplitude<T_Particle>()(particle);
            PMacc::atomicAddWrapper(&oldVal, amplitude);
        }
    };

}  // namespace detector
}  // namespace parataxis
