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
        using Type = PMacc::uint64_cu;

        struct OutputTransformer
        {
            HDINLINE Type
            operator()(const Type val) const
            {
                return val;
            }
        };

        explicit CountParticles(uint32_t curTimestep, const DetectorConfig& detector)
        {}

        template<typename T_DetectorBox, typename T_Particle >
        DINLINE void
        operator()(T_DetectorBox detectorBox, const Space2D& targetCellIdx, T_Particle& particle, const Space& globalCellIdx) const
        {
            Type& oldVal = detectorBox(targetCellIdx);
            atomicAdd(&oldVal, 1);
        }
    };

}  // namespace detector
}  // namespace parataxis
