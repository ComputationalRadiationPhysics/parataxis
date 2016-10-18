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

    class NoDetectorImpl: PMacc::ISimulationData
    {
        using Buffer = PMacc::GridBuffer<int, simDim>;
    public:
        struct DetectParticle
        {
            template<typename T_Particle, typename T_DetectorBox>
            HDINLINE void
            operator()(const T_Particle& particle, const Space superCellPosition, T_DetectorBox& detector) const
            {}
        };

        static std::string
        getName()
        {
            return "NoDetector";
        }

        PMacc::SimulationDataId getUniqueId() override
        {
            return getName();
        }

        void synchronize() override
        {}

        void init()
        {}

        typename Buffer::DataBoxType
        getHostDataBox()
        {
            return Buffer::DataBoxType();
        }

        typename Buffer::DataBoxType
        getDeviceDataBox()
        {
            return Buffer::DataBoxType();
        }

        Space2D
        getSize() const
        {
            return Space2D();
        }

        DetectParticle
        getDetectParticle(uint32_t timeStep) const
        {
            return DetectParticle();
        }
    };

}  // namespace detector
}  // namespace parataxis
