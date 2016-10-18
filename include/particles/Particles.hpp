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
#include <particles/ParticlesBase.hpp>
#include <particles/memory/buffers/ParticlesBuffer.hpp>

#include <dataManagement/ISimulationData.hpp>

namespace xrt{

namespace fields {
    class DensityField;
}  // namespace fields

    template<typename T_ParticleDescription>
    class Particles : public PMacc::ParticlesBase<T_ParticleDescription, MappingDesc>, public PMacc::ISimulationData
    {
    public:

        typedef PMacc::ParticlesBase<T_ParticleDescription, MappingDesc> ParticlesBaseType;
        typedef typename ParticlesBaseType::BufferType BufferType;
        typedef typename ParticlesBaseType::FrameType FrameType;
        typedef typename ParticlesBaseType::FrameTypeBorder FrameTypeBorder;
        typedef typename ParticlesBaseType::ParticlesBoxType ParticlesBoxType;

        Particles(MappingDesc cellDescription, PMacc::SimulationDataId datasetID);

        virtual ~Particles();

        void createParticleBuffer();

        void init(fields::DensityField* densityField);
        /**
         * Adds particles to the grid
         * \tparam T_DistributionFunctor Functor that returns number of particles for a given total GPU cell idx
         * \tparam T_PositionFunctor     Functor that returns a position for a given particle idx
         *                               Must also provide an init(totalGPUCellIdx, totalNumParToCreate) function
         */
        template<typename T_InitFunctor>
        void add(T_InitFunctor&& initFunctor, uint32_t timeStep);

        void update(uint32_t currentStep);

        PMacc::SimulationDataId getUniqueId() override;

        /** sync device data to host
         *
         * ATTENTION: - in the current implementation only supercell meta data are copied!
         *            - the shared (between all species) mallocMC buffer must be copied once
         *              by the user
         */
        void synchronize() override;

        void syncToDevice() override;

        /**
         * Handles particles that went out of the volume
         * @param direction
         */
        void processLeavingParticles(int32_t direction);

    private:
        PMacc::SimulationDataId datasetID;
        PMacc::GridLayout<simDim> gridLayout;
        fields::DensityField* densityField_;
        uint32_t lastProcessedStep_;
    };

} //namespace xrt
