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
#include "debug/LogLevels.hpp"
#include "math/Max.hpp"
#include "math/round.hpp"
#include <nvidia/reduce/Reduce.hpp>
#include <algorithms/TypeCast.hpp>

namespace parataxis {

    namespace kernel {

        template<class T_CtBox, class T_ParticleFillInfo, class T_Mapper>
        __global__ void
        countSpawnedPhotons(T_CtBox ctBox, T_ParticleFillInfo particleFillInfo, const Space localOffset, T_Mapper mapper)
        {
            static_assert(T_Mapper::AreaType == PMacc::BORDER, "Only borders should be filled");
            const Space superCellIdx(mapper.getSuperCellIndex(Space(blockIdx)));

            /*get local cell idx  */
            const Space localCellIndex = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT() + Space(threadIdx);

            const Space totalCellIdx = localOffset + localCellIndex;
            if(simDim == 3 && totalCellIdx[laserConfig::DIRECTION] != 0)
                return;
            particleFillInfo.init(localCellIndex);
            const float_X numPhotons = particleFillInfo.getPhotonCount();
            const uint32_t numParts = particleFillInfo.getCount(numPhotons);
            atomicAdd(&ctBox(0),  numParts);
        }

    }  // namespace kernel

    /**
     * Simulates a laser source that spawns particles of the given species
     */
    template<class T_Species>
    class LaserSource
    {
        using Species   = T_Species;
        using FrameType = typename Species::FrameType;
        using PhotonCount  = Resolve_t<laserConfig::photonCount::UsedValue>;
        using Distribution = Resolve_t<laserConfig::distribution::UsedValue>;
        using Position     = Resolve_t<laserConfig::position::UsedValue>;
        using Phase        = Resolve_t<laserConfig::phase::UsedValue>;
        using Direction    = Resolve_t<laserConfig::direction::UsedValue>;

// Consistency check
#if !PARATAXIS_WEIGHTED_PHOTONS
        static_assert(std::is_same<Distribution, particles::initPolicies::EqualPhotonsDistribution>::value,
                "Number of particles not equal to number of photons, define PARATAXIS_WEIGHTED_PHOTONS=1 to enable this");
#endif

        // Calculate the number of timesteps the laser is active. It is rounded to the nearest timestep (so error is at most half a timestep)
        static constexpr uint32_t numTimeStepsLaserPulse = math::floatToIntRound(laserConfig::PULSE_LENGTH / UNIT_TIME / DELTA_T);
        static constexpr float_64 phi_0 = 0; // Phase offset at t = 0 (in range [0, 2*PI) )

        static_assert(laserConfig::DIRECTION >= 0 && laserConfig::DIRECTION <= 2, "Invalid laser direction");

    public:
        using ParticleFillInfo = particles::ParticleFillInfo<PhotonCount, Distribution, Position, Phase, Direction>;

        void init()
        {
            PMacc::log< PARATAXISLogLvl::DOMAINS >("Laser pulse is %1% timesteps long") % numTimeStepsLaserPulse;
            uint32_t slotsAv = mallocMC::getAvailableSlots(sizeof(FrameType));
            uint64_t numParts = slotsAv * SuperCellSize::toRT().productOfComponents();
            PMacc::log< PARATAXISLogLvl::MEMORY > ("There are %1% slots available that can fit up to %2% particles") % slotsAv % numParts;
        }

        void update(uint32_t currentStep)
        {
            if(currentStep < numTimeStepsLaserPulse){
                addParticles(currentStep);
            }
        }

        void reset(uint32_t currentStep)
        {}

        void checkPhotonCt(uint32_t numTimesteps, MappingDesc cellDescription)
        {
            const SubGrid& subGrid = Environment::get().SubGrid();
            const Space localOffset = subGrid.getLocalDomain().offset;
            /* Add only to first cells */
            if(simDim == 3 && localOffset[laserConfig::DIRECTION] > 0)
                return;

            PMacc::DeviceBufferIntern<PMacc::uint64_cu, 1> ctPerTimestep(numTimesteps);

            const PMacc::BorderMapping<MappingDesc> mapper(cellDescription, laserConfig::EXCHANGE_DIR);
            Space blockSize = MappingDesc::SuperCellSize::toRT();
            if(simDim == 3)
                blockSize[laserConfig::DIRECTION] = 1;
            if(numTimeStepsLaserPulse < numTimesteps)
                numTimesteps = numTimeStepsLaserPulse;
            for(uint32_t timestep = 0; timestep < numTimesteps; ++timestep)
            {
                auto initFunctor = getInitFunctor(timestep);
                __cudaKernel(kernel::countSpawnedPhotons)
                    (mapper.getGridDim(), blockSize.toDim3())
                    ( ctPerTimestep.getDataBox().shift(timestep),
                      initFunctor,
                      localOffset,
                      mapper
                      );
            }
            PMacc::nvidia::reduce::Reduce reduce(1024);
            const uint64_t maxPartPerTs = reduce(math::Max(), ctPerTimestep.getBasePointer(), ctPerTimestep.getDataSpace().productOfComponents());
            float_64 sizeStraight, sizeDiagonal;
            if(simDim == 3)
                sizeStraight = precisionCast<float_64>(subGrid.getLocalDomain().size[laserConfig::DIRECTION]) * cellSize[laserConfig::DIRECTION];
            else
                sizeStraight = cellSize.x();
            sizeDiagonal = PMaccMath::abs(precisionCast<float_64>(subGrid.getLocalDomain().size) * precisionCast<float_64>(cellSize.toRT()));
            const uint64_t numTsStraight = PMaccMath::ceil(sizeStraight / SPEED_OF_LIGHT / DELTA_T);
            const uint64_t numTsDiagonal = PMaccMath::ceil(sizeDiagonal / SPEED_OF_LIGHT / DELTA_T);
            const uint64_t slotsAv = mallocMC::getAvailableSlots(sizeof(FrameType));
            const uint64_t numPartsAv = slotsAv * SuperCellSize::toRT().productOfComponents();
            /* round down */
            const uint64_t maxPartsPerTsStraight = numPartsAv / numTsStraight;
            const uint64_t maxPartsPerTsDiagonal = numPartsAv / numTsDiagonal;
            std::string msg;
            if(maxPartsPerTsDiagonal < maxPartPerTs || maxPartsPerTsStraight < maxPartPerTs)
            {
                msg = "There will be up to %1% particles spawned per timestep which can be to many to fit into the memory. "
                      "The maximum number of particles that could be spawned per timestep is %2%%3% for straight through propagation and %4%%5% for diagonal propagation";
            }else{
                msg = "There will be up to %1% particles spawned per timestep which most likely fit into the memory. "
                      "The maximum number of particles that could be spawned per timestep is %2%%3% for straight through propagation and %4%%5% for diagonal propagation";
            }
            PMacc::log< PARATAXISLogLvl::MEMORY >(msg.c_str())
                            % maxPartPerTs
                            % maxPartsPerTsStraight % (maxPartsPerTsStraight < maxPartPerTs ? "(!)":"")
                            % maxPartsPerTsDiagonal % (maxPartsPerTsDiagonal < maxPartPerTs ? "(!)":"");
        }

    private:
        ParticleFillInfo
        getInitFunctor(uint32_t timeStep) const
        {
           return ParticleFillInfo(timeStep, phi_0);
        }

        void addParticles(uint32_t timeStep)
        {
            auto initFunctor = getInitFunctor(timeStep);
            auto& dc = Environment::get().DataConnector();
            Species& particles = dc.getData<Species>(FrameType::getName(), true);
            particles.add(initFunctor);
            dc.releaseData(FrameType::getName());
        }
    };

}  // namespace parataxis
