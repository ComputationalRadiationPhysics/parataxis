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
#include "debug/LogLevels.hpp"
#include "math/Max.hpp"
#include "math/round.hpp"
#include <nvidia/reduce/Reduce.hpp>
#include <algorithms/TypeCast.hpp>

namespace xrt {

    namespace kernel {

        template<class T_CtBox, class T_ParticleFillInfo, class T_Mapper>
        __global__ void
        countSpawnedPhotons(T_CtBox ctBox, T_ParticleFillInfo particleFillInfo, const Space localOffset, uint32_t numTimesteps, T_Mapper mapper)
        {
            static_assert(T_Mapper::AreaType == PMacc::BORDER, "Only borders should be filled");
            const Space superCellIdx(mapper.getSuperCellIndex(Space(blockIdx)));

            /*get local cell idx  */
            const Space localCellIndex = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT() + Space(threadIdx);

            const Space totalCellIdx = localOffset + localCellIndex;
            if(simDim == 3 && totalCellIdx[laserConfig::DIRECTION] != 0)
                return;
            particleFillInfo.init(localCellIndex);
            for(uint32_t timeStep = 0; timeStep < numTimesteps; timeStep++)
            {
                const uint32_t numParts = particleFillInfo.getCount(timeStep);
                atomicAdd(&ctBox(timeStep), numParts);
            }
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
        using Distribution = Resolve_t<laserConfig::distribution::UsedValue>;
        using Position     = Resolve_t<laserConfig::position::UsedValue>;
        using Phase        = Resolve_t<laserConfig::phase::UsedValue>;
        using Direction    = Resolve_t<laserConfig::direction::UsedValue>;

        // Calculate the number of timesteps the laser is active. It is rounded to the nearest timestep (so error is at most half a timestep)
        static constexpr uint32_t numTimeStepsLaserPulse = math::floatToIntRound(laserConfig::PULSE_LENGTH / UNIT_TIME / DELTA_T);
        static constexpr float_64 phi_0 = 0; // Phase offset at t = 0 (in range [0, 2*PI) )

        static_assert(laserConfig::DIRECTION >= 0 && laserConfig::DIRECTION <= 2, "Invalid laser direction");

    public:

        void init()
        {
            PMacc::log< XRTLogLvl::DOMAINS >("Laser pulse is %1% timesteps long") % numTimeStepsLaserPulse;
            uint32_t slotsAv = mallocMC::getAvailableSlots(sizeof(FrameType));
            uint64_t numParts = slotsAv * SuperCellSize::toRT().productOfComponents();
            PMacc::log< XRTLogLvl::MEMORY > ("There are %1% slots available that can fit up to %2% particles") % slotsAv % numParts;
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
            if(numTimeStepsLaserPulse < numTimesteps)
                numTimesteps = numTimeStepsLaserPulse;
            PMacc::DeviceBufferIntern<PMacc::uint64_cu, 1> ctPerTimestep(numTimesteps);
            const SubGrid& subGrid = Environment::get().SubGrid();
            const Space localOffset = subGrid.getLocalDomain().offset;
            /* Add only to first cells */
            if(simDim == 3 && localOffset[laserConfig::DIRECTION] > 0)
                return;

            const PMacc::BorderMapping<MappingDesc> mapper(cellDescription, laserConfig::EXCHANGE_DIR);
            Space block = MappingDesc::SuperCellSize::toRT();
            if(simDim == 3)
                block[laserConfig::DIRECTION] = 1;
            auto initFunctor = getInitFunctor(0);
            __cudaKernel(kernel::countSpawnedPhotons)
                (mapper.getGridDim(), block.toDim3())
                ( ctPerTimestep.getDataBox(),
                  initFunctor,
                  localOffset,
                  numTimesteps,
                  mapper
                  );
            PMacc::nvidia::reduce::Reduce reduce(1024);
            const uint64_t maxPartPerTs = reduce(math::Max(), ctPerTimestep.getBasePointer(), ctPerTimestep.getCurrentSize());
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
            PMacc::log< XRTLogLvl::MEMORY >(msg.c_str())
                            % maxPartPerTs
                            % maxPartsPerTsStraight % (maxPartsPerTsStraight < maxPartPerTs ? "(!)":"")
                            % maxPartsPerTsDiagonal % (maxPartsPerTsDiagonal < maxPartPerTs ? "(!)":"");
        }

    private:
        particles::ParticleFillInfo<Distribution, Position, Phase, Direction>
        getInitFunctor(uint32_t timeStep) const
        {
           return particles::getParticleFillInfo(
                    Distribution(),
                    Position(),
                    Phase(phi_0, timeStep),
                    Direction()
                    );
        }

        void addParticles(uint32_t timeStep)
        {
            auto initFunctor = getInitFunctor(timeStep);
            auto& dc = Environment::get().DataConnector();
            Species& particles = dc.getData<Species>(FrameType::getName(), true);
            particles.add(initFunctor, timeStep);
            dc.releaseData(FrameType::getName());
        }
    };

}  // namespace xrt
