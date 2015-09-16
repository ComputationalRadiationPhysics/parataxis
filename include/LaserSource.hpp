#pragma once

#include "xrtTypes.hpp"

namespace xrt {

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
        using Momentum     = Resolve_t<laserConfig::momentum::UsedValue>;

        static constexpr uint32_t numTimeStepsLaserPulse = laserConfig::PULSE_LENGTH / UNIT_TIME / DELTA_T;
        uint32_t timeStepsProcessed = 0;

    public:

        void init()
        {
            PMacc::log< XRTLogLvl::DOMAINS >("Laser pulse is %1% timesteps long") % numTimeStepsLaserPulse;
            uint32_t slotsAv = mallocMC::getAvailableSlots(sizeof(FrameType));
            uint64_t numParts = slotsAv * SuperCellSize::toRT().productOfComponents();
            PMacc::log< XRTLogLvl::MEMORY > ("There are %1% slots available that can fit up to %2% particles") % slotsAv % numParts;
        }

        void processStep(uint32_t currentStep)
        {
            if(timeStepsProcessed < numTimeStepsLaserPulse){
                uint32_t numTimeSteps = 1;
                addParticles(timeStepsProcessed, numTimeSteps);
                timeStepsProcessed += numTimeSteps;
            }
        }

    private:
        void addParticles(uint32_t timeStep, uint32_t numTimeSteps)
        {
            Space totalSize = Environment::get().SubGrid().getTotalDomain().size;
            auto initFunctor = particles::getParticleFillInfo(
                    Distribution(totalSize.shrink<simDim-1>(laserConfig::DIRECTION + 1)),
                    Position(),
                    Phase(),
                    Momentum()
                    );
            auto& dc = Environment::get().DataConnector();
            Species& particles = dc.getData<Species>(FrameType::getName(), true);
            particles.add(initFunctor, timeStep, numTimeSteps);
            dc.releaseData(FrameType::getName());
        }
    };

}  // namespace xrt
