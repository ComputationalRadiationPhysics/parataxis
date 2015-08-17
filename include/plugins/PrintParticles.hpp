#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/IterateSpecies.hpp"
#include "particles/functors/CopySpeciesToHost.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <particles/memory/buffers/MallocMCBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <string>

namespace xrt {
namespace plugins {

    namespace detail {

        struct PrintParticle
        {
            template<class T_Particle>
            void
            operator()(const Space globalIdx, T_Particle&& particle)
            {
                std::cout << "Particle " << globalIdx << ": " << particle[position_] << "\n";
            }
        };

    }  // namespace detail

    template<class T_ParticlesType>
    class PrintParticles : public ISimulationPlugin
    {
        using ParticlesType = T_ParticlesType;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef floatD_X FloatPos;

        uint32_t notifyFrequency;

        std::string analyzerName;
        std::string analyzerPrefix;

    public:
        PrintParticles():
            notifyFrequency(0),
            analyzerName("PositionsParticles: write position of all particles of a species to std::cout"),
            analyzerPrefix(ParticlesType::FrameType::getName() + std::string("_position"))
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PrintParticles()
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((analyzerPrefix + ".period").c_str(),
                 po::value<uint32_t > (&notifyFrequency), "enable analyzer [for each n-th step]");
        }

        std::string pluginGetName() const override
        {
            return analyzerName;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Printing particles at timestep %1%") % currentStep;
            PMacc::DataConnector &dc = Environment::get().DataConnector();

            /* synchronizes the MallocMCBuffer to the host side */
            PMacc::MallocMCBuffer& mallocMCBuffer = dc.getData<PMacc::MallocMCBuffer>(PMacc::MallocMCBuffer::getName());
            //particles::functors::CopySpeciesToHost<PIC_Photons>()();

            int particlesCount = 0;
            auto& particles = dc.getData<PIC_Photons>(PIC_Photons::FrameType::getName());
            const Space localOffset = Environment::get().SubGrid().getLocalDomain().offset;
            PMacc::AreaMapping< PMacc::CORE + PMacc::BORDER, MappingDesc > mapper(*cellDescription_);
            particles::functors::IterateSpecies<PIC_Photons>()(
                    particlesCount,
                    particles.getHostParticlesBox(mallocMCBuffer.getOffset()),
                    localOffset,
                    mapper,
                    detail::PrintParticle()
                    );

            dc.releaseData(PIC_Photons::FrameType::getName());
            dc.releaseData(PMacc::MallocMCBuffer::getName());
            PMacc::log< XRTLogLvl::IN_OUT >("%1% particles printed") % particlesCount;
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
        {}
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {}

    protected:
        void pluginLoad() override
        {
            Environment::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
        }
    };

}  // namespace plugins
}  // namespace xrt
