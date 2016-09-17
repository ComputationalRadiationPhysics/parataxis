#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/IterateSpecies.hpp"
#include "particles/functors/CopySpeciesToHost.hpp"
#include "particles/filters/IndexFilter.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <particles/memory/buffers/MallocMCBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <dataManagement/DataConnector.hpp>
#include <traits/HasIdentifier.hpp>
#include <debug/VerboseLog.hpp>
#include <string>

namespace xrt {
namespace plugins {

    template<class T_Particles>
    class DebugHelper : public ISimulationPlugin
    {
        uint32_t notifyFrequency;
        std::string analyzerName;
        std::string analyzerPrefix;

    public:
        DebugHelper():
            notifyFrequency(0),
            analyzerName("Debug: Write some debug information to std::cout"),
            analyzerPrefix(T_Particles::FrameType::getName() + std::string("_debug"))
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        virtual ~DebugHelper()
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((analyzerPrefix + ".period").c_str(), po::value<uint32_t >(&notifyFrequency), "enable analyzer [for each n-th step]")
                ;
        }

        std::string pluginGetName() const override
        {
            return analyzerName;
        }

        void notify(uint32_t currentStep) override
        {
            constexpr size_t frameSize = sizeof(typename T_Particles::FrameType);
            uint32_t slotsAv = mallocMC::getAvailableSlots(frameSize);
            uint64_t numParts = slotsAv * SuperCellSize::toRT().productOfComponents();
            std::cout <<
                    (boost::format("Debug info at timestep %1%: There are %2% slots available (%3% Byte each) that can fit up to %4% particles")
                    % currentStep % slotsAv % frameSize % numParts)
                    << std::endl;
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
        {}
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {}

    protected:
        void pluginLoad() override
        {
            if(!notifyFrequency)
                return;

            Environment::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            PMacc::log< XRTLogLvl::PLUGINS >("Printing particle info for %1% every %2% timesteps") % T_Particles::FrameType::getName() % notifyFrequency;
        }
    };

}  // namespace plugins
}  // namespace xrt
