#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include <dataManagement/DataConnector.hpp>
#include <string>

namespace xrt {
namespace plugins {

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
            analyzerName("PositionsParticles: write position of one particle of a species to std::cout"),
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
            PMacc::DataConnector &dc = Environment::get().DataConnector();

            ParticlesType* particles = &(dc.getData<ParticlesType>(ParticlesType::FrameType::getName(), true));
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
