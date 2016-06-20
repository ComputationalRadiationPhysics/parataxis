#pragma once

#include "xrtTypes.hpp"
#include "fields/IFieldManipulator.hpp"
#include "plugins/ISimulationPlugin.hpp"

namespace xrt {
namespace fields {

template<class T_Field>
class FieldManipulator: public IFieldManipulator, public ISimulationPlugin
{
public:

    FieldManipulator()
    {
        Environment::get().PluginConnector().registerPlugin(this);
    }

    void notify( uint32_t currentStep ) override {}

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override {}

    void restart(uint32_t restartStep, const std::string restartDirectory) override {}

    void pluginRegisterHelp(boost::program_options::options_description& desc) override {}

    std::string pluginGetName() const override {
        return T_Field::getName() + "-Manipulator";
    }

    void update(uint32_t currentStep) override
    {}
};

}  // namespace fields
}  // namespace xrt
