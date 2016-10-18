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
