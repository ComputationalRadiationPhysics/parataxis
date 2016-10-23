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
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <debug/VerboseLog.hpp>
#include <string>

namespace parataxis {
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
            PMacc::log< PARATAXISLogLvl::PLUGINS >("Printing particle info for %1% every %2% timesteps") % T_Particles::FrameType::getName() % notifyFrequency;
        }
    };

}  // namespace plugins
}  // namespace parataxis
