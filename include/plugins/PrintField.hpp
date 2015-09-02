#pragma once

#include "xrtTypes.hpp"
#include "GatherSlice.hpp"
#include "PngCreator.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <string>
#include <sstream>

namespace xrt {
namespace plugins {

    template<class T_Field>
    class PrintField : public ISimulationPlugin
    {
        using Field = T_Field;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        GatherSlice<typename Field::Buffer::DataBoxType::ValueType> gather;

        bool isMaster;
        std::string name;
        std::string prefix;

        uint32_t notifyFrequency;
        std::string fileName;
        float_X slicePoint;

    public:
        PrintField():
            isMaster(false),
            name("PrintField: Outputs a slice of a field to a PNG"),
            prefix(Field::getName() + std::string("_printSlice")),
            notifyFrequency(0),
            slicePoint(0)
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PrintField()
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((prefix + ".period").c_str(), po::value<uint32_t>(&notifyFrequency), "enable analyzer [for each n-th step]")
                ((prefix + ".fileName").c_str(), po::value<std::string>(&this->fileName)->default_value("xrt"), "base file name to store slices in (_step.png will be appended)")
                ((prefix + ".slicePoint").c_str(), po::value<float_X>(&this->slicePoint)->default_value(0), "slice point 0.0 <= x <= 1.0")
                ;
        }

        std::string pluginGetName() const override
        {
            return name;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Outputting field at timestep %1%") % currentStep;
            PMacc::DataConnector &dc = Environment::get().DataConnector();

            Field& field = dc.getData<Field>(Field::getName());
            field.synchronize();
            /* gather::operator() gathers all the buffers and assembles those to
             * a complete picture discarding the guards.
             */
            Space size = Environment::get().SubGrid().getGlobalDomain().size;
            uint32_t offset;
            if(simDim == 2)
                offset = 0;
            else
            {
                offset= slicePoint * size[2];
                if(offset >= size[2])
                    offset = size[2] - 1;
            }
            auto picture = gather(field->getHostDataBox(), offset);
            if (isMaster){
                PngCreator png;
                std::stringstream fileName;
                fileName << this->fileName
                         << std::setw(6) << std::setfill('0') << currentStep
                         << ".png";

                png(fileName.str(), picture, size);
            }

            dc.releaseData(Field::getName());
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
        {}
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {}

    protected:
        void pluginLoad() override
        {
            if(slicePoint < 0 || slicePoint > 1)
            {
                std::cerr << "In " << name << " the slicePoint is outside of [0, 1]. Ignored!" << std::endl;
                return;
            }
            Environment::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            auto& subGrid = Environment::get().SubGrid();
            isMaster = gather.init(
                    MessageHeader(subGrid.getGlobalDomain().size, cellDescription_->getGridLayout(), subGrid.getLocalDomain().offset),
                    true);
        }
    };

}  // namespace plugins
}  // namespace xrt
