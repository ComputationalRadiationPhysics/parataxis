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
#include <memory>

namespace xrt {
namespace plugins {

    template<class T_Field>
    class PrintField : public ISimulationPlugin
    {
        using Field = T_Field;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        GatherSlice<Field, simDim> gather_;

        bool isMaster;
        std::string name;
        std::string prefix;

        uint32_t notifyFrequency;
        std::string fileName;
        float_X slicePoint;
        uint32_t nAxis_;

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
                ((prefix + ".fileName").c_str(), po::value<std::string>(&this->fileName)->default_value("field"), "base file name to store slices in (_step.png will be appended)")
                ((prefix + ".slicePoint").c_str(), po::value<float_X>(&this->slicePoint)->default_value(0), "slice point 0.0 <= x <= 1.0")
                ((prefix + ".axis").c_str(), po::value<uint32_t>(&this->nAxis_)->default_value(2), "Axis index to slice through (0=>x, 1=>y, 2=>z)")
                ;
        }

        std::string pluginGetName() const override
        {
            return name;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Outputting field at timestep %1%") % currentStep;

            auto &dc = Environment::get().DataConnector();

            Field& field = dc.getData<Field>(Field::getName(), false);
            gather_(field);
            if (gather_.hasData()){
                PngCreator png;
                std::stringstream fileName;
                fileName << this->fileName
                         << "_" << std::setw(6) << std::setfill('0') << currentStep
                         << ".png";

                using Box = PMacc::PitchedBox<typename Field::Type, 2>;
                PMacc::DataBox<Box> data(Box(
                        gather_.getData().getDataPointer(),
                        Space2D(),
                        Space2D(gather_.getData().size()),
                        gather_.getData().size().x() * sizeof(typename Field::Type)
                        ));
                png(fileName.str(), data, gather_.getData().size());
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
            gather_.init(slicePoint, nAxis_, Environment::get().SubGrid().getLocalDomain().size);
        }

     };

}  // namespace plugins
}  // namespace xrt
