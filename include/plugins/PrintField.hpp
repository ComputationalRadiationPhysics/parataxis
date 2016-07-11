#pragma once

#include "xrtTypes.hpp"
#include "GatherSlice.hpp"
#if XRT_ENABLE_PNG
#   include "plugins/imaging/PngCreator.hpp"
#endif
#if XRT_ENABLE_TIFF
#   include "plugins/imaging/TiffCreator.hpp"
#endif
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
        using UsedGatherSlice = GatherSlice<Field, simDim>;
        std::unique_ptr<UsedGatherSlice> gather_;

        bool isMaster;
        std::string name;
        std::string prefix;

        uint32_t notifyFrequency;
        std::string format;
        std::string fileName;
        uint32_t slicePoint;
        uint32_t nAxis_;

    public:
        PrintField():
            isMaster(false),
            name("PrintField: Outputs a slice of a field to a PNG or TIFF"),
            prefix(Field::getName() + std::string("_printSlice")),
            notifyFrequency(0),
            slicePoint(0),
            nAxis_(0)
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PrintField()
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((prefix + ".period").c_str(), po::value<uint32_t>(&notifyFrequency), "enable analyzer [for each n-th step]")
                ((prefix + ".format").c_str(), po::value<std::string>(&this->format), "file format (png or tiff)")
                ((prefix + ".fileName").c_str(), po::value<std::string>(&this->fileName)->default_value("field"), "base file name to store slices in (_step.png will be appended)")
                ((prefix + ".slicePoint").c_str(), po::value<uint32_t>(&this->slicePoint)->default_value(40), "slice point 0 <= x < simSize[axis]")
                ((prefix + ".axis").c_str(), po::value<uint32_t>(&this->nAxis_)->default_value(0), "Axis index to slice through (0=>x, 1=>y, 2=>z)")
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

            Field& field = dc.getData<Field>(Field::getName());
            (*gather_)(field);
            if (gather_->hasData()){
                std::stringstream fileName;
                fileName << this->fileName
                         << "_" << std::setw(6) << std::setfill('0') << currentStep
                         << "." << format;

                using Box = PMacc::PitchedBox<typename Field::Type, 2>;
                PMacc::DataBox<Box> data(Box(
                        gather_->getData().getDataPointer(),
                        Space2D(),
                        Space2D(gather_->getData().size()),
                        gather_->getData().size().x() * sizeof(typename Field::Type)
                        ));
                if(format == "png")
                {
#if XRT_ENABLE_PNG
                    imaging::PngCreator img;
                    img(fileName.str(), data, gather_->getData().size());
#endif
                }else
                {
#if XRT_ENABLE_TIFF
                    imaging::TiffCreator img;
                    img(fileName.str(), data, gather_->getData().size());
#endif
                }
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
            if(nAxis_ >= simDim)
            {
                std::cerr << "In " << name << " the axis is invalid. Ignored!" << std::endl;
                return;
            }
            if(slicePoint >= Environment::get().SubGrid().getGlobalDomain().size[nAxis_])
            {
                std::cerr << "In " << name << " the slicePoint is bigger than the simulation size. Ignored!" << std::endl;
                return;
            }
            std::transform(format.begin(), format.end(), format.begin(), ::tolower);
            if(format != "tiff" && format != "tif")
                format = "png";
#if !XRT_ENABLE_PNG
            format = "tif";
#elif !XRT_ENABLE_TIFF
            format = "png";
#endif
#if XRT_ENABLE_PNG || XRT_ENABLE_TIFF
            Environment::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            gather_.reset(new UsedGatherSlice(slicePoint, nAxis_));
#else
            PMacc::log<XRTLogLvl::PLUGINS>("Did not found tiff or png library. %1% is disabled") % getName();
#endif
        }

        void pluginUnload() override
        {
            gather_.reset();
        }

     };

}  // namespace plugins
}  // namespace xrt
