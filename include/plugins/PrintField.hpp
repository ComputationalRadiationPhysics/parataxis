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
        std::vector<UsedGatherSlice*> gather_;

        bool isMaster;
        std::string name;
        std::string prefix;

        uint32_t notifyFrequency;
        std::string format;
        std::string fileName;
        std::vector<uint32_t> slicePoints;
        uint32_t nAxis_;

    public:
        PrintField():
            isMaster(false),
            name("PrintField: Outputs a slice of a field to a PNG or TIFF"),
            prefix(Field::getName() + std::string("_printSlice")),
            notifyFrequency(0),
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
                ((prefix + ".slicePoint").c_str(), po::value<std::vector<uint32_t>>(&this->slicePoints)->multitoken(), "slice point 0 <= x < simSize[axis]")
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
            for(UsedGatherSlice* gather: gather_)
            {
                (*gather)(field);
                if (gather->hasData()){
                    std::stringstream fileName;
                    fileName << this->fileName
                             << "_" << std::setw(6) << std::setfill('0') << gather->slicePoint
                             << "_" << std::setw(6) << std::setfill('0') << currentStep
                             << "." << format;

                    using Box = PMacc::PitchedBox<typename Field::Type, 2>;
                    PMacc::DataBox<Box> data(Box(
                            gather->getData().getDataPointer(),
                            Space2D(),
                            Space2D(gather->getData().size()),
                            gather->getData().size().x() * sizeof(typename Field::Type)
                            ));
                    if(format == "png")
                    {
#if XRT_ENABLE_PNG
                        imaging::PngCreator img;
                        img(fileName.str(), data, gather->getData().size());
#endif
                    }else
                    {
#if XRT_ENABLE_TIFF
                        imaging::TiffCreator img;
                        img(fileName.str(), data, gather->getData().size());
#endif
                    }
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
            for(auto it = slicePoints.begin(); it != slicePoints.end();)
            {
                if(*it >= Environment::get().SubGrid().getGlobalDomain().size[nAxis_])
                {
                    std::cerr << "In " << name << " the slicePoint " << *it << " is bigger than the simulation size. Ignored!" << std::endl;
                    it = slicePoints.erase(it);
                }else
                    ++it;
            }
            if(slicePoints.empty())
                return;
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
            // TODO: Ineffective. Improve!
            for(uint32_t slicePoint: slicePoints)
                gather_.push_back(new UsedGatherSlice(slicePoint, nAxis_));
#else
            PMacc::log<XRTLogLvl::PLUGINS>("Did not found tiff or png library. %1% is disabled") % getName();
#endif
        }

        void pluginUnload() override
        {
            for(UsedGatherSlice* gather: gather_)
                delete gather;
            gather_.clear();
        }

     };

}  // namespace plugins
}  // namespace xrt
