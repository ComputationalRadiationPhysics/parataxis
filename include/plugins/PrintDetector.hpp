#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"
#include "ComplexTraits.hpp"
#include "math/abs.hpp"
#include "plugins/imaging/TiffCreator.hpp"
#include "TransformBox.hpp"

#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <mpi/MPIReduce.hpp>
#include <mpi/reduceMethods/Reduce.hpp>
#include <nvidia/functors/Add.hpp>
#include <memory/buffers/HostBufferIntern.hpp>
#include <string>
#include <sstream>

namespace xrt {
namespace plugins {

    template<class T_Detector>
    class PrintDetector : public ISimulationPlugin
    {
        using Detector = T_Detector;

        std::string name;
        std::string prefix;

        uint32_t notifyFrequency;
        std::string fileName;

        PMacc::mpi::MPIReduce reduce_;
        using ReduceMethod = PMacc::mpi::reduceMethods::Reduce;

        using Type = typename Detector::Type;
        std::unique_ptr< PMacc::HostBuffer<Type, 2> > masterBuffer_;

    public:
        PrintDetector():
            name("PrintDetector: Prints the detector to a TIFF"),
            prefix(Detector::getName() + std::string("_print")),
            notifyFrequency(0)
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((prefix + ".period").c_str(), po::value<uint32_t>(&notifyFrequency), "enable analyzer [for each n-th step]")
                ((prefix + ".fileName").c_str(), po::value<std::string>(&this->fileName)->default_value("detector"), "base file name (_step.tif will be appended)")
                ;
        }

        std::string pluginGetName() const override
        {
            return name;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Outputting detector at timestep %1%") % currentStep;
            PMacc::DataConnector& dc = Environment::get().DataConnector();

            Detector& detector = dc.getData<Detector>(Detector::getName());

            bool isMaster = reduce_.hasResult(ReduceMethod());
            Space2D size = detector.getSize();
            if(isMaster && !masterBuffer_)
                masterBuffer_.reset(new PMacc::HostBufferIntern<Type, 2>(size));
            reduce_(PMacc::nvidia::functors::Add(),
                   isMaster ? masterBuffer_->getDataBox().getPointer() : nullptr,
                   detector.getHostDataBox().getPointer(),
                   size.productOfComponents(),
                   ReduceMethod()
                   );

            if (isMaster){
                std::stringstream fileName;
                fileName << this->fileName
                         << "_" << std::setw(6) << std::setfill('0') << currentStep
                         << ".tif";

                imaging::TiffCreator tiff;
                tiff(fileName.str(), makeTransformBox(masterBuffer_->getDataBox(), typename Detector::OutputTransformer()), size);
            }

            dc.releaseData(Detector::getName());
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
        void pluginUnload() override
        {
            masterBuffer_.reset();
        }
    };

}  // namespace plugins
}  // namespace xrt
