#pragma once

#include "xrtTypes.hpp"
#include "PngCreator.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <cuSTL/algorithm/mpi/Gather.hpp>
#include <cuSTL/container/HostBuffer.hpp>
#include <cuSTL/cursor/tools/slice.hpp>
#include <cuSTL/algorithm/kernel/Foreach.hpp>
#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <string>
#include <sstream>
#include <memory>

namespace xrt {
namespace plugins {

    namespace detail {
        template<class T_Field, uint32_t T_simDim>
        struct GatherSlice;

        template<class T_Field>
        struct GatherSlice<T_Field, 2>
        {
            using Field = T_Field;
            using Gather = PMacc::algorithm::mpi::Gather<2>;
            using HostBuffer = PMacc::container::HostBuffer<typename Field::Type, 2>;

            void
            init(float_X slicePoint, uint32_t nAxis, Space2D fieldSize)
            {
                auto& gc = PMacc::Environment<2>::get().GridController();
                Space2D gpuDim = gc.getGpuNodes();
                Space2D globalSize = gpuDim * fieldSize;
                PMacc::zone::SphericZone<2> gpuGatheringZone(gpuDim);
                gather_.reset(new Gather(gpuGatheringZone));
                if(gather_->root())
                    masterField_.reset(new HostBuffer(globalSize));
                else
                    masterField_.reset(new HostBuffer(PMacc::math::Size_t<2>::create(0)));
            }

            void
            operator()(Field& field)
            {
                if(!gather_->participate())
                    return;
                field.synchronize();
                (*gather_)(
                        *masterField_,
                        field.getHostBuffer().cartBuffer().view(SuperCellSize::toRT(), -SuperCellSize::toRT())
                        );
            }

            bool
            hasData() const
            {
                return gather_->root();
            }

            HostBuffer&
            getData()
            {
                return *masterField_;
            }
        private:
            std::unique_ptr<Gather> gather_;
            std::unique_ptr<HostBuffer> masterField_;
        };

        struct ConversionFunctor
        {
            DINLINE void operator()(float_X& target, const float_X fieldData) const
            {
                target = fieldData;
            }
        };

        template<class T_Field>
        struct GatherSlice<T_Field, 3>
        {
            using Field = T_Field;
            using Gather = PMacc::algorithm::mpi::Gather<3>;
            using HostBuffer = PMacc::container::HostBuffer<typename Field::Type, 2>;
            using TmpBuffer = PMacc::GridBuffer<typename Field::Type, 2>;

            void
            init(float_X slicePoint, uint32_t nAxis, Space3D fieldSize)
            {
                auto& gc = PMacc::Environment<3>::get().GridController();
                Space3D gpuDim = gc.getGpuNodes();
                /* Global size of the field */
                Space3D globalSize = gpuDim * fieldSize;
                /* plane (idx in field array) in global field */
                int globalPlane = globalSize[nAxis] * slicePoint;
                /* GPU idx (in the axis dimension) that has the slice */
                int gpuPlane    = globalPlane / fieldSize[nAxis];

                PMacc::log< XRTLogLvl::IN_OUT >("Init gather slice at point %1% of axis %2% with size %3%/%4%") % globalPlane % nAxis % fieldSize % globalSize;

                PMacc::zone::SphericZone<3> gpuGatheringZone(gpuDim);
                /* Use only 1 GPU in the axis dimension */
                gpuGatheringZone.offset[nAxis] = gpuPlane;
                gpuGatheringZone.size  [nAxis] = 1;

                gather_.reset(new Gather(gpuGatheringZone));
                if(!gather_->participate())
                    return;
                /* Offset in the local field (if we have the slice) */
                localOffset_  = globalPlane % fieldSize[nAxis];
                twistedAxes_ = PMacc::math::UInt32<3>((nAxis + 1) % 3, (nAxis + 2) % 3, (nAxis + 3) % 3);

                /* Reduce size dimension */
                Space2D tmpSize(fieldSize[twistedAxes_[0]], fieldSize[twistedAxes_[1]]);
                Space2D masterSize(globalSize[twistedAxes_[0]], globalSize[twistedAxes_[1]]);

                tmpBuffer_.reset(new TmpBuffer(tmpSize));
                if(gather_->root())
                    masterField_.reset(new HostBuffer(masterSize));
                else
                    masterField_.reset(new HostBuffer(PMacc::math::Size_t<2>::create(0)));

            }

            void
            operator()(Field& field)
            {
                auto dBufferTmp(tmpBuffer_->getDeviceBuffer().cartBuffer());
                auto dBuffer(field.getGridBuffer().getDeviceBuffer().cartBuffer().view(SuperCellSize::toRT(), -SuperCellSize::toRT()));
                ConversionFunctor cf;
                PMacc::algorithm::kernel::Foreach<PMacc::math::CT::UInt32<4,4,1> >()(
                             dBufferTmp.zone(), dBufferTmp.origin(),
                             PMacc::cursor::tools::slice(dBuffer.originCustomAxes(twistedAxes_)(0, 0, localOffset_)),
                             cf );
                tmpBuffer_->deviceToHost();
                auto hBufferTmp(tmpBuffer_->getHostBuffer().cartBuffer());
                (*gather_)(*masterField_, hBufferTmp);
            }

            bool
            hasData() const
            {
                return gather_->root();
            }

            HostBuffer&
            getData()
            {
                return *masterField_;
            }
        private:
            float_X slicePoint_;
            PMacc::math::UInt32<3> twistedAxes_;
            uint32_t localOffset_;
            std::unique_ptr<Gather> gather_;
            std::unique_ptr<HostBuffer> masterField_;
            std::unique_ptr<TmpBuffer> tmpBuffer_;
        };

    }  // namespace detail

    template<class T_Field>
    class PrintField : public ISimulationPlugin
    {
        using Field = T_Field;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        detail::GatherSlice<Field, simDim> gather_;

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
