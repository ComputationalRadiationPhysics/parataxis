#pragma once

#include "xrtTypes.hpp"

#include <cuSTL/algorithm/mpi/Gather.hpp>
#include <cuSTL/container/HostBuffer.hpp>
#include <cuSTL/cursor/tools/slice.hpp>
#include <cuSTL/algorithm/kernel/Foreach.hpp>

#include <memory>

namespace xrt
{
    template<class T_Field, uint32_t T_simDim>
    struct GatherSlice;

    template<class T_Field>
    struct GatherSlice<T_Field, 2>
    {
        using Field = T_Field;
        using Gather = PMacc::algorithm::mpi::Gather<2>;
        using HostBuffer = PMacc::container::HostBuffer<typename Field::Type, 2>;

        GatherSlice(uint32_t slicePoint, uint32_t nAxis)
        {
            auto& env = PMacc::Environment<2>::get();
            PMacc::zone::SphericZone<2> gpuGatheringZone(env.GridController().getGpuNodes());
            gather_.reset(new Gather(gpuGatheringZone));
            if(gather_->root())
                masterField_.reset(new HostBuffer(env.SubGrid().getTotalDomain().size));
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

        GatherSlice(uint32_t slicePlane, uint32_t nAxis)
        {
            auto& env = PMacc::Environment<3>::get();
            Space3D globalSize = env.SubGrid().getTotalDomain().size;
            Space3D localSize = env.SubGrid().getLocalDomain().size;
            /* GPU idx (in the axis dimension) that has the slice */
            int localPlane = slicePlane / localSize[nAxis];

            PMacc::log< XRTLogLvl::IN_OUT >("Init gather slice at point %1% of axis %2% with size %3%/%4%")
                    % slicePlane % nAxis % localSize % globalSize;

            PMacc::zone::SphericZone<3> gpuGatheringZone(env.GridController().getGpuNodes());
            /* Use only 1 GPU in the axis dimension */
            gpuGatheringZone.offset[nAxis] = localPlane;
            gpuGatheringZone.size  [nAxis] = 1;

            gather_.reset(new Gather(gpuGatheringZone));
            if(!gather_->participate())
                return;
            /* Offset in the local field (if we have the slice) */
            localOffset_  = slicePlane % localSize[nAxis];
            twistedAxes_ = PMacc::math::UInt32<3>((nAxis + 1) % 3, (nAxis + 2) % 3, nAxis);

            /* Reduce size dimension */
            Space2D tmpSize(localSize[twistedAxes_[0]], localSize[twistedAxes_[1]]);
            Space2D masterSize(globalSize[twistedAxes_[0]], globalSize[twistedAxes_[1]]);

            PMacc::log< XRTLogLvl::IN_OUT >("Participation in gather operation. Local offset: %1%. Rank: %2%. Is root: %3%")
                    % localOffset_ % gather_->rank() % gather_->root();

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
        uint32_t slicePoint_;
        PMacc::math::UInt32<3> twistedAxes_;
        uint32_t localOffset_;
        std::unique_ptr<Gather> gather_;
        std::unique_ptr<HostBuffer> masterField_;
        std::unique_ptr<TmpBuffer> tmpBuffer_;
    };

} //namespace xrt


