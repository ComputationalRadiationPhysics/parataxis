#pragma once

#include "RNGProvider.hpp"
#include <mpi/SeedPerRank.hpp>
#include <dimensions/DataSpaceOperations.hpp>
#include <memory/boxes/CachedBox.hpp>
#include <nvidia/functors/Assign.hpp>
#include <mappings/threads/ThreadCollective.hpp>

namespace xrt {

    namespace kernel {

        template<class T_RNGBox, class T_Mapper>
        __global__ void
        initRNGProvider(T_RNGBox rngBox, Space localSize, uint32_t seed, const T_Mapper mapper)
        {
            const Space superCellIdx = mapper.getSuperCellIndex(Space(blockIdx));

            /* get local cell idx (w/o guards) */
            const Space localBlockOffset = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT();
            const Space localCellIdx = localBlockOffset + Space(threadIdx);
            const uint32_t cellIdx = PMacc::DataSpaceOperations<simDim>::map(localSize, localCellIdx);

            using BlockBoxSize =  PMacc::SuperCellDescription<SuperCellSize>;
            auto cachedRNGBox = PMacc::CachedBox::create<0, typename T_RNGBox::ValueType>(BlockBoxSize());

            cachedRNGBox(Space(threadIdx)) = nvrng::methods::Xor(seed, cellIdx);
            __syncthreads();
            const uint32_t linearThreadIdx = PMacc::DataSpaceOperations<simDim>::map<SuperCellSize>(Space(threadIdx));
            PMacc::ThreadCollective<BlockBoxSize> collective(linearThreadIdx);
            auto shiftedRNGBox = rngBox.shift(localBlockOffset);
            PMacc::nvidia::functors::Assign assign;
            collective(
                      assign,
                      cachedRNGBox,
                      shiftedRNGBox
                      );
        }

    }  // namespace kernel

    RNGProvider::RNGProvider(const MappingDesc& desc): cellDescription(desc), buffer(new Buffer(cellDescription.getGridSuperCells()))
    {}

    void RNGProvider::init(uint32_t seed)
    {
        PMacc::mpi::SeedPerRank<simDim> seedPerRank;
        seeds::Global globalSeed;
        seed ^= globalSeed();
        seed = seedPerRank(seed);

        Space block = SuperCellSize::toRT();
        __cudaKernelArea( kernel::initRNGProvider, this->cellDescription, PMacc::CORE + PMacc::BORDER )
        (block)
        ( buffer->getDeviceBuffer().getDataBox(),
          Environment::get().SubGrid().getLocalDomain().size,
          seed
          );

        Environment::get().DataConnector().registerData(*this);
    }

    RNGProvider::DataBoxType
    RNGProvider::getDeviceDataBox()
    {
        return buffer->getDeviceBuffer().getDataBox();
    }

    std::string
    RNGProvider::getName()
    {
        return "RNGProvider";
    }

    PMacc::SimulationDataId
    RNGProvider::getUniqueId()
    {
        return getName();
    }

    void RNGProvider::synchronize()
    {
        buffer->deviceToHost();
    }

}  // namespace xrt
