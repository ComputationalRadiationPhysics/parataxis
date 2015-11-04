#pragma once

#include "random/RNGProvider.hpp"
#include <mpi/SeedPerRank.hpp>
#include <dimensions/DataSpaceOperations.hpp>
#include <memory/boxes/CachedBox.hpp>
#include <nvidia/functors/Assign.hpp>
#include <mappings/threads/ThreadCollective.hpp>

namespace xrt {
namespace random {

    namespace kernel {

        template<class T_RNGMethod, class T_RNGBox, class T_Mapper>
        __global__ void
        initRNGProvider(T_RNGBox rngBox, uint32_t seed, const T_Mapper mapper)
        {
            const Space superCellIdx = mapper.getSuperCellIndex(Space(blockIdx));

            /* get local cell idx (w/o guards) */
            const Space blockCellIdx = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT();
            const Space localCellIdx = blockCellIdx + Space(threadIdx);
            const uint32_t cellIdx = PMacc::DataSpaceOperations<simDim>::map(mapper.getGridSuperCells() * SuperCellSize::toRT(), localCellIdx);

            using BlockBoxSize =  PMacc::SuperCellDescription<SuperCellSize>;
            auto cachedRNGBox = PMacc::CachedBox::create<0, typename T_RNGBox::ValueType>(BlockBoxSize());

            T_RNGMethod().init(cachedRNGBox(Space(threadIdx)).getState(), seed, cellIdx);
            __syncthreads();
            const uint32_t linearThreadIdx = PMacc::DataSpaceOperations<simDim>::map<SuperCellSize>(Space(threadIdx));
            PMacc::ThreadCollective<BlockBoxSize> collective(linearThreadIdx);
            auto shiftedRNGBox = rngBox.shift(blockCellIdx);
            PMacc::nvidia::functors::Assign assign;
            collective(
                      assign,
                      shiftedRNGBox,
                      cachedRNGBox
                      );
        }

        template<class T_Functor, class T_Mapper>
        __global__ void
        testRNGProvider(T_Functor random, const T_Mapper mapper)
        {
            const Space superCellIdx = mapper.getSuperCellIndex(Space(blockIdx));

            /* get local cell idx (w/o guards) */
            const Space localCellIdx = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT() + Space(threadIdx);
            const uint32_t cellIdx = PMacc::DataSpaceOperations<simDim>::map(mapper.getGridSuperCells() * SuperCellSize::toRT(), localCellIdx);

            random.init(localCellIdx);
            printf("%u: %g\n", cellIdx, random());
        }

    }  // namespace kernel

    RNGProvider::RNGProvider(const MappingDesc& desc):
    		cellDescription(desc),
    		buffer(new Buffer(cellDescription.getGridLayout().getDataSpaceWithoutGuarding()))
    {}

    void RNGProvider::init(uint32_t seed)
    {
        PMacc::mpi::SeedPerRank<simDim> seedPerRank;
        seeds::Global globalSeed;
        seed ^= globalSeed();
        seed = seedPerRank(seed);

        Space blockSize = SuperCellSize::toRT();
#if XRT_USE_MAPPED_RNG_BUFFER
        auto bufferBox = buffer->getDataBox();
#else
        auto bufferBox = buffer->getDeviceBuffer().getDataBox();
#endif
        __cudaKernelArea( kernel::initRNGProvider<RNGMethod>, this->cellDescription, PMacc::CORE + PMacc::BORDER )
        (blockSize)
        ( bufferBox,
          seed
          );

        Environment::get().DataConnector().registerData(*this);

#ifdef XRT_DEBUG_RNG
        Random<> random;
        __cudaKernelArea( kernel::testRNGProvider, this->cellDescription, PMacc::CORE + PMacc::BORDER )
        (blockSize)
        (random);
#endif
    }

    RNGProvider::DataBoxType
    RNGProvider::getDeviceDataBox()
    {
#if XRT_USE_MAPPED_RNG_BUFFER
        return buffer->getDataBox();
#else
        return buffer->getDeviceBuffer().getDataBox();
#endif
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
#if XRT_USE_MAPPED_RNG_BUFFER
        throw std::runtime_error("Calling synchronize on a mapped buffer is not possible!");
#else
        buffer->deviceToHost();
#endif
    }

}  // namespace random
}  // namespace xrt
