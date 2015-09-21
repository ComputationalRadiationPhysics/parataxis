#pragma once

#include "xrtTypes.hpp"
#include "ConditionalShrink.hpp"
#include <mpi/SeedPerRank.hpp>
#include <traits/GetUniqueTypeId.hpp>
#include <dimensions/DataSpaceOperations.hpp>
#include <nvidia/rng/RNG.hpp>
#include <nvidia/rng/methods/Xor.hpp>
#include <nvidia/rng/distributions/Uniform_float.hpp>


namespace xrt{

    /**
     * Wrapper around a uniform random number generator
     * For a given species, step and cell this functor returns a random number
     * in the range [0, 1) that is uniformly distributed
     */
    template<typename T_SpeciesType, int32_t T_shrinkDim = -1>
    struct Random
    {
        typedef T_SpeciesType SpeciesType;
        static constexpr int32_t shrinkDim = T_shrinkDim;
        static constexpr uint32_t dim = (shrinkDim >= 0) ? simDim - 1 : simDim;

        HINLINE Random(uint32_t currentStep, uint32_t seed)
        {
            typedef typename SpeciesType::FrameType FrameType;

            PMacc::mpi::SeedPerRank<simDim> seedPerRank;
            seeds::Global globalSeed;
            seed ^= globalSeed() ^ PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid();
            seed = seedPerRank(seed);
            seed ^= currentStep;

            const SubGrid& subGrid = Environment::get().SubGrid();
            ConditionalShrink<shrinkDim> shrink;
            localCells = shrink(subGrid.getLocalDomain().size);
            totalGpuOffset = shrink(subGrid.getLocalDomain().offset);
        }

        DINLINE void init(const PMacc::DataSpace<dim>& totalCellOffset)
        {
            const PMacc::DataSpace<dim> localCellIdx(totalCellOffset - totalGpuOffset);
            const uint32_t cellIdx = PMacc::DataSpaceOperations<dim>::map(localCells, localCellIdx);
            rng = nvrng::create(nvrng::methods::Xor(seed, cellIdx), nvrng::distributions::Uniform_float());
        }

        /** Returns a uniformly distributed value between [0, 1)
         *
         * @return float_X with value between [0.0, 1.0)
         */
        DINLINE float_X operator()()
        {
            return rng();
        }

    protected:
        typedef nvrng::RNG<nvrng::methods::Xor, nvrng::distributions::Uniform_float> RngType;

        PMACC_ALIGN(rng, RngType);
        PMACC_ALIGN(seed, uint32_t);
        PMACC_ALIGN(localCells, PMacc::DataSpace<dim>);
        PMACC_ALIGN(totalGpuOffset, PMacc::DataSpace<dim>);
    };

} //namespace xrt
