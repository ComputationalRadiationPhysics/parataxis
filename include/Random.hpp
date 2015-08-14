#pragma once

#include "xrtTypes.hpp"
#include <mpi/SeedPerRank.hpp>
#include <traits/GetUniqueTypeId.hpp>
#include <dimensions/DataSpaceOperations.hpp>
#include <nvidia/rng/RNG.hpp>
#include <nvidia/rng/methods/Xor.hpp>
#include <nvidia/rng/distributions/Uniform_float.hpp>


namespace xrt{

    template<typename T_SpeciesType>
    struct Random
    {
        namespace nvrng = PMacc::nvidia::rng;
        namespace rngMethods = nvrng::methods;
        namespace rngDistributions = nvrng::distributions;

        typedef T_SpeciesType SpeciesType;

        HINLINE Random(uint32_t currentStep)
        {
            typedef typename SpeciesType::FrameType FrameType;

            PMacc::mpi::SeedPerRank<simDim> seedPerRank;
            seed = seedPerRank(seeds::Global()(), PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid());
            seed ^= seeds::position;
            seed ^= currentStep;

            const SubGrid& subGrid = Environment::get().SubGrid();
            localCells = subGrid.getLocalDomain().size;
            totalGpuOffset = subGrid.getLocalDomain().offset;
        }

        DINLINE void init(const Space& totalCellOffset)
        {
            const Space localCellIdx(totalCellOffset - totalGpuOffset);
            const uint32_t cellIdx = PMacc::DataSpaceOperations<simDim>::map(localCells, localCellIdx);
            rng = nvrng::create(rngMethods::Xor(seed, cellIdx), rngDistributions::Uniform_float());
        }

        /** Returns a uniformly distributed value between [0, 1)
         *
         * @return float_X with value between [0.0, 1.0)
         */
        DINLINE float_X operator()(const uint32_t)
        {
            return rng;
        }

    protected:
        typedef nvrng::RNG<rngMethods::Xor, rngDistributions::Uniform_float> RngType;

        PMACC_ALIGN(rng, RngType);
        PMACC_ALIGN(seed, uint32_t);
        PMACC_ALIGN(localCells, Space);
        PMACC_ALIGN(totalGpuOffset, Space);
    };

} //namespace xrt
