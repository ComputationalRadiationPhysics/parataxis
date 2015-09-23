#pragma once

#include "xrtTypes.hpp"
#include "ConditionalShrink.hpp"
#include "RNGProvider.hpp"
#include <dimensions/DataSpaceOperations.hpp>
#include <nvidia/rng/RNG.hpp>
#include <nvidia/rng/distributions/Uniform_float.hpp>


namespace xrt{

    /**
     * Wrapper around a uniform random number generator
     * For a given species, step and cell this functor returns a random number
     * in the range [0, 1) that is uniformly distributed
     */
    template<int32_t T_shrinkDim = -1>
    struct Random
    {
        static constexpr int32_t shrinkDim = T_shrinkDim;
        static constexpr uint32_t dim = (shrinkDim >= 0) ? simDim - 1 : simDim;
        using RNGBox = RNGProvider::DataBoxType;
        using RNGPtr = RNGProvider::RNGMethod::StatePtr;
        using Distribution = nvrng::distributions::Uniform_float;

        HINLINE Random()
        {
            auto& dc = Environment::get().DataConnector();
            auto& provider = dc.getData<RNGProvider>(RNGProvider::getName(), true);

            rngBox = provider.getDeviceDataBox();

            const SubGrid& subGrid = Environment::get().SubGrid();
            ConditionalShrink<shrinkDim> shrink;
            localCells = shrink(subGrid.getLocalDomain().size);
            totalGpuOffset = shrink(subGrid.getLocalDomain().offset);

            dc.releaseData(RNGProvider::getName());
        }

        DINLINE void init(const PMacc::DataSpace<dim>& totalCellOffset)
        {
            const PMacc::DataSpace<dim> localCellIdx(totalCellOffset - totalGpuOffset);
            Space idx;
            for(int i=0; i<dim; i++)
                idx[i] = localCellIdx[i];
            rngPtr = rngBox(idx).getStatePtr();
        }

        /** Returns a uniformly distributed value between [0, 1)
         *
         * @return float_X with value between [0.0, 1.0)
         */
        DINLINE float_X operator()()
        {
            return Distribution()(rngPtr);
        }

    protected:
        typedef nvrng::RNG<nvrng::methods::Xor, nvrng::distributions::Uniform_float> RngType;

        PMACC_ALIGN(rngBox, RNGBox);
        PMACC_ALIGN(rngPtr, RNGPtr);
        PMACC_ALIGN(localCells, PMacc::DataSpace<dim>);
        PMACC_ALIGN(totalGpuOffset, PMacc::DataSpace<dim>);
    };

} //namespace xrt
