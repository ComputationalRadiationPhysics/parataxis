#pragma once

#include "xrtTypes.hpp"
#include "ConditionalShrink.hpp"
#include "random/RNGProvider.hpp"
#include "random/distributions/Uniform_float.h"
#include <dimensions/DataSpaceOperations.hpp>

namespace xrt {
namespace random {

    /**
     * Wrapper around a uniform random number generator
     * For a given species, step and cell this functor returns a random number
     * in the range [0, 1) that is uniformly distributed
     */
    template<int32_t T_shrinkDim = -1>
    struct Random: private distributions::Uniform_float<RNGProvider::RNGMethod>
    {
        static constexpr int32_t shrinkDim = T_shrinkDim;
        static constexpr uint32_t dim = (shrinkDim >= 0) ? simDim - 1 : simDim;
        using RNGBox = RNGProvider::DataBoxType;
        using RNGState = RNGProvider::RNGMethod::StateType;
        using Distribution = distributions::Uniform_float<RNGProvider::RNGMethod>;

        HINLINE Random()
        {
            auto& dc = Environment::get().DataConnector();
            auto& provider = dc.getData<RNGProvider>(RNGProvider::getName(), true);

            rngBox = provider.getDeviceDataBox();

            const SubGrid& subGrid = Environment::get().SubGrid();
            ConditionalShrink<shrinkDim> shrink;
            totalGpuOffset = shrink(subGrid.getLocalDomain().offset);

            dc.releaseData(RNGProvider::getName());
        }

        DINLINE void init(const PMacc::DataSpace<dim>& totalCellOffset)
        {
            const PMacc::DataSpace<dim> localCellIdx(totalCellOffset - totalGpuOffset);
            Space idx;
            for(int i=0; i<dim; i++)
                idx[i] = localCellIdx[i];
            rngBox = rngBox.shift(idx);
        }

        /** Returns a uniformly distributed value between [0, 1)
         *
         * @return float_X with value between [0.0, 1.0)
         */
        DINLINE float_X operator()()
        {
            return Distribution::operator()(rngBox(Space::create(0)).getState());
        }

    protected:
        PMACC_ALIGN8(rngBox, RNGBox);
        PMACC_ALIGN8(totalGpuOffset, PMacc::DataSpace<dim>);
    };

}  // namespace random
}  // namespace xrt
