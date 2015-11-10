#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Functor that returns a constant value as the distribution for any param
     */
    template<int32_t T_numParts>
    struct ConstDistribution
    {
        static constexpr int32_t numParts = T_numParts;

        ConstDistribution(Space totalSize){}

        DINLINE void
        init(Space totalCellIdx) const
        {}

        DINLINE int32_t
        operator()(uint32_t timeStep) const
        {
            return numParts;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
