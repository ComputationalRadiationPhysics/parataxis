#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Functor that returns a constant value as the distribution for any param
     * with an x() == 0, and 0 else
     */
    template<uint32_t T_numParts>
    struct ConstDistribution
    {
        static constexpr uint32_t numParts = T_numParts;

        DINLINE uint32_t
        operator()(Space totalGpuCellIdx) const
        {
            return (totalGpuCellIdx.x()) ? 0 : numParts;
        }
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
