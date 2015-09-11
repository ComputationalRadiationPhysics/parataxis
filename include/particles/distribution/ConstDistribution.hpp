#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace distribution {

    /**
     * Functor that returns a constant value as the distribution for any param
     */
    template<class T_NumParts>
    struct ConstDistribution
    {
        static constexpr float_X numParts = T_NumParts::value;

        ConstDistribution(Space2D totalSize){}

        DINLINE float_X
        operator()(Space2D totalCellIdx) const
        {
            return numParts;
        }
    };

}  // namespace distribution
}  // namespace particles
}  // namespace xrt
