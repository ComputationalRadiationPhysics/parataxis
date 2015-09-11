#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace distribution {

    /**
     * Functor that returns a constant value as the distribution for every time given
     */
    template<class T_NumParts>
    struct ConstDistributionTime
    {
        static constexpr float_X numParts = T_NumParts::value;

        ConstDistributionTime(float_X pulseLength){}

        DINLINE float_X
        operator()(float_X pulseTime) const
        {
            return numParts;
        }
    };

}  // namespace distribution
}  // namespace particles
}  // namespace xrt
