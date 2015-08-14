#pragma once

#include "xrtTypes.hpp"
#include <algorithms/math.hpp>
#include <cmath>

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Distributes particles evenly in a cell
     * Assumes cubic cells (same length on all sides)
     */
    template<class T_Species>
    struct EvenDistPosition
    {
        HINLINE EvenDistPosition(uint32_t currentStep)
        {}

        DINLINE void
        init(Space totalCellIdx, uint32_t totalNumParts)
        {
            using PMacc::algorithms::math::pow;
            partsPerDim = static_cast<uint32_t>(std::ceil(pow(static_cast<float_X>(totalNumParts), static_cast<float_X>(1.) / simDim)));
            invPartsPerDim = static_cast<float_X>(1) / static_cast<float_X>(partsPerDim);
        }

        DINLINE floatD_X
        operator()(uint32_t numPart)
        {
            floatD_X result;
            for(unsigned i = 0; i < simDim; ++i){
                uint32_t remaining = numPart / partsPerDim;
                uint32_t dimParts  = numPart - remaining;
                result[i] = dimParts * invPartsPerDim;
                numPart = remaining;
            }
            return result;
        }
    private:
        PMACC_ALIGN(partsPerDim, uint32_t);
        PMACC_ALIGN(invPartsPerDim, float_X);
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
