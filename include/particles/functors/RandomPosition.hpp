#pragma once

#include "xrtTypes.hpp"
#include "Random.hpp"

namespace xrt {
namespace particles {
namespace functors {

    template<class T_Species>
    struct RandomPosition
    {
        using Species = T_Species;
        using Random = xrt::Random<Species>;


        HINLINE RandomPosition(uint32_t currentStep): rand(currentStep)
        {}

        HINLINE void
        init(Space totalCellIdx, uint32_t totalNumParts)
        {
            rand.init(totalCellIdx);
        }

        DINLINE floatD_X
        operator()(uint32_t numPart)
        {
            floatD_X result;
            for(unsigned i = 0; i < simDim; ++i)
                result[i] = rand();
            return result;
        }
    private:
        Random rand;
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
