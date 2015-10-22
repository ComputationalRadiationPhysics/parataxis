#pragma once

#include "xrtTypes.hpp"
#include "random/Random.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    template<class T_Species>
    struct RandomPosition
    {
        using Random = random::Random<simDim == 3 ? laserConfig::DIRECTION : -1>;

        HINLINE RandomPosition(uint32_t currentStep)
        {}

        DINLINE void
        init(Space2D totalCellIdx)
        {
            rand.init(totalCellIdx);
        }

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE floatD_X
        operator()(uint32_t numPart)
        {
            floatD_X result;
            for(uint32_t i = 0; i < simDim; ++i)
                result[i] = rand() * laserConfig::distSize[i] / cellSize[i];
            return result;
        }
    private:
        Random rand;
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
