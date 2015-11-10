#pragma once

#include "xrtTypes.hpp"
#include "random/Random.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    template<class T_Species>
    struct RandomPosition
    {
        using Distribution = PMacc::random::distributions::Uniform_float<>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;

        HINLINE RandomPosition(uint32_t currentStep): offset(Environment::get().SubGrid().getLocalDomain().offset), rand(RNGProvider::createRandom<Distribution>())
        {}

        DINLINE void
        init(Space globalCellIdx)
        {
            rand.init(globalCellIdx - offset);
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
        PMACC_ALIGN8(offset, const Space);
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
