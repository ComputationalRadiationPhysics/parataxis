#pragma once

#include "xrtTypes.hpp"
#include <random/distributions/Uniform.hpp>

namespace xrt {
namespace particles {
namespace initPolicies {

    template<class T_Species>
    struct RandomPosition
    {
#if XRT_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;
#endif

        HINLINE RandomPosition(): offset(Environment::get().SubGrid().getLocalDomain().offset)
#if !XRT_USE_SLOW_RNG
                ,rand(RNGProvider::createRandom<Distribution>())
#endif
        {}

        DINLINE void
        init(Space globalCellIdx)
        {
            rand.init(globalCellIdx - offset);
        }

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE position_pic::type
        operator()(uint32_t numPart)
        {
            position_pic::type result;
            result.x() = 0;
            for(uint32_t i = 1; i < simDim; ++i)
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
