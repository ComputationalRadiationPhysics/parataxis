#pragma once

#include "xrtTypes.hpp"
#include <random/distributions/Uniform.hpp>

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Condition functor for scattering that depends on a probability that can be dependent
     * on the density in the particles cell.
     * The Config needs a static function calcProbability(float_X density) the returns the
     * probability for scattering in the range [0, 1]
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct OnRandom
    {
        using Config = T_Config;
        using Distribution = PMacc::random::distributions::Uniform<float>;
#if XRT_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using RNGHandle = typename RNGProvider::Handle;
        using Random = typename RNGHandle::GetRandomType<Distribution>::type;
#endif

        HINLINE explicit
        OnRandom(uint32_t currentStep): offset(Environment::get().SubGrid().getLocalDomain().offset)
#if !XRT_USE_SLOW_RNG
                ,rand(RNGProvider::createRandom<Distribution>())
#endif
        {}

        DINLINE void
        init(Space globalCellIdx)
        {
#if XRT_USE_SLOW_RNG
            rand.init(globalCellIdx - offset);
#else
            randHandle.init(globalCellIdx - offset);
            rand = randHandle.applyDistribution<Distribution>();
#endif
        }

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        DINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Direction& dir)
        {
            float_X probability = Config::calcProbability(density(Space::create(0)));
            return (rand() < probability);
        }

    private:
        PMACC_ALIGN8(offset, const Space);
#if !XRT_USE_SLOW_RNG
        PMACC_ALIGN8(randHandle, RNGHandle);
#endif
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
