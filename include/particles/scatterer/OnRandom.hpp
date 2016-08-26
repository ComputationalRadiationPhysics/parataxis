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
#if XRT_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;
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
            rand.init(globalCellIdx - offset);
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
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
