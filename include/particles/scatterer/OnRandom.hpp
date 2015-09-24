#pragma once

#include "xrtTypes.hpp"
#include "Random.hpp"

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

        HINLINE explicit
        OnRandom(uint32_t currentStep)
        {}

        DINLINE void
        init(Space totalCellIdx)
        {
            rand.init(totalCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        DINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Momentum& mom)
        {
            float_X probability = Config::calcProbability(density(Space::create(0)));
            return (rand() < probability);
        }

    private:
        PMACC_ALIGN8(rand, Random<>);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
