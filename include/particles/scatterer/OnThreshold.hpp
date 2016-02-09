#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Condition functor for scattering that returns true when the particle hits any density
     * with at least the given threshold
     */
    template<
        class T_Config,
        class T_Species = bmpl::_1
        >
    struct OnThreshold
    {
        static constexpr float_X threshold = T_Config::threshold;

        HINLINE explicit
        OnThreshold(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Direction& dir)
        {
            return density(Space::create(0)) >= threshold;
        }

    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
