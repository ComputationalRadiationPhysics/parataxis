#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Condition functor for scattering that returns false and therefore leads to no scattering
     */
    template<
        class T_Species = bmpl::_1
        >
    struct Never
    {
        HINLINE explicit
        Never(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Direction& dir)
        {
            return false;
        }

    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
