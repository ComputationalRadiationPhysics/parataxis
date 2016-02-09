#pragma once

#include "xrtTypes.hpp"

namespace xrt{
namespace particles {
namespace scatterer {

    /**
     * Scatterer that inverts the direction
     */
    template<class T_Species = bmpl::_1>
    struct Reflect
    {
        HINLINE explicit
        Reflect(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Direction& dir)
        {
            dir *= -1;
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
