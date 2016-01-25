#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"
#include <algorithms/math.hpp>

namespace xrt{
namespace particles {
namespace scatterer {

    /**
     * Scatterer that sets the direction to a fixed value
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct Fixed
    {
        float3_X direction_;

        HINLINE explicit
        Fixed(uint32_t)
        {
            direction_.x() = T_Config::x;
            direction_.y() = tan(T_Config::angleY) * T_Config::x;
            direction_.z() = tan(T_Config::angleZ) * T_Config::x;
            direction_ /= PMaccMath::abs(direction_);
        }

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Direction& dir)
        {
            dir = direction_;
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
