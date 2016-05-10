#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"
#include <algorithms/math.hpp>
#include <algorithms/TypeCast.hpp>

namespace xrt{
namespace particles {
namespace scatterer {

    /**
     * Scatterer that sets the direction linearly to the density
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct LinearDensity
    {
        float3_X direction_;

        HINLINE explicit
        LinearDensity(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Direction& dir)
        {
            dir.x() = 1;
            dir.y() = tan(float_X(T_Config::factorY) * density(Space::create(0)));
            dir.z() = tan(float_X(T_Config::factorZ) * density(Space::create(0)));
            dir = dir / PMaccMath::abs(dir);
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
