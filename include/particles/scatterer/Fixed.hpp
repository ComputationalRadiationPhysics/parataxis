#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"
#include <algorithms/math.hpp>
#include <algorithms/TypeCast.hpp>

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
            using namespace PMacc::algorithms::precisionCast;
            float3_64 tmpDir;
            tmpDir.x() = T_Config::x;
            tmpDir.y() = tan(T_Config::angleY) * T_Config::x;
            tmpDir.z() = tan(T_Config::angleZ) * T_Config::x;
            direction_ = precisionCast<float_X>(tmpDir / PMaccMath::abs(tmpDir));
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
