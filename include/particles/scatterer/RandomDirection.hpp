#pragma once

#include "xrtTypes.hpp"
#include "Random.hpp"
#include <algorithms/math.hpp>

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Scatterer that changes the momentum based on 2 random angles (spherical coordinates)
     */
    template<class T_Species = bmpl::_1>
    struct RandomDirection
    {

        HINLINE explicit
        RandomDirection(uint32_t currentStep)
        {}

        DINLINE void
        init(Space totalCellIdx)
        {
            rand.init(totalCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        DINLINE void
        operator()(const T_DensityBox& density, const T_Position& pos, T_Momentum& mom)
        {
            float_X polarAngle   = rand() * float_X(PI);
            float_X azimuthAngle = rand() * float_X(2 * PI);
            float_X sinPolar, cosPolar, sinAzimuth, cosAzimuth;
            PMaccMath::sincos(polarAngle, sinPolar, cosPolar);
            PMaccMath::sincos(azimuthAngle, sinAzimuth, cosAzimuth);
            mom.x() = sinPolar * cosAzimuth;
            mom.y() = sinPolar * sinAzimuth;
            mom.z() = cosPolar;
        }

    private:
        PMACC_ALIGN(rand, Random<>);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
