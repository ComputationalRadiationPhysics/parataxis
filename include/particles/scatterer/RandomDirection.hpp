#pragma once

#include "xrtTypes.hpp"
#include "random/Random.hpp"
#include <algorithms/math.hpp>

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Scatterer that changes the momentum based on 2 random angles (spherical coordinates)
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct RandomDirection
    {
        using Config = T_Config;

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
            float_X azimuthAngle   = rand() * float_X(Config::maxAzimuth - Config::minAzimuth) + float_X(Config::minAzimuth);
            const float_X maxPolarCos = PMaccMath::cos(float_X(Config::maxPolar));
            const float_X minPolarCos = PMaccMath::cos(float_X(Config::minPolar));
            // Polar angle is mostly in [0, PI] so cos(polar) is [1, -1] -> Reverse order of min/max here
            float_X polarAngle = acos(rand() * (minPolarCos - maxPolarCos) + maxPolarCos);
            float_X sinPolar, cosPolar, sinAzimuth, cosAzimuth;
            PMaccMath::sincos(polarAngle, sinPolar, cosPolar);
            PMaccMath::sincos(azimuthAngle, sinAzimuth, cosAzimuth);
            mom.x() = sinPolar * cosAzimuth;
            mom.y() = sinPolar * sinAzimuth;
            mom.z() = cosPolar;
        }

    private:
        PMACC_ALIGN8(rand, random::Random<>);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
