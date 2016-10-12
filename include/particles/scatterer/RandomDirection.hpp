/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include "xrtTypes.hpp"
#include "debug/LogLevels.hpp"
#include <random/distributions/Uniform.hpp>
#include <algorithms/math.hpp>
#include <debug/VerboseLog.hpp>
#include <cmath>

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * Scatterer that changes the direction based on 2 random angles (spherical coordinates)
     */
    template<class T_Config, class T_Species = bmpl::_1>
    struct RandomDirection
    {
        using Config = T_Config;
#if XRT_USE_SLOW_RNG
        using Random = SlowRNGFunctor;
#else
        using Distribution = PMacc::random::distributions::Uniform<float>;
        using Random = typename RNGProvider::GetRandomType<Distribution>::type;
#endif

        HINLINE explicit
        RandomDirection(uint32_t currentStep)
#if !XRT_USE_SLOW_RNG
                :rand(RNGProvider::createRandom<Distribution>())
#endif
        {
            static bool angleChecked = false;
            if(!angleChecked)
            {
                angleChecked = true;
                if(PMaccMath::abs(Config::minAzimuth) > PI || PMaccMath::abs(Config::maxAzimuth) > PI)
                    PMacc::log<XRTLogLvl::DOMAINS>("Azimuth angle not in range [-PI,PI]. Possibly reduced precision!");
                if(PMaccMath::abs(Config::minPolar) > PI || PMaccMath::abs(Config::maxPolar) > PI)
                    PMacc::log<XRTLogLvl::DOMAINS>("Polar angle not in range [-PI,PI]. Possibly reduced precision!");
            }
        }

        DINLINE void
        init(Space localCellIdx)
        {
            rand.init(localCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        DINLINE void
        operator()(const T_DensityBox& density, const T_Position& pos, T_Direction& dir)
        {
            /* Azimuth angle is the angle around the x-axis [0,2PI) and polar angle is the angle around the z-axis [0-PI)
             * Note that compared to e.g. wikipedia the z and x axis are swapped as our usual propagation direction is X
             * but it does not influence anything as the axis can be arbitrarily named */
            float_X azimuthAngle = rand() * float_X(Config::maxAzimuth - Config::minAzimuth) + float_X(Config::minAzimuth);
            // To get an even distribution on a unit sphere we need to modify the distribution of the polar angle using arccos.
            float_X polarAngle;
            // TODO: Calculate max angle this is valid for
            if(std::is_same<float_X, float_32>::value && Config::minPolar == 0. && Config::maxPolar <= 1e-2)
            {
                // For float32 we don't get small angles as we'd need to calculate the arccos around 1 where the possible
                // float values are sparse.
                // For small angle we can approximate the density distribution and arccos around 0, else we need to use double precision
                polarAngle = PMaccMath::sqrt<sqrt_X>(rand()) * Config::maxPolar;
            }else
            {
                // Optimization with compile-time evaluated ternary for common case
                const float_64 minPolarCos = (Config::minPolar == 0.) ? float_64(1.) : PMaccMath::cos(float_64(Config::minPolar));
                const float_64 maxPolarCos = PMaccMath::cos(float_64(Config::maxPolar));
                // Note that cos(minPolar)>=cos(maxPolor) for minPolar<=maxPolar
                polarAngle   = PMaccMath::acos<float_64>(rand() * (minPolarCos - maxPolarCos) + maxPolarCos);
            }
            /* Now we have the azimuth and polar angles by which we want to change the current direction. So we need some rotations:
             * Assume old direction = A, new direction = B, |A| = 1
             * There is a rotation matrix R_A so that A = R_A * e_X (with e_X = [1,0,0] )
             * It is also easy to generate a rotation Matrix R_B which transforms a point in Cartesian coordinates by the azimuth and polar angle
             * So we can say B = R_A * R_B * R_A^T * A (rotate A back to e_X, rotate by R_B and rotate again to As coordinate system)
             * This can be simplified as R_A^T * A = e_X and R_B' = R_B * e_X = [cos(polar), sin(polar)*sin(azimuth), sin(polar)*cos(azimuth)]
             * --> B = R_A * R_B'
             * To get R_A we need to find how to rotate e_X to A which means a rotation with cos(a) = A*e_X (dot product) around A x e_X (cross product)
             * Formula for the general case is on wikipedia but can be simplified with cos(a) = A_x, A_y² + A_z² = 1 - A_x² = sin²(a) = (1 + A_x)(1 - A_x)
             * So finally one gets B = {{x,-y,-z}, {y,x+z²/(1+x),-y*z/(1+x)},{+z,-y*z/(1+x),x+y²/(1+x)}}*{cos(t),sin(t)sin(p),sin(t)cos(p)}
             * With x,y,z=A_x,..., t=polar, p=azimuth. This solved by Wolfram Alpha results in the following formulas
             */

            trigo_X sinPolar, cosPolar, sinAzimuth, cosAzimuth;
            PMaccMath::sincos<trigo_X>(polarAngle, sinPolar, cosPolar);
            PMaccMath::sincos<trigo_X>(azimuthAngle, sinAzimuth, cosAzimuth);
            const float_X x = dir.x();
            const float_X y = dir.y();
            const float_X z = dir.z();
            if(PMaccMath::abs(1 + x) <= std::numeric_limits<float_X>::min())
            {
                // Special case: x=-1 --> y=z=0 (unit vector), so avoid division by zero
                dir.x() = -cosPolar;
                dir.y() = -sinAzimuth * sinPolar;
                dir.z() = -cosAzimuth * sinPolar;
            }else
            {
                dir.x() = x * cosPolar -          z            * cosAzimuth * sinPolar -          y            * sinAzimuth * sinPolar;
                dir.y() = y * cosPolar -      y * z / (1 + x)  * cosAzimuth * sinPolar + (x + z * z / (1 + x)) * sinAzimuth * sinPolar;
                dir.z() = z * cosPolar + (x + y * y / (1 + x)) * cosAzimuth * sinPolar -      y * z / (1 + x)  * sinAzimuth * sinPolar;
            }
        }

    private:
        PMACC_ALIGN8(rand, Random);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
