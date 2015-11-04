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
            /* Azimuth angle is the angle around the x-axis [0,2PI) and polar angle is the angle around the z-axis [0-PI)
             * Note that compared to e.g. wikipedia the z and x axis are swapped as our usual propagation direction is X
             * but it does not influence anything as the axis can be arbitrarily named */
            float_X azimuthAngle = rand() * float_X(Config::maxAzimuth - Config::minAzimuth) + float_X(Config::minAzimuth);
            // Here we'd actually need an adjustment so that the coordinates are evenly distributed on a unit sphere but for very small angles this is ok
            float_X polarAngle   = rand() * float_X(Config::maxPolar - Config::minPolar) + float_X(Config::minPolar);

            /* Now we have the azimuth and polar angles by which we want to change the current direction. So we need some rotations:
             * Assume old momentum = A, new momentum = B, |A| = 1
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

            float_X sinPolar, cosPolar, sinAzimuth, cosAzimuth;
            PMaccMath::sincos(polarAngle, sinPolar, cosPolar);
            PMaccMath::sincos(azimuthAngle, sinAzimuth, cosAzimuth);
            const float_X x = mom.x();
            const float_X y = mom.y();
            const float_X z = mom.z();
            if(PMaccMath::abs(1 + x) <= std::numeric_limits<float_X>::min())
            {
                // Special case: x=-1 --> y=z=0 (unit vector), so avoid division by zero
                mom.x() = -cosPolar;
                mom.y() = -sinAzimuth * sinPolar;
                mom.z() = -cosAzimuth * sinPolar;
            }else
            {
                mom.x() = x * cosPolar -          z            * cosAzimuth * sinPolar -          y            * sinAzimuth * cosPolar;
                mom.y() = y * cosPolar -      y * z / (1 + x)  * cosAzimuth * sinPolar + (x + z * z / (1 + x)) * sinAzimuth * sinPolar;
                mom.z() = z * cosPolar + (x + y * y / (1 + x)) * cosAzimuth * sinPolar -      y * z / (1 + x)  * sinAzimuth * sinPolar;
            }
        }

    private:
        PMACC_ALIGN8(rand, random::Random<>);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
