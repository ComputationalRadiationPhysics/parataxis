#pragma once

#include "xrtTypes.hpp"
#include <algorithms/math.hpp>
#include <cmath>

namespace xrt {
namespace particles {
namespace initPolicies {

    namespace detail {

        /**
         * Calculates the  number of particles per dim for a given
         * particle count and size of the area/volume trying to use
         * equally spacing along the axis
         */
        template<int32_t T_dim>
        struct CalcPartsPerDim;

        template<>
        struct CalcPartsPerDim<2>
        {
            template<class T_Vec>
            HDINLINE PMacc::math::Int<2>
            operator()(int32_t particleCount, T_Vec&& size) const
            {
                using PMacc::algorithms::math::sqrt;
                using PMacc::algorithms::math::float2int_rn;
                using PMacc::algorithms::math::ceil;
                PMacc::math::Int<2> res;
                res.x() = float2int_rn(sqrt(size[0] / size[1] * particleCount));
                if(res.x() == 0)
                    res.x() = 1;
                res.y() = ceil(float_X(particleCount) / res.x());
                return res;
            }
        };

        template<>
        struct CalcPartsPerDim<3>
        {
            template<class T_Vec>
            HDINLINE PMacc::math::Int<3>
            operator()(int32_t particleCount, T_Vec&& size) const
            {
                using PMacc::algorithms::math::pow;
                using PMacc::algorithms::math::sqrt;
                using PMacc::algorithms::math::float2int_rn;
                using PMacc::algorithms::math::ceil;
                PMacc::math::Int<3> res;
                res.x() = float2int_rn(
                            pow(
                                size[0] * size[0] / (size[1] * size[2]) * particleCount,
                                float_X(1./3.)
                            )
                          );
                if(res.x() == 0)
                    res.x() = 1;
                res.y() = float2int_rn(sqrt(size[1] / size[2] * particleCount / res.x()));
                if(res.y() == 0)
                     res.y() = 1;
                res.z() = ceil(float_X(particleCount) / (res.x() * res.y()));

                return res;
            }
        };

    }  // namespace detail

    /**
     * Distributes particles evenly in a cell
     */
    struct EvenDistPosition
    {
        DINLINE void
        init(Space totalCellIdx) const
        {}

        DINLINE void
        setCount(int32_t particleCount)
        {
            partsPerDim = detail::CalcPartsPerDim<simDim>()(particleCount, laserConfig::distSize);
        }

        DINLINE position_pic::type
        operator()(int32_t numPart)
        {
            position_pic::type result;
            for(int32_t i = 0; i < simDim; ++i){
                int32_t remaining = numPart / partsPerDim[i];
                int32_t curDimIdx  = numPart - remaining;
                result[i] = curDimIdx * laserConfig::distSize[i] / cellSize[i];
                numPart = remaining;
            }
            return result;
        }
    private:
        PMACC_ALIGN(partsPerDim, PMacc::math::Int<simDim>);
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
