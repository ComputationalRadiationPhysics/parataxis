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

#include "parataxisTypes.hpp"
#include <algorithms/math.hpp>

namespace parataxis {
namespace particles {
namespace initPolicies {

    namespace detail {

        /**
         * Calculates the  number of particles per dim for a given
         * particle count and size of the area/volume trying to use
         * equal spacing along the axis
         */
        template<uint32_t T_dim>
        struct CalcPartsPerDim;

        template<>
        struct CalcPartsPerDim<2>
        {
            template<class T_Vec>
            HDINLINE PMacc::math::UInt32<2>
            operator()(uint32_t particleCount, T_Vec&& size) const
            {
                using PMacc::algorithms::math::sqrt;
                using PMacc::algorithms::math::float2int_rn;
                using PMacc::algorithms::math::float2int_ru;
                PMacc::math::UInt32<2> res;
                res.x() = float2int_rn(sqrt(size[0] / size[1] * particleCount));
                if(res.x() == 0)
                    res.x() = 1;
                res.y() = float2int_ru(float_X(particleCount) / res.x());
                return res;
            }
        };

        template<>
        struct CalcPartsPerDim<3>
        {
            template<class T_Vec>
            HDINLINE PMacc::math::UInt32<3>
            operator()(uint32_t particleCount, T_Vec&& size) const
            {
                using PMacc::algorithms::math::pow;
                using PMacc::algorithms::math::sqrt;
                using PMacc::algorithms::math::float2int_rn;
                using PMacc::algorithms::math::float2int_ru;
                PMacc::math::UInt32<3> res;
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
                res.z() = float2int_ru(float_X(particleCount) / (res.x() * res.y()));

                return res;
            }
        };

    }  // namespace detail

    /**
     * Distributes particles evenly in a cell
     */
    struct EvenDistPosition
    {
        EvenDistPosition(uint32_t /*timestep*/){}

        DINLINE void
        init(Space /*localCellIdx*/) const
        {}

        DINLINE void
        setCount(uint32_t particleCount)
        {
            partsPerDim = detail::CalcPartsPerDim<simDim>()(particleCount, laserConfig::distSize);
        }

        DINLINE position_pic::type
        operator()(uint32_t numPart)
        {
            position_pic::type result;
            for(uint32_t i = 0; i < simDim; ++i){
                uint32_t remaining = numPart / partsPerDim[i];
                uint32_t curDimIdx  = numPart - remaining;
                result[i] = curDimIdx * laserConfig::distSize[i] / cellSize[i];
                numPart = remaining;
            }
            return result;
        }
    private:
        PMACC_ALIGN(partsPerDim, PMacc::math::UInt32<simDim>);
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace parataxis
