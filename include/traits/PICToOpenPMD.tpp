/**
 * Copyright 2016 Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "xrtTypes.hpp"
#include "traits/SIBaseUnits.hpp"

namespace xrt
{
namespace traits
{

    template<>
    struct OpenPMDName<DensityField>
    {
        static std::string get()
        {
            return "electron_density";
        }
    };

    /** Forward names that are identical in PIConGPU & openPMD
     */
    template<typename T_Identifier>
    struct OpenPMDName
    {
        static std::string get()
        {
            return T_Identifier::getName();
        }
    };

    /** Translate the globalCellIdx (unitless index) into the openPMD
     *  positionOffset (3D position vector, length)
     */
    template<typename T_Type>
    struct OpenPMDName<PMacc::globalCellIdx<T_Type> >
    {
        static std::string get()
        {
            return std::string("positionOffset");
        }
    };

    /** the globalCellIdx can be converted into a positionOffset
     *  until the beginning of the cell by multiplying with the component-wise
     *  cell size in SI
     */
    template<typename T_Type, typename T_Frame>
    struct OpenPMDUnit<PMacc::globalCellIdx<T_Type>, T_Frame>
    {
        static std::vector<double> get()
        {
            std::vector<double> unit(simDim);
            /* cell positionOffset needs two transformations to get to SI:
               cell begin -> dimensionless scaling to grid -> SI */
            for( uint32_t i=0; i < simDim; ++i )
                unit[i] = cellSize[i] * UNIT_LENGTH;

            return unit;
        }

        /** the openPMD positionOffset is an actual (vector) with a lengths that
         *  is added to the position (vector) attribute
         */
        static std::vector<float_64> getDimension()
        {
            std::vector<float_64> unitDimension(NUnitDimension, 0.0);
            unitDimension.at(SIBaseUnits::length) = 1.0;

            return unitDimension;
        }
    };

} // namespace traits
} // namespace xrt
