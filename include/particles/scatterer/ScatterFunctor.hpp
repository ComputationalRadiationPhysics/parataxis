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

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * General scatter functor composite that handles scattering in 2 steps:
     *     - It first calls the condition functor which should whether
     *       the particles direction should be changed
     *     - If the direction should be changed, the direction functor is called
     *       which should then adjust the direction
     *     - Returns true iff the particle was scattered
     * The interface of the functors is similar to this one:
     *     - ctor takes the currentStep
     *     - init(Space totalCellIdx) which is called before the first call to the functor
     *     - functor taking densityBox centered at the particles cell, position and
     *       direction of the particle.
     *       The Condition functor should return a bool and gets only const-Refs as the arguments
     *
     * The functors must also support bmpl::apply which passes the Species-Type as the argument
     */
    template<
        class T_Condition,
        class T_ChangeDirection,
        class T_Species = bmpl::_1
        >
    struct ScatterFunctor
    {
        using Condition = typename bmpl::apply<T_Condition, T_Species>::type;
        using ChangeDirection = typename bmpl::apply<T_ChangeDirection, T_Species>::type;

        HINLINE explicit
        ScatterFunctor(uint32_t currentStep): condition(currentStep), changeDirection(currentStep)
        {}

        HDINLINE void
        init(Space totalCellIdx)
        {
            condition.init(totalCellIdx);
            changeDirection.init(totalCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, T_Direction& dir)
        {
            if(condition(density, pos, const_cast<const T_Direction&>(dir)))
            {
                changeDirection(density, pos, dir);
                return true;
            }else
                return false;
        }

    private:
        PMACC_ALIGN8(condition, Condition);
        PMACC_ALIGN8(changeDirection, ChangeDirection);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
