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

namespace parataxis {
namespace particles {
namespace scatterer {

    /**
     * Condition functor for scattering that returns false and therefore leads to no scattering
     */
    template<
        class T_Species = bmpl::_1
        >
    struct Never
    {
        HINLINE explicit
        Never(uint32_t)
        {}

        HDINLINE void
        init(Space)
        {}

        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE bool
        operator()(const T_DensityBox& density, const T_Position& pos, const T_Direction& dir)
        {
            return false;
        }

    };

}  // namespace scatterer
}  // namespace particles
}  // namespace parataxis
