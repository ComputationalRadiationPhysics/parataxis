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
#include <math/Vector.hpp>

namespace xrt {
namespace detector {

    struct DetectorConfig
    {
        const Space2D size;
        const PMacc::math::Vector<float_X, 2> anglePerCell;

        DetectorConfig(const Space2D& size, const PMacc::math::Vector<float_X, 2>& anglePerCell):
            size(size), anglePerCell(anglePerCell)
        {}
    };

}  // namespace detector
}  // namespace xrt
