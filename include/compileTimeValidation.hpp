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

namespace xrt {

    // The PMacc algorithm assumes that we can only get to the neighbor cell in one timestep
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_WIDTH,  "Cells are to big");
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_HEIGHT, "Cells are to big");
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_DEPTH,  "Cells are to big");

    // It is (almost always) an error, if we cannot pass half a cell in 1 timestep
    // However we could have a higher resolution in one direction so fail only if this matches in ALL directions
    static_assert(DELTA_T * SPEED_OF_LIGHT > CELL_WIDTH * 0.5 ||
            DELTA_T * SPEED_OF_LIGHT > CELL_HEIGHT * 0.5 ||
            DELTA_T * SPEED_OF_LIGHT > CELL_DEPTH * 0.5,  "Cells are to small");

}  // namespace xrt
