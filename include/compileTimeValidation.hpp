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
