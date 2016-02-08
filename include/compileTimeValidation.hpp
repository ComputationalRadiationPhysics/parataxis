#pragma once

namespace xrt {

    // The PMacc algorithm assumes that we can only get to the neighbor cell in one timestep
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_WIDTH,  "Cells are to big");
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_HEIGHT, "Cells are to big");
    static_assert(DELTA_T * SPEED_OF_LIGHT <= CELL_DEPTH,  "Cells are to big");

    // It is (almost always) an error, if we cannot pass half a cell in 1 timestep
    static_assert(DELTA_T * SPEED_OF_LIGHT > CELL_WIDTH * 0.5,  "Cells are to small");
    static_assert(DELTA_T * SPEED_OF_LIGHT > CELL_HEIGHT * 0.5, "Cells are to small");
    static_assert(DELTA_T * SPEED_OF_LIGHT > CELL_DEPTH * 0.5,  "Cells are to small");

}  // namespace xrt
