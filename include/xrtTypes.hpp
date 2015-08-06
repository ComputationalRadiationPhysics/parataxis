#pragma once

#include <types.h>
#include <dimensions/DataSpace.hpp>
#include <memory/buffers/GridBuffer.hpp>
#include <Environment.hpp>

namespace xrt {

    constexpr unsigned SIMDIM = DIM2;
    constexpr unsigned SC_SIZE = 16; /* arbitrarily chosen SuperCellSize! */
    typedef float Prec;
    typedef PMacc::Environment<SIMDIM> Environment;
    typedef PMacc::DataSpace<SIMDIM> Space;
    typedef PMacc::GridController<SIMDIM> GC;
    typedef PMacc::GridBuffer< Prec, SIMDIM > Buffer;
    typedef PMacc::GridLayout<SIMDIM> GridLayout;
    typedef PMacc::SubGrid<SIMDIM> SubGrid;

    enum CommTag
    {
        BUFF
    };

}  // namespace xrt
