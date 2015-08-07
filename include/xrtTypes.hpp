#pragma once

#include "param.loader"

#include <types.h>
#include <dimensions/DataSpace.hpp>
#include <memory/buffers/GridBuffer.hpp>
#include <Environment.hpp>

namespace xrt {

    typedef float Prec;
    typedef PMacc::Environment<simDim> Environment;
    typedef PMacc::DataSpace<simDim> Space;
    typedef PMacc::DataSpace<2> Space2D;
    typedef PMacc::GridController<simDim> GC;
    typedef PMacc::GridBuffer< Prec, simDim > Buffer;
    typedef PMacc::GridLayout<simDim> GridLayout;
    typedef PMacc::SubGrid<simDim> SubGrid;

    enum CommTag
    {
        BUFF
    };

}  // namespace xrt
