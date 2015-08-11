#pragma once

#include "simulation_defines.hpp"
#include <types.h>
#include <dimensions/DataSpace.hpp>
#include <dimensions/GridLayout.hpp>
#include <memory/buffers/GridBuffer.hpp>
#include <Environment.hpp>

namespace xrt {

    typedef PMacc::Environment<simDim> Environment;
    typedef PMacc::DataSpace<simDim> Space;
    typedef PMacc::DataSpace<2> Space2D;
    typedef PMacc::GridController<simDim> GC;
    typedef PMacc::GridBuffer< precisionXRT::precisionType, simDim > Buffer;
    typedef PMacc::GridLayout<simDim> GridLayout;
    typedef PMacc::SubGrid<simDim> SubGrid;

}  // namespace xrt
