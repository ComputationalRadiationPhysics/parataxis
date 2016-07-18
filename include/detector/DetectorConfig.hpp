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
