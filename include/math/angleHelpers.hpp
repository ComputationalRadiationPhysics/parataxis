#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace math {

    template<typename T>
    HDINLINE T
    rad2deg(const T valInRad)
    {
        return valInRad * 360 / (2 * PI);
    }

}  // namespace math
}  // namespace xrt
