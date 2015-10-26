#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace math {

    struct Max
    {
        template<typename Dst, typename Src >
        DINLINE void operator()(Dst & dst, const Src & src) const
        {
            dst = max(dst, src);
        }
    };

}  // namespace math
}  // namespace xrt
