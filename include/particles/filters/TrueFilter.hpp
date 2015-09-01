#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace filters {

    /**
     * Filter that returns true for all parameter values
     */
    struct TrueFilter{

        template<class T_Frame>
        HDINLINE bool
        operator()(Space, T_Frame&, uint32_t) const
        {
            return true;
        }
    };

}  // namespace filters
}  // namespace particles
}  // namespace xrt
