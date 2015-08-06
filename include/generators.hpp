#pragma once

#include <types.h>

namespace xrt {
namespace generators {

    /**
     * Generates a line (in 2D, --> Point in 1D/Area in 3D)
     * Returns value if current index in \tT_fixedDim equals pos, 0 otherwise
     */
    template<typename T, unsigned T_fixedDim>
    struct Line{
        const size_t pos_;
        const T value_;
        Line(size_t pos, T value = T(1)): pos_(pos), value_(value){}

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            return idx[T_fixedDim] == pos_ ? value_ : 0;
        }
    };

}  // namespace generators
}  // namespace xrt
