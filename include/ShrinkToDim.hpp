#pragma once

#include "xrtTypes.hpp"

namespace xrt {

    /// Shrinks a DataSpace to the new dimensionality by removing the first dimension(s)
    /// Acts like the shrink-member-function but leaves nD->nD untouched
    template<uint32_t T_newDims, uint32_t T_oldDims>
    struct ShrinkToDim
    {
        static_assert(T_newDims < T_oldDims, "New dimensions must be less than old dimensions");
        HDINLINE PMacc::DataSpace<T_newDims>
        operator()(const PMacc::DataSpace<T_oldDims>& old)
        {
            return old.template shrink<T_newDims>(T_oldDims - T_newDims);
        }
    };

    template<uint32_t T_dims>
    struct ShrinkToDim<T_dims, T_dims>
    {
        HDINLINE PMacc::DataSpace<T_dims>
        operator()(const PMacc::DataSpace<T_dims>& old)
        {
            return old;
        }
    };

    template<uint32_t T_newDims, uint32_t T_oldDims>
    HDINLINE PMacc::DataSpace<T_newDims>
    shrinkToDim(const PMacc::DataSpace<T_oldDims>& old)
    {
        return ShrinkToDim<T_newDims, T_oldDims>()(old);
    }

}  // namespace xrt
