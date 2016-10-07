/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
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
