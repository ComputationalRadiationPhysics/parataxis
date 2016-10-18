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
namespace particles {
namespace filters {

    /**
     * Filter that returns true, if the globalCellIdx is in a given region
     */
    struct IndexFilter{

        IndexFilter(Space offset, Space size): startIdx_(offset), endIdx_(offset + size)
        {}

        template<class T_Frame>
        HDINLINE bool
        operator()(Space globalCellIdx, T_Frame& frame, uint32_t partInFrameIdx) const
        {
            for(uint32_t i=0; i<simDim; ++i)
            {
                if(globalCellIdx[i] < startIdx_[i] || globalCellIdx[i] >= endIdx_[i])
                    return false;
            }
            return true;
        }
    protected:
        PMACC_ALIGN(startIdx_, const Space);
        PMACC_ALIGN(endIdx_, const Space);
    };

}  // namespace filters
}  // namespace particles
}  // namespace xrt
