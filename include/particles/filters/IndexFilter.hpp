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
