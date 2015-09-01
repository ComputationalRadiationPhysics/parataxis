#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace filters {

    /**
     * Filter that returns true, if the globalCellIdx is in a given region
     */
    struct IndexFilter{

        IndexFilter(Space offset, Space size): minIdx_(offset), maxIdx_(offset + size)
        {}

        template<class T_Frame>
        HDINLINE bool
        operator()(Space globalCellIdx, T_Frame& frame, uint32_t partInFrameIdx) const
        {
            for(uint32_t i=0; i<simDim; ++i)
            {
                if(globalCellIdx[i] < minIdx_[i] || globalCellIdx[i] > maxIdx_[i])
                    return false;
            }
            return true;
        }
    protected:
        PMACC_ALIGN(minIdx_, const Space);
        PMACC_ALIGN(maxIdx_, const Space);
    };

}  // namespace filters
}  // namespace particles
}  // namespace xrt
