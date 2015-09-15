#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Same in-cell position for all particles
     */
    template<class T_Config>
    struct ConstPosition
    {
        using Config = T_Config;

        HDINLINE void
        init(Space2D totalCellIdx)
        {}

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE floatD_X
        operator()(uint32_t numPart)
        {
            return ToVector<Config, simDim>()();
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
