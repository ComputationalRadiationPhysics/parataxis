#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Same direction for all particles
     */
    template<class T_Config>
    struct ConstDirection
    {
        using Config = T_Config;

        HDINLINE void
        init(Space totalCellIdx)
        {}

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE direction::type
        operator()(uint32_t timeStep)
        {
            const direction::type dir =  ToVector<Config, direction::type::dim>()();
            // We need unit vectors!
            return dir / PMaccMath::abs(dir);
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
