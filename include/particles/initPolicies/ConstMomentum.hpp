#pragma once

#include "xrtTypes.hpp"
#include "ToVector.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Same momentum for all particles
     */
    template<class T_Config>
    struct ConstMomentum
    {
        using Config = T_Config;

        HDINLINE void
        init(Space2D totalCellIdx)
        {}

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE momentum::type
        operator()(uint32_t timeStep)
        {
            return ToVector<Config, momentum::type::dim>()();
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
