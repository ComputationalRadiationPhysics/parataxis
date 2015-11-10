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
        init(Space totalCellIdx)
        {}

        HDINLINE void
        setCount(int32_t particleCount)
        {}

        DINLINE momentum::type
        operator()(uint32_t timeStep)
        {
            const momentum::type mom =  ToVector<Config, momentum::type::dim>()();
            // We need unit vectors!
            return mom / PMaccMath::abs(mom);
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
