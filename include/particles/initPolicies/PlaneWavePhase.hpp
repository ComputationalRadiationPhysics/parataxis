#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetAngFrequency.hpp"

namespace xrt {
namespace particles {
namespace initPolicies {

    /**
     * Returns a phase that is only time-variant (e.g. the case with plane waves)
     */
    template<class T_Species>
    struct PlaneWavePhase
    {
        using Species = T_Species;

        DINLINE void
        init(Space totalCellIdx) const
        {}

        DINLINE float_X
        operator()(uint32_t timeStep) const
        {
            const float_X omega = functors::GetAngularFrequency<Species>()();
            /* phase phi = phi_0 - omega * t; Here we assume phi_0 = 0 */
            return - omega * timeStep * DELTA_T;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
