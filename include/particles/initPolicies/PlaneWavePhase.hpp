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
        init(Space2D totalCellIdx) const
        {}

        DINLINE int32_t
        operator()(uint32_t timeStep) const
        {
            const float_X omega = functors::GetAngularFrequency<Species>()();
            return omega * timeStep * DELTA_T;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
