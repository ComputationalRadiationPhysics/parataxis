#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetPhaseByTimestep.hpp"

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
        float_X curPhase;

        PlaneWavePhase(float_64 phi_0, uint32_t currentStep)
        {
            // Get current phase (calculated exactly in high precision)
            curPhase = functors::GetPhaseByTimestep<Species>()(currentStep, phi_0);
        }

        DINLINE void
        init(Space totalCellIdx) const
        {}

        DINLINE float_X
        operator()(uint32_t timeStep) const
        {
            return curPhase;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
