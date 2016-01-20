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
        float_X curPhase;

        PlaneWavePhase(float_64 phi_0, uint32_t currentStep)
        {
            /* phase phi = phi_0 - omega * t;
             * Note: This MUST be calculated in double precision as single precision is inexact after ~100 timesteps
             *       Double precision is enough for about 10^10 timesteps
             *       More timesteps (in SP&DP) are possible, if the product is implemented as a summation with summands reduced to 2*PI */
            static const float_64 omega = particles::functors::GetAngularFrequency<Species>()();
            static const float_64 phaseDiffPerTimestep = fmod(omega * DELTA_T, 2 * PI);
            curPhase = fmod(phi_0 - phaseDiffPerTimestep * static_cast<float_64>(currentStep), 2 * PI);
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
