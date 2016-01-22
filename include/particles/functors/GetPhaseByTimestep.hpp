#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetAngFrequency.hpp"

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Returns the phase for a given timestep
     */
    template<typename T_Species>
    struct GetPhaseByTimestep
    {
        using Species = T_Species;

        template<typename T_Result = float_X>
        HINLINE T_Result
        operator()(const uint32_t timestep, T_Result phi_0 = 0) const
        {
            /* phase phi = phi_0 - omega * t;
             * Note: This MUST be calculated in double precision as single precision is inexact after ~100 timesteps
             *       Double precision is enough for about 10^10 timesteps
             *       More timesteps (in SP&DP) are possible, if the product is implemented as a summation with summands reduced to 2*PI */
            static const float_64 omega = GetAngularFrequency<Species>()();
            static const float_64 phaseDiffPerTimestep = fmod(omega * DELTA_T, 2 * PI);
            // Reduce summands to range of 2*PI to avoid bit canceling
            float_64 dPhi = fmod(phaseDiffPerTimestep * static_cast<float_64>(timestep), 2 * PI);
            phi_0 = fmod(phi_0, 2 * PI);
            float_64 result = phi_0 - dPhi;
            // Keep in range of [0,2*PI)
            if(result < 0)
                result += 2*PI;
            return result;
        }
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
