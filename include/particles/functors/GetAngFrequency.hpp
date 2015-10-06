#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetWavelength.hpp"

namespace xrt {
namespace particles {
namespace functors {

    /**
     * Returns the angular frequency (omega) of a species
     */
    template<typename T_Species>
    struct GetAngularFrequency
    {
        using Species = T_Species;

        HDINLINE float_X
        operator()() const
        {
            // k = 2 * PI / lambda
            const float_X waveNumber = 2 * float_X(PI) / GetWavelength<Species>()();
            // Assume vacuum in the medium -> w = k * v = k * c
            const float_X omega = waveNumber * SPEED_OF_LIGHT;
            return omega;
        }

    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
