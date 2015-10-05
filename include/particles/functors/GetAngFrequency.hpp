#pragma once

#include "xrtTypes.hpp"

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
        using FrameType = typename Species::FrameType;

        using HasWavelength = HasFlag_t<FrameType, wavelength<> > ;
        static_assert(HasWavelength::value, "Species has no wavelength set");
        using Wavelength = GetResolvedFlag_t<FrameType, wavelength<> >;

        HDINLINE float_X
        operator()() const
        {
            // k = 2 * PI / lambda
            const float_X waveNumber = 2 * float_X(PI) / (Wavelength::getValue() / UNIT_LENGTH);
            // Assume vacuum in the medium -> w = k * v = k * c
            const float_X omega = waveNumber * SPEED_OF_LIGHT;
            return omega;
        }

    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
