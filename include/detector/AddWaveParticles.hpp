#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetAngFrequency.hpp"
#include <math/Complex.hpp>

namespace xrt {
namespace detector {

    /**
     * Functor that can be used as an AccumPolicy for \see PhotonDetector
     * It adds particles per cell with wave properties (phase and amplitude)
     */
    template<class T_Species = bmpl::_1>
    class AddWaveParticles
    {
        using Species = T_Species;
        using FrameType = typename Species::FrameType;
        static_assert(HasFlag_t<FrameType, amplitude<> >::value, "Species has no amplitude set");
        using Amplitude = GetResolvedFlag_t<FrameType, amplitude<> >;
    public:
        using Type = PMacc::math::Complex<float_64>;

        template< typename T_Particle >
        HDINLINE Type
        operator()(Type oldVal, T_Particle& particle, float_X currentTime) const
        {
            const float_64 omega = particles::functors::GetAngularFrequency<Species>()();
            /* Add a phase offset based on the current time. This makes the detector oscillate with the
             * particles frequency so the addition of 2 particles that had the same way results in adding
             * their amplitudes. However the phase is (phi_0 + omega*dt) with dt being the traveling time
             * instead of (phi_0 - omega * t_0') with t_0' being the start time of the 2nd particle.
             * I assume that the intensity will be correct although there is an absolute phase offset compared
             * to the correct phase that is proportional to the detector distance (->TODO: Check)
             */
            float_64 phase = particle[startPhase_] + float_64(currentTime) * omega;
            float_64 sinPhase, cosPhase;
            PMaccMath::sincos(phase, sinPhase, cosPhase);
            return oldVal + Type(Amplitude::getValue() * cosPhase, Amplitude::getValue() * sinPhase);
        }
    };

}  // namespace detector
}  // namespace xrt
