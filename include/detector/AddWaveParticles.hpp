#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/GetPhaseByTimestep.hpp"
#include "particles/functors/GetWavelength.hpp"
#include "particles/functors/GetAngFrequency.hpp"
#include <math/Complex.hpp>
#include <algorithms/math.hpp>
#include <basicOperations.hpp>

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

        const float_X curPhase_; // Phase contribution at current timestep [0, 2*PI)
   public:
        using FloatType = float_64;
        using Type = PMacc::math::Complex<FloatType>;

        struct OutputTransformer
        {
            HDINLINE FloatType
            operator()(const Type& val) const
            {
                return PMaccMath::abs2(val);
            }
        };

        explicit AddWaveParticles(uint32_t curTimestep):
                curPhase_(-particles::functors::GetPhaseByTimestep<Species>()(curTimestep + 1) + 2*PI)
        {
            // Phase should be w*t. GetPhaseByTimestep returns phi_0 - w*t -> Invert it and add 2*PI to make it [0, 2*PI)
            // Note that we need the next timestep. The particles are moved, so they have the position they would have at the END of the current ts
            // -> Use phase from the end of this timestep (start of next ts)
            if(std::is_same<float, float_X>::value &&
                    PMaccMath::max(CELL_WIDTH, PMaccMath::max(CELL_HEIGHT, CELL_DEPTH)) > 10e5 * particles::functors::GetWavelength<Species>()())
            {
                // Particle phase must be calculated for particles at most 1 cell out of the volume
                // Using the current algorithm error analysis led to the equation above
                throw std::runtime_error("Precisions is to low for the selected cell dimensions or wavelength");
            }
            // Also: 2*alpha_max * MaxExtendOfVolume < 10³ * lambda is assumed (about 1000 maxima are far more than enough)
        }

        template< typename T_Particle >
        DINLINE void
        operator()(Type& oldVal, T_Particle& particle, const Space& globalCellIdx) const
        {
            /* Phase is: k*dn + w*t + phi_0 with dn...distance to detector, k...wavenumber, phi_0...start phase
             * w*t is the same for all particles so we pre-calculate it (reduce to 2*PI) in high precision
             * dn must be exact compared to lambda which is hard.
             * We would also need to calculate dn to a fixed point for all photons in the detector-cell (e.g.) middle for prober interference
             * In the end, only the phase difference matters. We take the ray from the cell (0,0) from the exiting plane as a reference and calculate the phase
             * difference to this. It is the projection of the vector from the reference point to the particles position on the reference ray,
             * whose vector is given by the particles direction (for large distances all rays to a given detector cell are parallel)
             */
            float_X phase = particle[startPhase_] + curPhase_;
            if(phase > static_cast<float_X>(2*PI))
                phase -= static_cast<float_X>(2*PI);

            /* The projection is k * (dir * pos)/|dir| (dot product)
             * dir is already the unit vector hence we have don't need the division.
             * For better precision summands are reduced mod 2*PI
             */
            const auto dir = particle[direction_];
            const float_X omega = particles::functors::GetAngularFrequency<Species>()();
            const float_X k = omega / SPEED_OF_LIGHT;
            // Add the negated dot product (reduced by 2*PI), negated as the difference to the reference ray gets smaller with increasing index
            // the x-Position is 0 by definition so don't use it
            const float_X distDiffG = globalCellIdx.y() * CELL_HEIGHT * dir.y() + globalCellIdx.z() * CELL_DEPTH * dir.z();
            phase += PMaccMath::fmod(-distDiffG * k, static_cast<float_X>(2*PI));
            // Now add the negated dot product for the remaining in-cell position
            const float_X distDiffI = float_X(particle[position_].x()) * CELL_WIDTH  * dir.x() +
                                      float_X(particle[position_].y()) * CELL_HEIGHT * dir.y() +
                                      float_X(particle[position_].z()) * CELL_DEPTH  * dir.z();
            phase += PMaccMath::fmod(-distDiffI * k, static_cast<float_X>(2*PI));

            /*if(dir.z() > 1e-6)
            {
                printf("Dir: %g, %g, %g\n", dir.x(), dir.y(), dir.z());
                printf("%i,%i -> %g+%g+%g+%g=%g -> %g\n", globalCellIdx.y(), globalCellIdx.z(), particle[startPhase_], curPhase_,
                        -distDiffG * k,
                        -distDiffI * k, particle[startPhase_] + curPhase_, phase+2*PI);
            }*/

            trigo_X sinPhase, cosPhase;
            PMaccMath::sincos<trigo_X>(phase, sinPhase, cosPhase);
            PMacc::atomicAddWrapper(&oldVal.get_real(), FloatType(Amplitude::getValue() * cosPhase));
            PMacc::atomicAddWrapper(&oldVal.get_imag(), FloatType(Amplitude::getValue() * sinPhase));
        }
    };

}  // namespace detector
}  // namespace xrt
