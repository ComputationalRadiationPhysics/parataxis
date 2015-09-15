#pragma once

#include "xrtTypes.hpp"
#include <traits/HasFlag.hpp>
#include <traits/GetFlagType.hpp>

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
        using FrameType = typename Species::FrameType;

        DINLINE void
        init(Space2D totalCellIdx) const
        {}

        DINLINE int32_t
        operator()(uint32_t timeStep) const
        {
            using PMacc::traits::HasFlag;
            using PMacc::traits::GetFlagType;

            typedef typename HasFlag<FrameType, wavelength<> >::type hasWavelength;
            static_assert(hasWavelength::value, "Species has no wavelength set");
            typedef typename PMacc::traits::Resolve<
                        typename GetFlagType<FrameType, wavelength<> >::type
                    >::type foundWavelength;
            const float_X waveNumber = 2 * float_X(PI) / (foundWavelength::getValue() / UNIT_LENGTH);
            return waveNumber * SPEED_OF_LIGHT * timeStep * DELTA_T;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace xrt
