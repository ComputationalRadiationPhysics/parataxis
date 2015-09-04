#pragma once

#include "particles/ProcessLeavingParticles.hpp"
#include <particles/ParticleDescription.hpp>

namespace xrt {
namespace particles {

    template<
        typename T_Name,
        typename T_SuperCellSize,
        typename T_ValueTypeSeq,
        typename T_Flags = bmpl::vector0<>,
        typename T_MethodsList = bmpl::vector0<>,
        typename T_FrameExtensionList = bmpl::vector0<>
    >
    using ParticleDescription = PMacc::ParticleDescription<
            T_Name,
            T_SuperCellSize,
            T_ValueTypeSeq,
            T_Flags,
            PMacc::HandleGuardRegion<
                PMacc::particles::ExchangeParticles,
                ProcessLeavingParticles
            >,
            T_MethodsList,
            T_FrameExtensionList
    >;

}  // namespace particles
}  // namespace xrt
