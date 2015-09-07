#pragma once

namespace xrt {
namespace particles {

    /**
     * Policy for HandleGuardRegions that calls processLeavingParticles which handles particles before deleting them
     */
    struct ProcessLeavingParticles
    {
        template< class T_Particles >
        void
        handleOutgoing(T_Particles& par, int32_t direction) const
        {
            par.processLeavingParticles(direction);
        }

        template< class T_Particles >
        void
        handleIncoming(T_Particles& par, int32_t direction) const
        {}
    };

}  // namespace particles
}  // namespace xrt
