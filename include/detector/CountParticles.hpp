#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace detector {

    /**
     * Functor that can be used as an AccumPolicy for \see PhotonDetector
     * It simply counts the number of particles for each cell
     */
    template<class T_Species = bmpl::_1>
    class CountParticles
    {
    public:
        using Type = uint32_t;

        template< typename T_Particle >
        DINLINE void
        operator()(Type& oldVal, T_Particle& particle, float_64 currentTime) const
        {
            atomicAdd(&oldVal, 1);
        }
    };

}  // namespace detector
}  // namespace xrt
