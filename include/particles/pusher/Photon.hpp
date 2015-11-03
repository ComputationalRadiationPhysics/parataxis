#pragma once

#include "xrtTypes.hpp"

namespace xrt{
namespace particles {
namespace pusher {

    /**
     * Pusher used to move particles with the speed of light
     */
    struct Photon
    {
        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        HDINLINE void operator()(const T_DensityBox&, T_Position& pos, T_Momentum& mom)
        {
            // This is required due to a bug(?) that you can't pass global constexpr as const& in device code
            constexpr float_X SPEED = SPEED_OF_LIGHT;
            // This assumes a unit vector for the momentum
            const T_Momentum vel  = mom * SPEED;

            for(uint32_t d=0; d<simDim; ++d)
            {
                pos[d] += (vel[d] * DELTA_T) / cellSize[d];
            }
        }
    };

}  // namespace pusher
}  // namespace particles
}  // namespace xrt
