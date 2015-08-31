#pragma once

#include "xrtTypes.hpp"

namespace xrt{
namespace particles {
namespace functors {

    /**
     * Pusher used to move particles with the speed of light
     */
    struct PhotonPusher
    {
        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        HDINLINE void operator()(const T_DensityBox&, T_Position& pos, T_Momentum& mom)
        {
            const float_X momAbs = PMaccMath::abs( mom );
            const T_Position vel  = mom * ( SPEED_OF_LIGHT / momAbs );

            for(uint32_t d=0; d<simDim; ++d)
            {
                pos[d] += (vel[d] * DELTA_T) / cellSize[d];
            }
        }
    };


}  // namespace functors
}  // namespace particles
}  // namespace xrt
