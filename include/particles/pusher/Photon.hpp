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
        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox&, T_Position& pos, T_Direction& dir)
        {
            // Allow higher precision in position
            using PosType = typename T_Position::type;
            for(uint32_t d=0; d<simDim; ++d)
            {
                // This assumes a unit vector for the direction, otherwise we need to normalize it here
                pos[d] += (static_cast<PosType>(dir[d]) * static_cast<PosType>(SPEED_OF_LIGHT) * static_cast<PosType>(DELTA_T)) / static_cast<PosType>(cellSize[d]);
            }
        }
    };

}  // namespace pusher
}  // namespace particles
}  // namespace xrt
