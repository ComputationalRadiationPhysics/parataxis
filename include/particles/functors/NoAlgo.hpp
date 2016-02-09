#pragma once

#include "xrtTypes.hpp"

namespace xrt{
namespace particles {
namespace functors {

    /**
     * Functor that can be used for scattering and pushing but does nothing
     */
    struct NoAlgo
    {
        template<class T_DensityBox, typename T_Position, typename T_Direction>
        HDINLINE void operator()(const T_DensityBox&, T_Position& pos, T_Direction& dir)
        {}
    };


}  // namespace functors
}  // namespace particles
}  // namespace xrt
