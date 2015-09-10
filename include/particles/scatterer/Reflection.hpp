#pragma once

#include "xrtTypes.hpp"

namespace xrt{
namespace particles {
namespace scatterer {

    /**
     * Scatterer that inverts the momentum when any density is found
     */
    struct Reflection
    {
        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        HDINLINE void operator()(const T_DensityBox& density, T_Position& pos, T_Momentum& mom)
        {
            if(density(Space::create(0)) > float_X(1e-3))
            {
                mom *= -1;
            }
        }
    };


}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
