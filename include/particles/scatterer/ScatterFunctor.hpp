#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace scatterer {

    /**
     * General scatter functor composite that handles scattering in 2 steps:
     *     - It first calls the condition functor which should whether
     *       the particles momentum should be changed
     *     - If the momentum should be changed, the direction functor is called
     *       which should then adjust the momentum
     * The interface of the functors is similar to this one:
     *     - ctor takes the currentStep
     *     - init(Space totalCellIdx) which is called before the first call to the functor
     *     - functor taking densityBox centered at the particles cell, position and
     *       momentum of the particle.
     *       The Condition functor should return a bool and gets only const-Refs as the arguments
     *
     * The functors must also support bmpl::apply which passes the Species-Type as the argument
     */
    template<
        class T_Condition,
        class T_Direction,
        class T_Species = bmpl::_1
        >
    struct ScatterFunctor
    {
        using Condition = typename bmpl::apply<T_Condition, T_Species>::type;
        using Direction = typename bmpl::apply<T_Direction, T_Species>::type;

        HINLINE explicit
        ScatterFunctor(uint32_t currentStep): condition_(currentStep), direction_(currentStep)
        {}

        HDINLINE void
        init(Space totalCellIdx)
        {
            condition_.init(totalCellIdx);
            direction_.init(totalCellIdx);
        }

        template<class T_DensityBox, typename T_Position, typename T_Momentum>
        HDINLINE void
        operator()(const T_DensityBox& density, const T_Position& pos, T_Momentum& mom)
        {
            if(condition_(density, pos, const_cast<const T_Momentum&>(mom)))
            {
                direction_(density, pos, mom);
            }
        }

    private:
        PMACC_ALIGN8(condition_, Condition);
        PMACC_ALIGN8(direction_, Direction);
    };

}  // namespace scatterer
}  // namespace particles
}  // namespace xrt
