#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {

    /**
     * Collection of policies related to filling the grid with particles
     *
     * \tparam T_Count    Returns the number of particles for a given cell and time
     *                    init(Space2D totalCellIdx)
     *                    Functor: int32_t(int32_t timeStep)
     * \tparam T_Position Returns the in-cell position for a given cell and particle number
     *                    init(Space2D totalCellIdx)
     *                    setCount(int32_t particleCount)
     *                    Functor: float_D(int32_t numParticle), gets called for all particles
     *                      with i in [0, particleCount)
     * \tparam T_Phase    Returns the phase of the particles in a given cell at a given time
     *                    init(Space2D totalCellIdx)
     *                    Functor: float_X(int32_t timeStep)
     * \tparam T_Momentum Returns the initial momentum of the particles for a given cell and time
     *                    init(Space2D totalCellIdx)
     *                    Functor: float_D(int32_t timeStep)
     *
     */
    template<
        class T_Count,
        class T_Position,
        class T_Phase,
        class T_Momentum
    >
    struct ParticleFillInfo
    {
        using Count    = T_Count;
        using Position = T_Position;
        using Phase    = T_Phase;
        using Momentum = T_Momentum;

    private:
        Count getCount_;
    public:
        Position getPosition;
        Phase getPhase;
        Momentum getMomentum;

        ParticleFillInfo(const Count& count, const Position& position, const Phase& phase, const Momentum& momentum):
            getCount_(count), getPosition(position), getPhase(phase), getMomentum(momentum)
        {}

        HDINLINE void
        init(Space2D totalCellIdx)
        {
            getCount_.init(totalCellIdx);
            getPosition.init(totalCellIdx);
            getPhase.init(totalCellIdx);
            getMomentum.init(totalCellIdx);
        }

        HDINLINE int32_t
        getCount(int32_t timeStep)
        {
            int32_t ct = getCount_(timeStep);
            getPosition.setCount(ct);
            return ct;
        }
    };

    template<
            class T_Count,
            class T_Position,
            class T_Phase,
            class T_Momentum
        >
    ParticleFillInfo<T_Count, T_Position, T_Phase, T_Momentum>
    getParticleFillInfo(const T_Count& count, const T_Position& position, const T_Phase& phase, const T_Momentum& momentum)
    {
        return ParticleFillInfo<T_Count, T_Position, T_Phase, T_Momentum>(count, position, phase, momentum);
    }

}  // namespace particles
}  // namespace xrt
