/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include "parataxisTypes.hpp"

namespace parataxis {
namespace particles {

    /**
     * Collection of policies related to filling the grid with particles
     *
     * \tparam T_Count     Returns the number of particles for a given cell and time
     *                     init(Space2D localCellIdx)
     *                     Functor: int32_t(int32_t timeStep)
     * \tparam T_Position  Returns the in-cell position for a given cell and particle number
     *                     init(Space2D localCellIdx)
     *                     setCount(int32_t particleCount)
     *                     Functor: float_D(int32_t numParticle), gets called for all particles
     *                      with i in [0, particleCount)
     * \tparam T_Phase     Returns the phase of the particles in a given cell at a given time
     *                     init(Space2D localCellIdx)
     *                     Functor: float_X(int32_t timeStep)
     * \tparam T_Direction Returns the initial direction of the particles for a given cell and time
     *                     init(Space2D localCellIdx)
     *                     Functor: float_D(int32_t timeStep)
     *
     */
    template<
        class T_Count,
        class T_Position,
        class T_Phase,
        class T_Direction
    >
    struct ParticleFillInfo
    {
        using Count    = T_Count;
        using Position = T_Position;
        using Phase    = T_Phase;
        using Direction = T_Direction;

    private:
        Count getCount_;
    public:
        Position getPosition;
        Phase getPhase;
        Direction getDirection;

        ParticleFillInfo(const Count& count, const Position& position, const Phase& phase, const Direction& direction):
            getCount_(count), getPosition(position), getPhase(phase), getDirection(direction)
        {}

        HDINLINE void
        init(Space localCellIdx)
        {
            getCount_.init(localCellIdx);
            getPosition.init(localCellIdx);
            getPhase.init(localCellIdx);
            getDirection.init(localCellIdx);
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
            class T_Direction
        >
    ParticleFillInfo<T_Count, T_Position, T_Phase, T_Direction>
    getParticleFillInfo(const T_Count& count, const T_Position& position, const T_Phase& phase, const T_Direction& direction)
    {
        return ParticleFillInfo<T_Count, T_Position, T_Phase, T_Direction>(count, position, phase, direction);
    }

}  // namespace particles
}  // namespace parataxis
