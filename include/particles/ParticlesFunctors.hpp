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

#include "debug/LogLevels.hpp"

#include <debug/VerboseLog.hpp>
#include <pmacc_types.hpp>

namespace xrt {
namespace particles {

    template<typename T_SpeciesName>
    struct AssignNull
    {
        typedef T_SpeciesName SpeciesName;

        template<typename T_StorageTuple>
        void operator()(T_StorageTuple& tuple)
        {
            tuple[SpeciesName()] = nullptr;
        }
    };

    template<typename T_SpeciesName>
    struct CallDelete
    {
        typedef T_SpeciesName SpeciesName;

        template<typename T_StorageTuple>
        void operator()(T_StorageTuple& tuple)
        {
            __delete(tuple[SpeciesName()]);
        }
    };

    template<typename T_SpeciesName>
    struct CreateSpecies
    {
        typedef T_SpeciesName SpeciesName;
        typedef typename SpeciesName::type SpeciesType;

        template<typename T_StorageTuple, typename T_CellDescription>
        HINLINE void operator()(T_StorageTuple& tuple, T_CellDescription* cellDesc) const
        {
            tuple[SpeciesName()] = new SpeciesType(cellDesc->getGridLayout(), *cellDesc, SpeciesType::FrameType::getName());
        }
    };

    template<typename T_SpeciesName>
    struct CallCreateParticleBuffer
    {
        typedef T_SpeciesName SpeciesName;
        typedef typename SpeciesName::type SpeciesType;

        template<typename T_StorageTuple>
        HINLINE void operator()(T_StorageTuple& tuple) const
        {

            typedef typename SpeciesType::FrameType FrameType;

            PMacc::log< XRTLogLvl::MEMORY >("mallocMC: free slots for species %3%: %1% a %2%") %
                mallocMC::getAvailableSlots(sizeof (FrameType)) %
                sizeof (FrameType) %
                FrameType::getName();

            tuple[SpeciesName()]->createParticleBuffer();
        }
    };

    template<typename T_SpeciesName>
    struct CallInit
    {
        typedef T_SpeciesName SpeciesName;
        typedef typename SpeciesName::type SpeciesType;

        template<typename T_StorageTuple>
        HINLINE void operator()(T_StorageTuple& tuple) const
        {
            tuple[SpeciesName()]->init();
        }
    };

    template<typename T_SpeciesName>
    struct CallReset
    {
        typedef T_SpeciesName SpeciesName;
        typedef typename SpeciesName::type SpeciesType;

        template<typename T_StorageTuple>
        HINLINE void operator()(T_StorageTuple& tuple,
                                const uint32_t currentStep)
        {
            tuple[SpeciesName()]->reset(currentStep);
        }
    };

}  // namespace particles
}  // namespace xrt
