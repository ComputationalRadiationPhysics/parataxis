#pragma once

#include "debug/LogLevels.hpp"

#include <debug/VerboseLog.hpp>
#include <types.h>

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
