/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Richard Pausch,
 *                     Felix Schmitt, Alexander Grund
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

#include "Particles.hpp"

#include "particles/Particles.kernel"
#include "particles/ParticlesInit.kernel"
#include "particles/scatterer/ScatterFunctor.hpp"
#include "debug/LogLevels.hpp"
#include "GetFlagOrDefault.hpp"

#include <dataManagement/DataConnector.hpp>
#include <particles/memory/buffers/ParticlesBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <mappings/kernel/ExchangeMapping.hpp>
#include <mappings/kernel/BorderMapping.hpp>
#include <mappings/simulation/GridController.hpp>
#include <traits/GetUniqueTypeId.hpp>
#include <dimensions/SuperCellDescription.hpp>
#include <math/Vector.hpp>
#include <algorithms/reverseBits.hpp>
#include <fields/DensityField.hpp>
#include <type_traits>

namespace parataxis{

    namespace detail {

        template<uint32_t T_simDim>
        struct AddExchanges;

        template<>
        struct AddExchanges<2>
        {
            template<class T_Buffer>
            static void
            add(T_Buffer* particlesBuffer, uint32_t commTag)
            {
                using PMacc::Mask;
                particlesBuffer->addExchange( Mask( PMacc::LEFT ) + Mask( PMacc::RIGHT ),
                                              exchangeSize::X, commTag);
                particlesBuffer->addExchange( Mask( PMacc::TOP ) + Mask( PMacc::BOTTOM ),
                                              exchangeSize::Y, commTag);
                //edges of the simulation area
                particlesBuffer->addExchange( Mask( PMacc::RIGHT + PMacc::TOP ) + Mask( PMacc::LEFT + PMacc::TOP ) +
                                              Mask( PMacc::LEFT + PMacc::BOTTOM ) + Mask( PMacc::RIGHT + PMacc::BOTTOM ),
                                              exchangeSize::Edges, commTag);
            }
        };

        template<>
        struct AddExchanges<3>
        {
            template<class T_Buffer>
            static void
            add(T_Buffer* particlesBuffer, uint32_t commTag)
            {
                AddExchanges<2>::add(particlesBuffer, commTag);
                using PMacc::Mask;

                particlesBuffer->addExchange( Mask( PMacc::FRONT ) + Mask( PMacc::BACK ),
                                              exchangeSize::Z, commTag);
                //edges of the simulation area
                particlesBuffer->addExchange( Mask( PMacc::FRONT + PMacc::TOP ) + Mask( PMacc::BACK + PMacc::TOP ) +
                                              Mask( PMacc::FRONT + PMacc::BOTTOM ) + Mask( PMacc::BACK + PMacc::BOTTOM ),
                                              exchangeSize::Edges, commTag);
                particlesBuffer->addExchange( Mask( PMacc::FRONT + PMacc::RIGHT ) + Mask( PMacc::BACK + PMacc::RIGHT ) +
                                              Mask( PMacc::FRONT + PMacc::LEFT ) + Mask( PMacc::BACK + PMacc::LEFT ),
                                              exchangeSize::Edges, commTag);
                //corner of the simulation area
                particlesBuffer->addExchange( Mask( PMacc::TOP + PMacc::FRONT + PMacc::RIGHT ) + Mask( PMacc::TOP + PMacc::BACK + PMacc::RIGHT ) +
                                              Mask( PMacc::BOTTOM + PMacc::FRONT + PMacc::RIGHT ) + Mask( PMacc::BOTTOM + PMacc::BACK + PMacc::RIGHT ),
                                              exchangeSize::Corner, commTag);
                particlesBuffer->addExchange( Mask( PMacc::TOP + PMacc::FRONT + PMacc::LEFT ) + Mask( PMacc::TOP + PMacc::BACK + PMacc::LEFT ) +
                                              Mask( PMacc::BOTTOM + PMacc::FRONT + PMacc::LEFT ) + Mask( PMacc::BOTTOM + PMacc::BACK + PMacc::LEFT ),
                                              exchangeSize::Corner, commTag);
            }
        };

    }  // namespace detail

    template<typename T_ParticleDescription>
    Particles<T_ParticleDescription>::Particles( MappingDesc cellDescription, PMacc::SimulationDataId datasetID ) :
        PMacc::ParticlesBase<T_ParticleDescription, MappingDesc>( cellDescription ), gridLayout( cellDescription.getGridLayout() ), datasetID( datasetID ),
        densityField_(nullptr), lastProcessedStep_(0)
    {
        this->particlesBuffer = new BufferType( gridLayout.getDataSpace(), gridLayout.getGuard() );

        const uint32_t commTag = PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + static_cast<uint32_t>(CommTag::SPECIES_FIRSTTAG);
        PMacc::log< PARATAXISLogLvl::MEMORY > ( "communication tag for species %1%: %2%" ) % FrameType::getName() % commTag;

        detail::AddExchanges<simDim>::add(this->particlesBuffer, commTag);
    }

    template< typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::createParticleBuffer()
    {
        this->particlesBuffer->createParticleBuffer();
    }

    template< typename T_ParticleDescription>
    Particles<T_ParticleDescription>::~Particles()
    {
        delete this->particlesBuffer;
    }

    template< typename T_ParticleDescription>
    PMacc::SimulationDataId Particles<T_ParticleDescription>::getUniqueId()
    {
        return datasetID;
    }

    template< typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::synchronize()
    {
        this->particlesBuffer->deviceToHost();
    }

    template< typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::syncToDevice()
    {}

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::init(fields::DensityField* densityField)
    {
        densityField_ = densityField;
        PMacc::Environment<>::get().DataConnector().registerData( *this );
    }

    template<typename T_ParticleDescription>
    template<typename T_InitFunctor>
    void Particles<T_ParticleDescription>::add(T_InitFunctor&& initFunctor, uint32_t timeStep)
    {
        const SubGrid& subGrid = Environment::get().SubGrid();
        Space localOffset = subGrid.getLocalDomain().offset;
        /* Add only to first cells */
        if(simDim == 3 && localOffset[laserConfig::DIRECTION] > 0)
            return;

        const PMacc::BorderMapping<MappingDesc> mapper(this->cellDescription, laserConfig::EXCHANGE_DIR);
        Space block = MappingDesc::SuperCellSize::toRT();
        if(simDim == 3)
            block[laserConfig::DIRECTION] = 1;
        __cudaKernel(kernel::fillGridWithParticles<Particles>)
            (mapper.getGridDim(), block.toDim3())
            ( initFunctor,
              localOffset,
              this->particlesBuffer->getDeviceParticleBox(),
              timeStep,
              mapper
              );

    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::update(uint32_t currentStep)
    {
        /* If the species defines a pusher/scatterer use it, otherwise fall back to default (None) */
        /* if nothing was defined we use None as fallback */
        using Pusher = GetFlagOrDefault_t<FrameType, particlePusher<>, particles::pusher::None>;
        using ScatterCondition = GetFlagOrDefault_t<FrameType, particleScatterCondition<>, particles::scatterer::conditions::Never>;
        using ScatterDirection = GetFlagOrDefault_t<FrameType, particleScatterDirection<>, particles::scatterer::direction::Reflect>;

        using Scatterer = particles::scatterer::ScatterFunctor<ScatterCondition, ScatterDirection, Particles>;

        /* Create the frame solver used to manipulate the particle along its way */
        using FrameSolver = kernel::PushParticlePerFrame<Scatterer, Pusher>;
        FrameSolver frameSolver{Scatterer(currentStep)};

        /* This contains the working area for one block on the field(s).
         * It can include margin/halo if the field needs to be interpolated between that of the surrounding cells
         */
        typedef PMacc::SuperCellDescription<
            typename MappingDesc::SuperCellSize/*,
            LowerMargin,
            UpperMargin*/
            > BlockArea;

        Space blockSize = MappingDesc::SuperCellSize::toRT();

        /* Change position of particles and set flags whether to move them out of their cell */
        __cudaKernelArea( kernel::moveAndMarkParticles<BlockArea>, this->cellDescription, PMacc::CORE + PMacc::BORDER )
            (blockSize)
            ( this->getDeviceParticlesBox(),
              Environment::get().SubGrid().getLocalDomain().offset,
              densityField_->getDeviceDataBox(),
              frameSolver
              );
        /* Actually move particles out of their cells keeping the frames in a valid state */
        ParticlesBaseType::template shiftParticles< PMacc::CORE + PMacc::BORDER >();

        lastProcessedStep_ = currentStep;
    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::processLeavingParticles(int32_t direction)
    {
        using Detector = GetFlagOrDefault_t<FrameType, particleDetector<>, detector::NoDetector>;
        auto& dc = Environment::get().DataConnector();
        Detector& detector = dc.getData<Detector>(Detector::getName(), true);

        PMacc::ExchangeMapping<PMacc::GUARD, MappingDesc> mapper(this->cellDescription, direction);
        Space gridSize(mapper.getGridDim());

        const Space localOffset = Environment::get().SubGrid().getLocalDomain().offset;
        const auto detectParticle = detector.getDetectParticle(lastProcessedStep_);

        __cudaKernel(kernel::detectAndDeleteParticles)
                (gridSize, Particles::TileSize)
                (this->getDeviceParticlesBox(),
                 localOffset,
                 detectParticle,
                 detector.getDeviceDataBox(),
                 mapper);
        dc.releaseData(Detector::getName());
    }

} // namespace parataxis
