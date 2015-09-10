#pragma once

#include "Particles.hpp"

#include "particles/functors/NoAlgo.hpp"
#include "particles/Particles.kernel"
#include "particles/ParticlesInit.kernel"
#include "debug/LogLevels.hpp"

#include <dataManagement/DataConnector.hpp>
#include <particles/memory/buffers/ParticlesBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <mappings/simulation/GridController.hpp>
#include <traits/GetUniqueTypeId.hpp>
#include <traits/Resolve.hpp>
#include <traits/HasFlag.hpp>
#include <traits/GetFlagType.hpp>
#include <dimensions/SuperCellDescription.hpp>
#include <math/Vector.hpp>
#include <algorithms/reverseBits.hpp>
#include <DensityField.hpp>
#include <type_traits>

namespace xrt{

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
        densityField_(nullptr), detector_(nullptr), nextPartId_(PMacc::DataSpace<1>(1))
    {
        this->particlesBuffer = new BufferType( gridLayout.getDataSpace(), gridLayout.getGuard() );

        const uint32_t commTag = PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + static_cast<uint32_t>(CommTag::SPECIES_FIRSTTAG);
        PMacc::log< XRTLogLvl::MEMORY > ( "communication tag for species %1%: %2%" ) % FrameType::getName() % commTag;

        detail::AddExchanges<simDim>::add(this->particlesBuffer, commTag);

        // Get a unique counter start value per rank
        *nextPartId_.getHostBuffer().getBasePointer() = PMacc::reverseBits(Environment::get().GridController().getGlobalRank());
        nextPartId_.hostToDevice();
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
    void Particles<T_ParticleDescription>::init(DensityField* densityField, detector::Detector* detector)
    {
        densityField_ = densityField;
        detector_ = detector;
        PMacc::Environment<>::get().DataConnector().registerData( *this );
    }

    template<typename T_ParticleDescription>
    template<typename T_DistributionFunctor, typename T_PositionFunctor>
    void Particles<T_ParticleDescription>::add(T_DistributionFunctor&& distributionFunctor, T_PositionFunctor&& positionFunctor)
    {
        PMacc::log< XRTLogLvl::SIM_STATE >("adding particles for species %1%") % FrameType::getName();

        const SubGrid& subGrid = Environment::get().SubGrid();
        Space totalGpuCellOffset = subGrid.getLocalDomain().offset;

        dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );
        __cudaKernelArea(kernel::fillGridWithParticles<Particles>, this->cellDescription, PMacc::CORE + PMacc::BORDER)
            (block)
            ( distributionFunctor,
              positionFunctor,
              totalGpuCellOffset,
              this->particlesBuffer->getDeviceParticleBox(),
              nextPartId_.getDeviceBuffer().getBasePointer()
              );


        this->fillAllGaps();
    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::update(uint32_t currentStep)
    {
        using PMacc::traits::HasFlag;
        using PMacc::traits::GetFlagType;

        /* If the species defines a pusher/scatterer use it, otherwise fall back to default (None) */
        typedef typename HasFlag<FrameType, particlePusher<> >::type hasPusher;
        typedef typename GetFlagType<FrameType, particlePusher<> >::type FoundPusher;
        typedef typename HasFlag<FrameType, particleScatterer<> >::type hasScatterer;
        typedef typename GetFlagType<FrameType, particleScatterer<> >::type FoundScatterer;
        /* if nothing was defined we use None as fallback */
        typedef typename PMacc::traits::Resolve<
                    typename std::conditional<hasPusher::value, FoundPusher, particles::pusher::None >::type
                >::type SelectedPusher;
        typedef typename PMacc::traits::Resolve<
                    typename std::conditional<hasScatterer::value, FoundScatterer, particles::scatterer::None >::type
                >::type SelectedScatterer;

        /* Create the frame solver used to manipulate the particle along its way */
        typedef kernel::PushParticlePerFrame<SelectedPusher, SelectedScatterer> FrameSolver;

        /* This contains the working area for one block on the field(s).
         * It can include margin/halo if the field needs to be interpolated between that of the surrounding cells
         */
        typedef PMacc::SuperCellDescription<
            typename MappingDesc::SuperCellSize/*,
            LowerMargin,
            UpperMargin*/
            > BlockArea;

        dim3 block( MappingDesc::SuperCellSize::toRT().toDim3() );

        /* Change position of particles and set flags whether to move them out of their cell */
        __cudaKernelArea( kernel::moveAndMarkParticles<BlockArea>, this->cellDescription, PMacc::CORE + PMacc::BORDER )
            (block)
            ( this->getDeviceParticlesBox(),
              densityField_->getDeviceDataBox(),
              FrameSolver()
              );
        /* Actually move particles out of their cells keeping the frames in a valid state */
        ParticlesBaseType::template shiftParticles< PMacc::CORE + PMacc::BORDER >();

        lastProcessedStep_ = currentStep;
    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::processLeavingParticles(int32_t direction)
    {
        PMacc::ExchangeMapping<PMacc::GUARD, MappingDesc> mapper(this->cellDescription, direction);
        dim3 grid(mapper.getGridDim());

        const Space localOffset = Environment::get().SubGrid().getLocalDomain().offset;
        const auto detectParticle = detector_->getDetectParticle(lastProcessedStep_);

        __cudaKernel(kernel::detectAndDeleteParticles)
                (grid, Particles::TileSize)
                (this->getDeviceParticlesBox(),
                 localOffset,
                 detectParticle,
                 detector_->getDeviceDataBox(),
                 mapper);
    }

} // namespace xrt
