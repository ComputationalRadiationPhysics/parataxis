#pragma once

#include "Particles.hpp"

#include "Particles.kernel"

#include "debug/LogLevels.hpp"

#include <dataManagement/DataConnector.hpp>
#include <particles/memory/buffers/ParticlesBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <mappings/simulation/GridController.hpp>
#include <traits/GetUniqueTypeId.hpp>
#include <traits/Resolve.hpp>
#include <math/Vector.hpp>

namespace xrt{

    namespace detail {

        template<unsigned T_simDim>
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
        PMacc::ParticlesBase<T_ParticleDescription, MappingDesc>( cellDescription ), gridLayout( cellDescription.getGridLayout() ), datasetID( datasetID )
    {
        this->particlesBuffer = new BufferType( gridLayout.getDataSpace(), gridLayout.getGuard() );

        const uint32_t commTag = PMacc::traits::GetUniqueTypeId<FrameType, uint32_t>::uid() + static_cast<uint32_t>(CommTag::SPECIES_FIRSTTAG);
        PMacc::log< XRTLogLvl::MEMORY > ( "communication tag for species %1%: %2%" ) % FrameType::getName() % commTag;

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
    {

    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::init()
    {
        PMacc::Environment<>::get().DataConnector().registerData( *this );
    }

    template<typename T_ParticleDescription>
    void Particles<T_ParticleDescription>::update(uint32_t )
    {
        ParticlesBaseType::template shiftParticles < PMacc::CORE + PMacc::BORDER > ();
    }

    template< typename T_ParticleDescription>
    template< typename T_SrcParticleDescription,
              typename T_ManipulateFunctor>
    void Particles<T_ParticleDescription>::deviceCloneFrom( Particles< T_SrcParticleDescription> &src, T_ManipulateFunctor& functor )
    {
        dim3 block( PMacc::math::CT::volume<SuperCellSize>::type::value );

        PMacc::log< XRTLogLvl::SIM_STATE > ( "clone species %1%" ) % FrameType::getName();
        __picKernelArea( kernelCloneParticles, this->cellDescription, PMacc::CORE + PMacc::BORDER )
            (block) ( this->getDeviceParticlesBox(), src.getDeviceParticlesBox(), functor );
        this->fillAllGaps();
    }

} // namespace xrt
