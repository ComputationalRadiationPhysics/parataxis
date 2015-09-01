#pragma once

#include "xrtTypes.hpp"
#include "particles/functors/IterateSpecies.hpp"
#include "particles/functors/CopySpeciesToHost.hpp"
#include "particles/filters/IndexFilter.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"

#include <particles/memory/buffers/MallocMCBuffer.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <string>

namespace xrt {
namespace plugins {

    namespace detail {

        template<class T_ParticlesType>
        struct PrintParticle
        {
            template<class T_Particle>
            void
            operator()(const Space globalIdx, T_Particle&& particle)
            {
                // Convert global + local position to position in µm
                floatD_64 pos;
                for(int i=0; i<simDim; ++i)
                    pos[i] = (float_64(globalIdx[i]) + particle[position_][i]) * cellSize[i] * UNIT_LENGTH * 1e6;

                std::cout << "Particle " << globalIdx << " (" << T_ParticlesType::FrameType::getName() << particle[globalId_] << "): " << " => " << pos << "[µm]\n";
            }
        };

    }  // namespace detail

    template<class T_ParticlesType>
    class PrintParticles : public ISimulationPlugin
    {
        using ParticlesType = T_ParticlesType;

        typedef MappingDesc::SuperCellSize SuperCellSize;
        typedef floatD_X FloatPos;

        uint32_t notifyFrequency;

        std::string analyzerName;
        std::string analyzerPrefix;
        std::vector<unsigned> idxOffset;
        std::vector<unsigned> idxSize;

        Space idxOff, idxSz;

    public:
        PrintParticles():
            notifyFrequency(0),
            analyzerName("PositionsParticles: write position of all particles of a species to std::cout"),
            analyzerPrefix(ParticlesType::FrameType::getName() + std::string("_position"))
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        virtual ~PrintParticles()
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((analyzerPrefix + ".period").c_str(), po::value<uint32_t >(&notifyFrequency), "enable analyzer [for each n-th step]")
                ((analyzerPrefix + ".offset").c_str(), po::value<std::vector<uint32_t> >(&idxOffset)->multitoken(), "Print only particles of cells with idx greater than this")
                ((analyzerPrefix + ".size").c_str(), po::value<std::vector<uint32_t> >(&idxSize)->multitoken(), "Print only particles of that many cells (in each dimension)")
                ;
        }

        std::string pluginGetName() const override
        {
            return analyzerName;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Printing particles at timestep %1% (%2%ns)") % currentStep % (currentStep * DELTA_T * UNIT_TIME * 1e9);
            PMacc::DataConnector &dc = Environment::get().DataConnector();

            /* synchronizes the MallocMCBuffer to the host side */
            PMacc::MallocMCBuffer& mallocMCBuffer = dc.getData<PMacc::MallocMCBuffer>(PMacc::MallocMCBuffer::getName());
            //particles::functors::CopySpeciesToHost<PIC_Photons>()();

            int particlesCount = 0;
            auto& particles = dc.getData<PIC_Photons>(PIC_Photons::FrameType::getName());
            const Space localOffset = Environment::get().SubGrid().getLocalDomain().offset;
            PMacc::AreaMapping< PMacc::CORE + PMacc::BORDER, MappingDesc > mapper(*cellDescription_);
            particles::functors::IterateSpecies<PIC_Photons>()(
                    particlesCount,
                    particles.getHostParticlesBox(mallocMCBuffer.getOffset()),
                    localOffset,
                    mapper,
                    particles::filters::IndexFilter(idxOff, idxSz),
                    detail::PrintParticle<ParticlesType>()
                    );

            dc.releaseData(PIC_Photons::FrameType::getName());
            dc.releaseData(PMacc::MallocMCBuffer::getName());
            PMacc::log< XRTLogLvl::IN_OUT >("%1% particles printed") % particlesCount;
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
        {}
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {}

    protected:
        void pluginLoad() override
        {
            if(!notifyFrequency)
                return;

            Environment::get().PluginConnector().setNotificationPeriod(this, notifyFrequency);
            if(idxOffset.empty() && idxSize.empty())
            {
                // Use an area from the center as the default
                auto& subGrid = Environment::get().SubGrid();
                Space offset = subGrid.getTotalDomain().offset + subGrid.getTotalDomain().size / 2;
                for(unsigned i = 0; i<simDim; ++i)
                {
                    idxOffset.push_back(offset[i]);
                    idxSize.push_back(5);
                }
                idxOffset[0] = subGrid.getTotalDomain().offset[0];
                idxSize[0] = subGrid.getTotalDomain().size[0];
            }
            idxOffset.resize(simDim);
            idxSize.resize(simDim, 1);
            for(unsigned i = 0; i<simDim; ++i)
            {
                idxOff[i] = idxOffset[i];
                idxSz[i]  = idxSize[i];
            }
            PMacc::log< XRTLogLvl::PLUGINS >("Printing particles in range %1% [%2%] every %3% timesteps") % idxOff % idxSz % notifyFrequency;
        }
    };

}  // namespace plugins
}  // namespace xrt
