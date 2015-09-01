#pragma once

#include "xrtTypes.hpp"

#include "convertToSpace.hpp"
#include "particles/functors/ConstDistribution.hpp"
#include "particles/functors/EvenDistPosition.hpp"
#include "particles/Particles.tpp"

#include "Field.hpp"
#include "generators.hpp"
#include "debug/LogLevels.hpp"

#include <particles/memory/buffers/MallocMCBuffer.hpp>
#include <simulationControl/SimulationHelper.hpp>
#include <dimensions/DataSpace.hpp>
#include <mappings/kernel/MappingDescription.hpp>
#include <traits/NumberOfExchanges.hpp>
#include <compileTime/conversion/TypeToPointerPair.hpp>
#include <debug/VerboseLog.hpp>
#include <eventSystem/EventSystem.hpp>

#include <boost/program_options.hpp>
#include <memory>
#include <vector>

namespace xrt {

    namespace po   = boost::program_options;

    class Simulation: public PMacc::SimulationHelper<simDim>
    {
        using Parent = PMacc::SimulationHelper<simDim>;

        PMacc::MallocMCBuffer *mallocMCBuffer;

        PIC_Photons* particleStorage;

        std::vector<uint32_t> gridSize, devices, periodic;

        /* Only valid after pluginLoad */
        MappingDesc cellDescription;

        std::unique_ptr<Field> densityField;

    public:

        Simulation() :
            particleStorage(nullptr)
        {}

        virtual ~Simulation()
        {
            mallocMC::finalizeHeap();
        }

        void notify(uint32_t currentStep) override
        {}

        void pluginRegisterHelp(po::options_description& desc) override
        {
            SimulationHelper<simDim>::pluginRegisterHelp(desc);
            desc.add_options()
                ("devices,d", po::value<std::vector<uint32_t> > (&devices)->multitoken(), "number of devices in each dimension")

                ("grid,g", po::value<std::vector<uint32_t> > (&gridSize)->multitoken(), "size of the simulation grid")

                ("periodic", po::value<std::vector<uint32_t> > (&periodic)->multitoken(),
                 "specifying whether the grid is periodic (1) or not (0) in each dimension, default: no periodic dimensions");
        }

        std::string pluginGetName() const override
        {
            return "X-Ray Tracing";
        }

        uint32_t init() override
        {
            densityField.reset(new Field(cellDescription));

            /* After all memory consuming stuff is initialized we can setup mallocMC with the remaining memory */
            initMallocMC();

            /* ... and allocate the particles (which uses mallocMC) */
            particleStorage = new PIC_Photons(cellDescription, PIC_Photons::FrameType::getName());

            size_t freeGpuMem(0);
            Environment::get().EnvMemoryInfo().getMemoryInfo(&freeGpuMem);
            PMacc::log< XRTLogLvl::MEMORY > ("free mem after all mem is allocated %1% MiB") % (freeGpuMem / MiB);

            if (this->restartRequested)
                std::cerr << "Restarting is not yet supported. Starting from zero" << std::endl;

            densityField->init();
            particleStorage->init(densityField.get());

            densityField->createDensityDistribution(densityFieldInitializer);
            particleStorage->add(particles::functors::ConstDistribution<1>(), particles::functors::EvenDistPosition<PIC_Photons>(0));

            return 0;
        }

        void movingWindowCheck(uint32_t currentStep) override
        {}

        /**
         * Run one simulation step.
         *
         * @param currentStep iteration number of the current step
         */
        void runOneStep(uint32_t currentStep) override
        {
            __startTransaction(__getTransactionEvent());
            particleStorage->update(currentStep);
            PMacc::EventTask commEvt = particleStorage->asyncCommunication(__getTransactionEvent());
            PMacc::EventTask updateEvt = __endTransaction();
            __setTransactionEvent(commEvt + updateEvt);
        }

        const MappingDesc*
        getMappingDesc()
        {
            return &cellDescription;
        }

    protected:
        void pluginLoad() override
        {
            Space periodic  = convertToSpace(this->periodic, true, "");
            Space devices   = convertToSpace(this->devices, 1, "devices (-d)");
            Space gridSize  = convertToSpace(this->gridSize, 1, "grid (-g)");

            /* Set up device mappings and create streams etc. */
            Environment::get().initDevices(devices, periodic);

            /* Divide grid evenly among devices */
            GC& gc = Environment::get().GridController();
            Space localGridSize(gridSize / devices);
            Space localGridOffset(gc.getPosition() * localGridSize);
            /* Set up environment (subGrid and singletons) with this size */
            Environment::get().initGrids( gridSize, localGridSize, localGridOffset);
            PMacc::log< XRTLogLvl::DOMAINS > ("rank %1%; local size %2%; local offset %3%;") %
                    gc.getPosition().toString() % localGridSize.toString() % localGridOffset.toString();

            Parent::pluginLoad();

            /* Our layout is the subGrid with guard cells as big as one super cell */
            GridLayout layout(localGridSize, MappingDesc::SuperCellSize::toRT());
            /* Create with 1 border and 1 guard super cell */
            cellDescription = MappingDesc(layout.getDataSpace(), 1, 1);
            checkGridConfiguration(gridSize, layout);
        }

        void pluginUnload() override
        {

            Parent::pluginUnload();

            __delete(mallocMCBuffer);
            __delete(particleStorage);
        }

        void initMallocMC()
        {
            size_t freeGpuMem(0);
            Environment::get().EnvMemoryInfo().getMemoryInfo(&freeGpuMem);
            freeGpuMem -= reservedGPUMemorySize;

            if( Environment::get().EnvMemoryInfo().isSharedMemoryPool() )
            {
                freeGpuMem /= 2;
                PMacc::log< XRTLogLvl::MEMORY > ("Shared RAM between GPU and host detected - using only half of the 'device' memory.");
            }
            else
                PMacc::log< XRTLogLvl::MEMORY > ("RAM is NOT shared between GPU and host.");

            // initializing the heap for particles
            mallocMC::initHeap(freeGpuMem);
            this->mallocMCBuffer = new PMacc::MallocMCBuffer();
        }

    private:
        void checkGridConfiguration(Space globalGridSize, GridLayout layout)
        {
            for(uint32_t i=0; i<simDim; ++i)
            {
                // global size must be a divisor of supercell size
                // note: this is redundant, while using the local condition below
                assert(globalGridSize[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
                // local size must be a divisor of supercell size
                assert(layout.getDataSpaceWithoutGuarding()[i] % MappingDesc::SuperCellSize::toRT()[i] == 0);
                // local size must be at least 3 superCells (1x core + 2x border)
                // note: size of border = guard_size (in superCells)
                assert(layout.getDataSpaceWithoutGuarding()[i] / MappingDesc::SuperCellSize::toRT()[i] >= 3);
            }
        }
    };

}  // namespace xrt
