#pragma once

#include "xrtTypes.hpp"

#include "Field.hpp"
#include "GatherSlice.hpp"
#include "PngCreator.hpp"
#include "generators.hpp"

#include <dimensions/DataSpace.hpp>
#include <mappings/kernel/MappingDescription.hpp>
#include <traits/NumberOfExchanges.hpp>

#include <memory>

namespace xrt {

    class Simulation
    {
        const int32_t steps;
        Space gridSize, devices, periodic;

        Field<MappingDesc> field;
        std::unique_ptr<Buffer> densityBuf;
        GatherSlice<typename Buffer::DataBoxType::ValueType> gather;
        bool isMaster;

    public:

        Simulation(int32_t steps, Space gridSize, Space devices, Space periodic) :
            steps(steps), gridSize(gridSize), isMaster(false)
        {
            /* Set up device mappings and create streams etc. */
            Environment::get().initDevices(devices, periodic);

            /* Divide grid evenly among devices */
            Space localGridSize(gridSize / devices);
            /* Set up environment (subGrid and singletons) with this size */
            GC& gc = Environment::get().GridController();
            Environment::get().initGrids( gridSize, localGridSize, gc.getPosition() * localGridSize);
        }

        ~Simulation()
        {}

        void init()
        {
            /* Get our grid */
            const SubGrid& subGrid = Environment::get().SubGrid();
            /* Our layout is the subGrid with guard cells as big as one super cell */
            GridLayout layout( subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT());

            /* Create our field with 1 border and 1 guard super cell */
            field.init(MappingDesc(layout.getDataSpace(), 1, 1));
            densityBuf.reset(new Buffer(layout, false));

            auto guardingCells(Space::create(1));
            for (uint32_t i = 1; i < PMacc::traits::NumberOfExchanges<simDim>::value; ++i)
            {
                densityBuf->addExchange(PMacc::GUARD, PMacc::Mask(i), guardingCells, CommTag::BUFF);
            }

            isMaster = gather.init(MessageHeader(gridSize, layout, subGrid.getLocalDomain().offset), true);
            field.createDensityDistribution(densityBuf->getDeviceBuffer().getDataBox(), generators::Line<float, 0>(50));
        }

        void start()
        {
            /* gather::operator() gathers all the buffers and assembles those to  *
             * a complete picture discarding the guards.                          */
            densityBuf->deviceToHost();
            auto picture = gather(densityBuf->getHostBuffer().getDataBox());
            if (isMaster){
                PngCreator png;
                png(0, picture, gridSize);
            }
        }
    };

}  // namespace xrt
