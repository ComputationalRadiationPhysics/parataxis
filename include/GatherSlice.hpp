#pragma once

#include "xrtTypes.hpp"
#include "ReduceZ.hpp"

#include <mappings/simulation/GridController.hpp>
#include <memory/boxes/PitchedBox.hpp>
#include <dimensions/DataSpace.hpp>

#include <mpi.h>
#include <memory>
#include <type_traits>

namespace xrt
{

    struct MessageHeader
    {
        MessageHeader(){}

        MessageHeader(Space simSize, GridLayout layout, Space nodeOffset) :
        simSize(simSize),
        nodeOffset(nodeOffset)
        {
            nodeSize = layout.getDataSpace();
            nodePictureSize = layout.getDataSpaceWithoutGuarding();
            nodeGuardCells = layout.getGuard();
        }

        Space simSize;
        Space nodeSize;
        Space nodePictureSize;
        Space nodeGuardCells;
        Space nodeOffset;

    };

    template<typename T_ValueType>
    struct GatherSlice
    {
        using ValueType = T_ValueType;
        using Box2D = PMacc::DataBox< PMacc::PitchedBox<ValueType, 2> >;

        GatherSlice() : isMaster(false), numRanks(0), filteredData(nullptr), fullData(nullptr), isMPICommInitialized(false)
        {}

        ~GatherSlice()
        {
            filteredData.reset();
            fullData.reset();
            if (isMPICommInitialized)
            {
                MPI_Comm_free(&comm);
                isMPICommInitialized=false;
            }
            isMaster = false;
        }

        /*
         * Initializes a MPI group with all ranks that have active set
         * @return true if current rank is the master of the new group
         */
        bool init(const MessageHeader& header, bool isActive)
        {
            nodeSize = header.nodeSize;
            simSize = header.simSize;

            uint32_t countRanks = Environment::get().GridController().getGpuNodes().productOfComponents();
            std::vector<int32_t> gatherRanks(countRanks); /* rank in WORLD_GROUP or -1 if inactive */
            std::vector<int32_t> groupRanks(countRanks);  /* rank in new group */
            int32_t mpiRank = Environment::get().GridController().getGlobalRank();
            if (!isActive)
                mpiRank = -1;

            MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, &gatherRanks[0], 1, MPI_INT, MPI_COMM_WORLD));
            numRanks = 0;
            for (uint32_t i = 0; i < countRanks; ++i)
            {
                if (gatherRanks[i] != -1)
                {
                    groupRanks[numRanks] = gatherRanks[i];
                    numRanks++;
                }
            }

            MPI_Group group;
            MPI_Group newgroup;
            MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &group));
            MPI_CHECK(MPI_Group_incl(group, numRanks, &groupRanks.front(), &newgroup));

            MPI_CHECK(MPI_Comm_create(MPI_COMM_WORLD, newgroup, &comm));

            if (mpiRank != -1)
            {
                MPI_Comm_rank(comm, &mpiRank);
                isMPICommInitialized = true;
            }else
                return false;

            isMaster = mpiRank == 0;

            /* Collect message headers to master rank */
            if(isMaster)
                headers.reset(new MessageHeader[numRanks]);
            MessageHeader tmpHeader (header);
            MPI_CHECK(MPI_Gather(&tmpHeader, sizeof(MessageHeader), MPI_CHAR,
                                 headers.get(), sizeof(MessageHeader), MPI_CHAR, 0, comm));

            if(!isMaster)
                return false;

            for(uint32_t i=0; i<numRanks; ++i)
            {
                if(headers[i].nodeSize != nodeSize)
                    throw std::runtime_error("NodeSizes must be the same on all nodes");
            }
            return true;
        }

        template<class Box >
        Box2D operator()(Box data, uint32_t zOffset)
        {
            static_assert(std::is_same<typename Box::ValueType, ValueType>::value, "Wrong type");

            const size_t numElements = nodeSize[0] * nodeSize[1];
            if (!fullData && isMaster)
                fullData.reset(new ValueType[numElements * numRanks]);

            const size_t sizeElements = numElements * sizeof (ValueType);

            auto reducedBox = ReduceZ<simDim>::get(data, zOffset);
            MPI_CHECK(MPI_Gather(reducedBox.getPointer(), sizeElements, MPI_CHAR,
                                 fullData.get(), sizeElements, MPI_CHAR,
                                 0, comm));

            if(!isMaster)
            {
                return Box2D(PMacc::PitchedBox<ValueType, 2 > (
                            nullptr,
                            Space2D(),
                            Space2D(simSize.x(), simSize.y()),
                            simSize.x() * sizeof (ValueType)
                        ));
            }

            if (!filteredData)
                filteredData.reset(new ValueType[simSize.productOfComponents()]);

            /*create box with valid memory*/
            auto dstBox = Box2D(PMacc::PitchedBox<ValueType, 2 > (
                                                       filteredData.get(),
                                                       Space2D(),
                                                       Space2D(simSize.x(), simSize.y()),
                                                       simSize.x() * sizeof (ValueType)
                                                       ));


            for (uint32_t i = 0; i < numRanks; ++i)
            {
                MessageHeader& head = headers[i];
                auto srcBox = Box2D(PMacc::PitchedBox<ValueType, 2 > (
                                                               fullData.get() + nodeSize.productOfComponents() * i,
                                                               Space2D(),
                                                               Space2D(head.nodeSize.x(), head.nodeSize.y()),
                                                               head.nodeSize.x() * sizeof (ValueType)
                                                               ));

                insertData(dstBox, srcBox, head.nodeOffset, head.nodePictureSize, head.nodeGuardCells);
            }

            return dstBox;
        }

        template<class DstBox, class SrcBox>
        void insertData(DstBox& dst, const SrcBox& src, Space offsetToSimNull, Space srcSize, Space nodeGuardCells)
        {
            for (int32_t y = 0; y < srcSize.y(); ++y)
            {
                for (int32_t x = 0; x < srcSize.x(); ++x)
                {
                    dst(Space2D(x + offsetToSimNull.x(), y + offsetToSimNull.y())) =
                        src(Space2D(nodeGuardCells.x() + x, nodeGuardCells.y() + y));
                }
            }
        }

    private:

        /* Master only */
        std::unique_ptr<ValueType[]> filteredData;
        std::unique_ptr<ValueType[]> fullData;
        std::unique_ptr<MessageHeader[]> headers;
        /* end Master only */

        MPI_Comm comm;
        Space simSize, nodeSize; /* Sizes of the simulation and the chunk on this node */
        bool isMaster;
        uint32_t numRanks;
        bool isMPICommInitialized;
    };

} //namespace xrt


