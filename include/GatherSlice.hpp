#pragma once

#include "xrtTypes.hpp"

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

            int countRanks = Environment::get().GridController().getGpuNodes().productOfComponents();
            std::vector<int> gatherRanks(countRanks); /* rank in WORLD_GROUP or -1 if inactive */
            std::vector<int> groupRanks(countRanks);  /* rank in new group */
            int mpiRank = Environment::get().GridController().getGlobalRank();
            if (!isActive)
                mpiRank = -1;

            MPI_CHECK(MPI_Allgather(&mpiRank, 1, MPI_INT, &gatherRanks[0], 1, MPI_INT, MPI_COMM_WORLD));
            numRanks = 0;
            for (int i = 0; i < countRanks; ++i)
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

            for(int i=0; i<numRanks; ++i)
            {
                if(headers[i].nodeSize != nodeSize)
                    throw std::runtime_error("NodeSizes must be the same on all nodes");
            }
            return true;
        }

        template<class Box >
        Box operator()(Box data)
        {
            static_assert(std::is_same<typename Box::ValueType, ValueType>::value, "Wrong type");

            if (!fullData && isMaster)
                fullData.reset(new ValueType[nodeSize.productOfComponents() * numRanks]);

            const size_t elementsCount = nodeSize.productOfComponents() * sizeof (ValueType);

            MPI_CHECK(MPI_Gather(data.getPointer(), elementsCount, MPI_CHAR,
                                 fullData.get(), elementsCount, MPI_CHAR,
                                 0, comm));

            if(!isMaster)
            {
                return Box(PMacc::PitchedBox<ValueType, SIMDIM > (
                            nullptr,
                            Space(),
                            simSize,
                            simSize.x() * sizeof (ValueType)
                        ));
            }

            if (!filteredData)
                filteredData.reset(new ValueType[simSize.productOfComponents()]);

            /*create box with valid memory*/
            Box dstBox = Box(PMacc::PitchedBox<ValueType, SIMDIM > (
                                                       filteredData.get(),
                                                       Space(),
                                                       simSize,
                                                       simSize.x() * sizeof (ValueType)
                                                       ));


            for (int i = 0; i < numRanks; ++i)
            {
                MessageHeader& head = headers[i];
                Box srcBox = Box(PMacc::PitchedBox<ValueType, SIMDIM > (
                                                               fullData.get() + nodeSize.productOfComponents() * i,
                                                               Space(),
                                                               head.nodeSize,
                                                               head.nodeSize.x() * sizeof (ValueType)
                                                               ));

                insertData(dstBox, srcBox, head.nodeOffset, head.nodePictureSize, head.nodeGuardCells);
            }

            return dstBox;
        }

        template<class DstBox, class SrcBox>
        void insertData(DstBox& dst, const SrcBox& src, Space offsetToSimNull, Space srcSize, Space nodeGuardCells)
        {
            for (int y = 0; y < srcSize.y(); ++y)
            {
                for (int x = 0; x < srcSize.x(); ++x)
                {
                    dst(Space(x + offsetToSimNull.x(), y + offsetToSimNull.y())) =
                        src(Space(nodeGuardCells.x() + x, nodeGuardCells.y() + y));
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
        int numRanks;
        bool isMPICommInitialized;
    };

} //namespace xrt


