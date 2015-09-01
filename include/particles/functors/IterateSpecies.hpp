#pragma once

#include "xrtTypes.hpp"

#include <dimensions/DataSpaceOperations.hpp>

namespace xrt{
namespace particles{
namespace functors{

    /** Iterates over all particles of a species and calls a functor for each one
     *
     * @tparam T_Species type of species
     *
     */
    template< typename T_Species >
    struct IterateSpecies
    {
    public:
        /**
         *
         * @param counter        Reference to counter variable that is incremented for each handled particle
         * @param partBox        (Host-)Box of the species
         * @param localOffset    Local offset of the box
         * @param mapper         Mapping instance (e.g. AreaMapping)
         * @param filter         Functor that is called with (Space globalCellIdx, Frame&, int partInFrameIdx).
         *                       If return value = false then particle is skipped
         * @param handleParticle Functor that is called with (Space globalCellIdx, Particle&)
         */
        template<class T_PartBox, class T_Mapping, class T_Filter, class T_HandleParticle>
        void operator()(int& counter, T_PartBox&& partBox, Space localOffset, T_Mapping mapper, T_Filter&& filter, T_HandleParticle&& handleParticle)
        {
            Space gridDim = mapper.getGridDim();
            int gridSize = gridDim.productOfComponents();
            for(int linearBlockIdx = 0; linearBlockIdx < gridSize; ++linearBlockIdx)
            {
                Space blockIdx(PMacc::DataSpaceOperations<simDim>::map(gridDim, linearBlockIdx));

                typedef typename T_PartBox::FrameType FrameType;
                typedef T_Mapping Mapping;
                typedef typename Mapping::SuperCellSize Block;

                const Space block = mapper.getSuperCellIndex(blockIdx);
                const Space superCellPosition((block - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());

                bool isValid;
                FrameType* framePtr = &(partBox.getFirstFrame(block, isValid));

                while (isValid) //move over all Frames
                {
                    constexpr int particlePerFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
                    for (int threadIdx = 0; threadIdx < particlePerFrame; ++threadIdx)
                    {
                        auto particle = (*framePtr)[threadIdx];
                        if (particle[PMacc::multiMask_] == 1)
                        {
                            /*calculate global cell index*/
                            Space localCell(PMacc::DataSpaceOperations<simDim>::map<Block>(particle[PMacc::localCellIdx_]));
                            Space globalCellIdx = localOffset + superCellPosition + localCell;
                            if(filter(globalCellIdx, *framePtr, threadIdx))
                            {
                                handleParticle(globalCellIdx, particle);
                                ++counter;
                            }
                        }
                    }
                    /*get next frame in supercell*/
                    framePtr = &(partBox.getNextFrame(*framePtr, isValid));
                }
            }
        }
    };

} // namespace functors
} // namespace plugins
} // namespace xrt
