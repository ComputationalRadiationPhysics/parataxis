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
        template<class T_SrcBox, class T_Mapping, class T_HandleParticleFunctor>
        void operator()(int& counter, T_SrcBox srcBox, Space particleOffset, T_Mapping mapper, T_HandleParticleFunctor handleParticle)
        {
            Space gridDim = mapper.getGridDim();
            int gridSize = gridDim.productOfComponents();
            for(int linearBlockIdx = 0; linearBlockIdx < gridSize; ++linearBlockIdx)
            {
                Space blockIdx(PMacc::DataSpaceOperations<simDim>::map(gridDim, linearBlockIdx));

                typedef typename T_SrcBox::FrameType FrameType;
                typedef T_Mapping Mapping;
                typedef typename Mapping::SuperCellSize Block;

                FrameType *srcFramePtr;
                const int particlePerFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;

                const Space block = mapper.getSuperCellIndex(blockIdx);
                const Space superCellPosition((block - mapper.getGuardingSuperCells()) * mapper.getSuperCellSize());

                bool isValid;
                srcFramePtr = &(srcBox.getFirstFrame(block, isValid));

                while (isValid) //move over all Frames
                {
                    for (int threadIdx = 0; threadIdx < particlePerFrame; ++threadIdx)
                    {
                        auto parSrc = (*srcFramePtr)[threadIdx];
                        if (parSrc[PMacc::multiMask_] == 1)
                        {
                            /*calculate global cell index*/
                            Space localCell(PMacc::DataSpaceOperations<simDim>::template map<Block>(parSrc[PMacc::localCellIdx_]));
                            Space globalCellIdx = particleOffset + superCellPosition + localCell;
                            handleParticle(globalCellIdx, parSrc);
                            ++counter;
                        }
                    }
                    /*get next frame in supercell*/
                    srcFramePtr = &(srcBox.getNextFrame(*srcFramePtr, isValid));
                }
            }
        }
    };

} // namespace functors
} // namespace plugins
} // namespace xrt
