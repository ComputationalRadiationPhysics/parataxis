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
         * @param filter         Functor that is called with (Space globalCellIdx, Frame&, uint32_t partInFrameIdx).
         *                       If return value = false then particle is skipped
         * @param handleParticle Functor that is called with (Space globalCellIdx, Particle&)
         */
        template<class T_PartBox, class T_Mapping, class T_Filter, class T_HandleParticle>
        void operator()(uint32_t& counter, T_PartBox&& partBox, Space localOffset, T_Mapping mapper, T_Filter&& filter, T_HandleParticle&& handleParticle)
        {
            Space gridDim = mapper.getGridDim();
            int32_t gridSize = gridDim.productOfComponents();
            assert(gridSize >= 0); /* This is the size of 1 supercell in general, which is small enough */
            for(int32_t linearBlockIdx = 0; linearBlockIdx < gridSize; ++linearBlockIdx)
            {
                Space blockIdx(PMacc::DataSpaceOperations<simDim>::map(gridDim, linearBlockIdx));

                typedef typename T_PartBox::FrameType FrameType;

                const Space superCellIdx = mapper.getSuperCellIndex(blockIdx);
                const Space superCellPosition = (superCellIdx - mapper.getGuardingSuperCells()) * SuperCellSize::toRT() +
                                                localOffset;

                bool isValid;
                FrameType* framePtr = &(partBox.getFirstFrame(superCellIdx, isValid));

                while (isValid) //move over all Frames
                {
                    constexpr int32_t particlePerFrame = PMacc::math::CT::volume<SuperCellSize>::type::value;
                    for (int32_t threadIdx = 0; threadIdx < particlePerFrame; ++threadIdx)
                    {
                        auto particle = (*framePtr)[threadIdx];
                        if (particle[PMacc::multiMask_] == 1)
                        {
                            /*calculate global cell index*/
                            Space localCell(PMacc::DataSpaceOperations<simDim>::map<SuperCellSize>(particle[PMacc::localCellIdx_]));
                            Space globalCellIdx = superCellPosition + localCell;
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
