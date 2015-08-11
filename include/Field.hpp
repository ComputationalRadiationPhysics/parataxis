#pragma once

#include "xrtTypes.hpp"

#include <dimensions/DataSpaceOperations.hpp>
#include <mappings/kernel/AreaMapping.hpp>
#include <eventSystem/EventSystem.hpp>

#include "types.h"

namespace xrt {

    namespace kernel
    {
        template<class T_BoxWriteOnly, class T_Space, class T_Mapping, class T_Generator>
        __global__ void createDensityDistribution(T_BoxWriteOnly buffWrite, T_Space localDomainOffset, T_Mapping mapper, T_Generator generator)
        {
            /* get position in local domain in units of SuperCells for this block */
            const Space blockSuperCellIdx(mapper.getSuperCellIndex(Space(blockIdx)));
            /* convert position in unit of cells */
            const Space blockCellIdx = blockSuperCellIdx * T_Mapping::SuperCellSize::toRT();
            /* get offset to the blockCellIdx for this thread */
            const Space cellOffset(threadIdx);

            /* Calculate the global cellIdx by removing potential guard cells and adding the localDomainOffset*/
            const Space blockSuperCellOffset(mapper.getSuperCellIndex(Space()));
            const Space globalCellIdx = (blockSuperCellIdx - blockSuperCellOffset) * T_Mapping::SuperCellSize::toRT()
                                        + cellOffset + localDomainOffset;
            buffWrite(blockCellIdx + cellOffset) = generator(globalCellIdx);
        }
    }

    template<class T_MappingDesc>
    class Field
    {
        using MappingDesc = T_MappingDesc;

        MappingDesc mapping;
    public:
        void init(const MappingDesc & desc)
        {
            mapping = desc;
        }

        template<class T_Box, class T_Generator>
        void createDensityDistribution(T_Box&& writeBox, T_Generator&& generator)
        {
            PMacc::AreaMapping < PMacc::CORE + PMacc::BORDER, MappingDesc > mapper(mapping);
            __cudaKernel(kernel::createDensityDistribution)
                    (mapper.getGridDim(), MappingDesc::SuperCellSize::toRT().toDim3())
                    (
                     writeBox,
                     Environment::get().SubGrid().getLocalDomain().offset,
                     mapper,
                     generator);
        }
    };

}  // namespace xrt
