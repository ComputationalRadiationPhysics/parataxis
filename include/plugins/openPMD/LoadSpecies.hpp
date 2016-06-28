#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/LoadParticleAttribute.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include "plugins/openPMD/PatchReader.hpp"
#include "plugins/common/helpers.hpp"
#include <compileTime/conversion/MakeSeq.hpp>
#include <compileTime/conversion/RemoveFromSeq.hpp>
#include <particles/Identifier.hpp>
#include <particles/ParticleDescription.hpp>
#include <particles/operations/splitIntoListOfFrames.kernel>
#include <particles/memory/boxes/TileDataBox.hpp>
#include <algorithms/ForEach.hpp>
#include <forward.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>

namespace xrt {
namespace plugins {
namespace openPMD {

 /** Read particles from HDF5 file into device structures
 *
 * @tparam T_Species type of species
 *
 */
template<typename T_Species>
struct LoadSpecies
{
    typedef typename T_Species::FrameType FrameType;
    typedef typename FrameType::ParticleDescription ParticleDescription;
    typedef typename FrameType::ValueTypeSeq ParticleAttributeList;


    /* delete multiMask and localCellIdx in hdf5 particle*/
    typedef bmpl::vector<PMacc::multiMask, PMacc::localCellIdx> TypesToDelete;
    typedef typename PMacc::RemoveFromSeq<ParticleAttributeList, TypesToDelete>::type ParticleCleanedAttributeList;

    /* add globalCellIdx for hdf5 particle*/
    typedef typename PMacc::MakeSeq<
            ParticleCleanedAttributeList,
            PMacc::globalCellIdx<globalCellIdx_pic>
    >::type ParticleNewAttributeList;

    typedef typename PMacc::ReplaceValueTypeSeq<ParticleDescription, ParticleNewAttributeList>::type NewParticleDescription;

    typedef PMacc::Frame<OperatorCreateVectorBox, NewParticleDescription> Hdf5FrameType;

    template<class T_Reader>
    HINLINE void operator()(T_Reader& reader, const MappingDesc& cellDescription, const uint32_t restartChunkSize)
    {
        using namespace PMacc::algorithms::forEach;

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: (begin) read species: %1%") % FrameType::getName();
        reader.setCurrentDataset(std::string("particles/") + FrameType::getName());

        ParticlePatches particlePatches = PatchReader()(reader, Environment::get().GridController().getGlobalSize());
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  Loaded patches: %1%") % particlePatches.toString();


        const auto& subGrid = Environment::get().SubGrid();

        const Space patchOffset = subGrid.getLocalDomain().offset + subGrid.getGlobalDomain().offset;
        const Space patchExtent = subGrid.getLocalDomain().size;
        uint64_t patchNumParticles = 0;
        uint64_t patchParticleOffset = 0;
        uint64_t totalNumParticles2 = 0;

        for(size_t i = 0; i <particlePatches.size(); ++i)
            totalNumParticles2 += particlePatches.numParticles[i];

        for(size_t i = 0; i <particlePatches.size(); ++i)
        {
            bool exactlyMyPatch = true;

            for( uint32_t d = 0; d < simDim; ++d )
            {
                if(particlePatches.offsets[d][i] != (uint64_t)patchOffset[d])
                    exactlyMyPatch = false;
                if(particlePatches.extents[d][i] != (uint64_t)patchExtent[d])
                    exactlyMyPatch = false;
            }

            if(exactlyMyPatch)
            {
                PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  Found local patch: %1%") % i;
                patchNumParticles = particlePatches.numParticles[i];
                patchParticleOffset = particlePatches.numParticlesOffset[i];
                break;
            }
        }

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  Loading %1% particles from offset %2%") % patchNumParticles % patchParticleOffset;

        Hdf5FrameType hostFrame;
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  malloc mapped memory: %1%") % Hdf5FrameType::getName();
        /*malloc mapped memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, AllocMemory<bmpl::_1, MappedMemAllocator> > mallocMem;
        mallocMem(PMacc::forward(hostFrame), patchNumParticles);

        ForEach<typename Hdf5FrameType::ValueTypeSeq, hdf5::LoadParticleAttribute<bmpl::_1> > loadAttributes;
        loadAttributes(PMacc::forward(reader), PMacc::forward(hostFrame), patchNumParticles, patchParticleOffset, totalNumParticles2);

        if(patchNumParticles > 0)
        {
            PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  get mapped memory device pointer: %1%") % Hdf5FrameType::getName();
            /*load device pointer of mapped memory*/
            Hdf5FrameType deviceFrame;
            ForEach<typename Hdf5FrameType::ValueTypeSeq, GetDevicePtr<bmpl::_1> > getDevicePtr;
            getDevicePtr(PMacc::forward(deviceFrame), PMacc::forward(hostFrame));

            auto& dc = Environment::get().DataConnector();
            /* load particle without copy particle data to host */
            T_Species& speciesTmp = dc.getData<T_Species>(FrameType::getName(), true);

            dim3 block(PMacc::math::CT::volume<SuperCellSize>::type::value);

            /* counter is used to apply for work, count used frames and count loaded particles
             * [0] -> offset for loading particles
             * [1] -> number of loaded particles
             * [2] -> number of used frames
             *
             * all values are zero after initialization
             */
            PMacc::HostDeviceBuffer<uint32_t, 1> counterBuffer(PMacc::DataSpace<1>(3));

            constexpr uint32_t cellsInSuperCell = PMacc::math::CT::volume<SuperCellSize>::type::value;

            const uint32_t iterationsForLoad = ceil(float_64(patchNumParticles) / float_64(restartChunkSize));
            uint32_t leftOverParticles = patchNumParticles;

            for (uint32_t i = 0; i < iterationsForLoad; ++i)
            {
                /* only load a chunk of particles per iteration to avoid blow up of frame usage
                 */
                uint32_t currentChunkSize = std::min(leftOverParticles, restartChunkSize);
                PMacc::log<XRTLogLvl::IN_OUT>("HDF5:   load particles on device chunk offset=%1%; chunk size=%2%; left particles %3%") %
                    (i * restartChunkSize) % currentChunkSize % leftOverParticles;

                __cudaKernel(PMacc::particles::operations::splitIntoListOfFrames)
                    (ceil(float_64(currentChunkSize) / float_64(cellsInSuperCell)), cellsInSuperCell)
                    (counterBuffer.getDeviceBuffer().getDataBox(),
                     speciesTmp.getDeviceParticlesBox(), deviceFrame,
                     patchNumParticles,
                     subGrid.getLocalDomain().offset, /*relative to data domain (not to physical domain)*/
                     cellDescription
                     );
                speciesTmp.fillAllGaps();
                leftOverParticles -= currentChunkSize;
            }

            counterBuffer.deviceToHost();
            PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  wait for last processed chunk: %1%") % Hdf5FrameType::getName();
            __getTransactionEvent().waitForFinished();

            PMacc::log<XRTLogLvl::IN_OUT>("HDF5: used frames to load particles: %1%") % counterBuffer.getHostBuffer().getDataBox()[2];

            if (counterBuffer.getHostBuffer().getDataBox()[1] != patchNumParticles)
            {
                PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  error load species | counter is %1% but should %2%")
                        % counterBuffer.getHostBuffer().getDataBox()[1] % patchNumParticles;
            }
            assert(counterBuffer.getHostBuffer().getDataBox()[1] == patchNumParticles);

            /*free host memory*/
            ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<bmpl::_1, MappedMemAllocator> > freeMem;
            freeMem(PMacc::forward(hostFrame));
        }
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: ( end ) read species: %1%") % FrameType::getName();
    }
};

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
