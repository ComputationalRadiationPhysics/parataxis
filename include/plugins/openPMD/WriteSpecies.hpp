#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/WriteParticleAttribute.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include "plugins/common/helpers.hpp"
#include <compileTime/conversion/MakeSeq.hpp>
#include <compileTime/conversion/RemoveFromSeq.hpp>
#include <particles/Identifier.hpp>
#include <particles/ParticleDescription.hpp>
#include <particles/memory/boxes/TileDataBox.hpp>
#include <particles/operations/CountParticles.hpp>
#include <particles/operations/ConcatListOfFrames.hpp>
#include <algorithms/ForEach.hpp>
#include <forward.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/pair.hpp>

namespace xrt {
namespace plugins {
namespace openPMD {

 /** copy particle to host memory and dump to HDF5 file
 *
 * @tparam T_Species type of species
 *
 */
template<typename T_Species>
struct WriteSpecies
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

    template<class T_Writer>
    HINLINE void operator()(T_Writer& writer, const MappingDesc& cellDescription)
    {
        using namespace PMacc::algorithms::forEach;

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: (begin) write species: %1%") % Hdf5FrameType::getName();
        auto& dc = Environment::get().DataConnector();
        const auto& subGrid = Environment::get().SubGrid();
        /* load particle without copy particle data to host */
        T_Species& speciesTmp = dc.getData<T_Species >(FrameType::getName());

        /* count number of particles for this species on the device */
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) count particles: %1%") % Hdf5FrameType::getName();
        uint64_t numParticles = PMacc::CountParticles::countOnDevice< PMacc::CORE + PMacc::BORDER >(
            speciesTmp,
            cellDescription,
            NoFilter()
        );

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) count particles: %1% = %2%") % Hdf5FrameType::getName() % numParticles;

        Hdf5FrameType hostFrame;
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) malloc host memory: %1%") % Hdf5FrameType::getName();
        /*malloc memory on host*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, AllocMemory<bmpl::_1, ArrayAllocator> > mallocMem;
        mallocMem(PMacc::forward(hostFrame), numParticles);
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) malloc host memory: %1%") % Hdf5FrameType::getName();

        if (numParticles != 0)
        {
            PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) copy particle host (with hierarchy) to host (without hierarchy): %1% (%2%)")
                    % Hdf5FrameType::getName() % numParticles;

            PMacc::MallocMCBuffer& mallocMCBuffer = dc.getData<PMacc::MallocMCBuffer>(PMacc::MallocMCBuffer::getName());

            int globalParticleOffset = 0;
            PMacc::AreaMapping<PMacc::CORE + PMacc::BORDER, MappingDesc> mapper(cellDescription);

            PMacc::particles::operations::ConcatListOfFrames<simDim> concatListOfFrames(mapper.getGridDim());

            concatListOfFrames(
                                globalParticleOffset,
                                hostFrame,
                                speciesTmp.getHostParticlesBox(mallocMCBuffer.getOffset()),
                                NoFilter(),
                                subGrid.getLocalDomain().offset, /*relative to data domain (not to physical domain)*/
                                mapper
                                );

            dc.releaseData(PMacc::MallocMCBuffer::getName());

            assert(globalParticleOffset == numParticles);
            PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) copy particle host (with hierarchy) to host (without hierarchy): %1% (%2%)")
                    % Hdf5FrameType::getName() % globalParticleOffset;
        }

        /* We rather do an allgather at this point then letting libSplash
         * do an allgather during write to find out the global number of
         * particles.
         */
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) collect particle sizes for %1%") % Hdf5FrameType::getName();

        auto& gc = Environment::get().GridController();

        const uint64_t numRanks( gc.getGlobalSize() );
        const uint64_t myRank( gc.getGlobalRank() );

        /* For collective write calls we need the information:
         *   - how many particles will be written globally
         *   - what is my particle offset within this global data set
         *
         * interleaved in array:
         *   numParticles for mpi rank, mpi rank
         *
         * the mpi rank is an arbitrary quantity and might change after a
         * restart, but we only use it to order our patches and offsets
         */
        std::vector<uint64_t> particleCounts(2 * numRanks, 0u);
        uint64_t myParticlePatch[2];
        myParticlePatch[0] = numParticles;
        myParticlePatch[1] = myRank;

        /* we do the scan over MPI ranks since it does not matter how the
         * global rank or scalar position (which are not identical) are
         * ordered as long as the particle attributes are also written in
         * the same order (which is by global rank) */
        uint64_t numParticlesOffset = 0;
        uint64_t numParticlesGlobal = 0;

        MPI_CHECK(MPI_Allgather(
            myParticlePatch, 2, MPI_UINT64_T,
            &(*particleCounts.begin()), 2, MPI_UINT64_T,
            gc.getCommunicator().getMPIComm()
        ));

        for( uint64_t r = 0; r < numRanks; ++r )
        {
            numParticlesGlobal += particleCounts.at(2 * r);
            if( particleCounts.at(2 * r + 1) < myParticlePatch[ 1 ] )
                numParticlesOffset += particleCounts.at(2 * r);
        }
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) collect particle sizes for %1%") % Hdf5FrameType::getName();

        /* dump non-constant particle records to hdf5 file */
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) write particle records for %1% (%2%:%3%)")
                % Hdf5FrameType::getName() % numParticles % numParticlesOffset;

        writer.setCurrentDataset(std::string("particles/") + FrameType::getName());

        ForEach<typename Hdf5FrameType::ValueTypeSeq, hdf5::WriteParticleAttribute<bmpl::_1> > writeToHdf5;
        writeToHdf5(
            PMacc::forward(writer),
            PMacc::forward(hostFrame),
            numParticles,
            numParticlesOffset,
            numParticlesGlobal
        );

        /*free host memory*/
        ForEach<typename Hdf5FrameType::ValueTypeSeq, FreeMemory<bmpl::_1, ArrayAllocator> > freeMem;
        freeMem(PMacc::forward(hostFrame));

        /* write constant particle records to hdf5 file */
        // No macro particles -> weighting = 1
        writeConstantRecord(writer["weighting"], numParticlesGlobal, 1, 1, std::vector<float_64>(traits::NUnitDimension, 0));

        const float_64 chargeVal = GetResolvedFlag_t<FrameType, charge<>>::getValue();
        std::vector<float_64> chargeUnitDimension(traits::NUnitDimension, 0);
        chargeUnitDimension.at(traits::SIBaseUnits::time) = 1.0;
        chargeUnitDimension.at(traits::SIBaseUnits::electricCurrent) = 1.0;
        writeConstantRecord(writer["charge"], numParticlesGlobal, chargeVal, UNIT_CHARGE, chargeUnitDimension);

        const float_64 massVal = GetResolvedFlag_t<FrameType, mass<>>::getValue();
        std::vector<float_64> massUnitDimension(traits::NUnitDimension, 0);
        massUnitDimension.at(traits::SIBaseUnits::mass) = 1.0;
        writeConstantRecord(writer["mass"], numParticlesGlobal, massVal, UNIT_MASS, massUnitDimension);

        auto writeAttribute = writer.getAttributeWriter();
        writeAttribute("particleShape", float(0));
        writeAttribute("currentDeposition", "none");
        writeAttribute("particlePush", "other");
        writeAttribute("particleInterpolation", "uniform");
        writeAttribute("particleSmoothing", "none");

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) write particle records for %1%") % Hdf5FrameType::getName();

        /* write species particle patch meta information */
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  (begin) writing particlePatches for %1%") % Hdf5FrameType::getName();

        writer.setCurrentDataset(writer.getCurrentDataset() + "/particlePatches");

        /* offset and size of our particle patches
         *   - numPatches: we write as many patches as MPI ranks
         *   - myPatchOffset: we write in the order of the MPI ranks
         *   - myPatchEntries: every MPI rank writes exactly one patch
         */
        const splash::Dimensions numPatches( numRanks, 1, 1 );
        const splash::Domain localDomain = hdf5::makeSplashDomain<1>(myRank, 1);

        /* numParticles: number of particles in this patch */
        writer["numParticles"].getFieldWriter()(numParticles, numPatches, localDomain);

        /* numParticlesOffset: number of particles before this patch */
        writer["numParticlesOffset"].getFieldWriter()(numParticlesOffset, numPatches, localDomain);

        /* offset: absolute position where this particle patch begins including
         *         global domain offsets (slides), etc.
         * extent: size of this particle patch, upper bound is excluded
         */
        auto offsetWriter = writer["offset"];
        auto extentWriter = writer["extent"];
        const std::string name_lookup[] = {"x", "y", "z"};
        for (uint32_t d = 0; d < simDim; ++d)
        {
            const uint64_t patchOffset = subGrid.getGlobalDomain().offset[d] +
                                         subGrid.getLocalDomain().offset[d];
            const uint64_t patchExtent = subGrid.getLocalDomain().size[d];
            /* offsets and extent of the patch are positions (lengths)
             * and need to be scaled like the cell idx of a particle
             */
            std::vector<float_64> unitCellIdx = traits::OpenPMDUnit<PMacc::globalCellIdx<globalCellIdx_pic>, FrameType>::get();

            auto curWriter = offsetWriter[name_lookup[d]];
            curWriter.getFieldWriter()(patchOffset, numPatches, localDomain);
            curWriter.getAttributeWriter()("unitSI", unitCellIdx.at(d));

            curWriter = extentWriter[name_lookup[d]];
            curWriter.getFieldWriter()(patchExtent, numPatches, localDomain);
            curWriter.getAttributeWriter()("unitSI", unitCellIdx.at(d));
        }

        std::vector<float_64> unitDimensionCellIdx = traits::OpenPMDUnit<PMacc::globalCellIdx<globalCellIdx_pic>, FrameType>::getDimension();

        offsetWriter.getAttributeWriter()("unitDimension", unitDimensionCellIdx);
        extentWriter.getAttributeWriter()("unitDimension", unitDimensionCellIdx);


        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:  ( end ) writing particlePatches for %1%") % Hdf5FrameType::getName();

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: ( end ) writing species: %1%") % Hdf5FrameType::getName();
    }

private:
    /** Writes a constant particle record (weighted for a real particle)
     *
     * @param value of the record
     * @param unitSI conversion factor to SI
     * @param unitDimension power in terms of SI base units for this record
     */
    template<class T_Writer>
    static void writeConstantRecord(
        T_Writer&& writer,
        const uint64_t numParticles,
        const float_64 value,
        const float_64 unitSI,
        const std::vector<float_64>& unitDimension
    )
    {
        // Write a dummy field so we can add attributes (workaround for splash)
        auto& gc = Environment::get().GridController();
        writer["dummy"].getFieldWriter()(
            uint32_t(0),
            hdf5::makeSplashSize<1>(gc.getGlobalSize()),
            hdf5::makeSplashDomain<1>(gc.getGlobalRank(), 1)
            );

        auto writeAttribute = writer.getAttributeWriter();
        writeAttribute("value", value);
        writeAttribute("shape", std::array<uint64_t,1>{numParticles});
        writeAttribute("unitSI", unitSI);
        writeAttribute("unitDimension", unitDimension);
        writeAttribute("timeOffset", float_X(0));

        /* ED-PIC extension:
         *   - this is a record describing a *real* particle (0: false)
         *   - it needs to be scaled linearly (w^1.0) to get the *macro*
         *     particle record
         */
        writeAttribute("macroWeighted", uint32_t(0));
        writeAttribute("weightingPower", float_64(1));
    }
};

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
