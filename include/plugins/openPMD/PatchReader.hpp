#pragma once

#include "plugins/openPMD/ParticlePatches.hpp"

namespace xrt {
namespace plugins {
namespace openPMD {

struct PatchReader
{
    template<class T_SplashReader>
    ParticlePatches operator()(T_SplashReader&& reader, uint32_t numPatches);

private:
    template<class T_SplashReader>
    static void read(T_SplashReader&& reader, std::vector<uint64_t>& values, uint32_t numPatches);
};

template<class T_SplashReader>
ParticlePatches PatchReader::operator()(T_SplashReader&& reader, uint32_t numPatches)
{
    ParticlePatches patches(numPatches);

    auto curReader = reader["particlePatches"];

    read(curReader["numParticles"], patches.numParticles, numPatches);
    read(curReader["numParticlesOffset"], patches.numParticlesOffset, numPatches);

    const std::string name_lookup[] = {"x", "y", "z"};
    for (uint32_t d = 0; d < simDim; ++d)
    {
        read(curReader["offset/" + name_lookup[d]], patches.offsets[d], numPatches);
        read(curReader["extent/" + name_lookup[d]], patches.extents[d], numPatches);
    }

    return patches;
}

template<class T_SplashReader>
void PatchReader::read(T_SplashReader&& reader, std::vector<uint64_t>& values, uint32_t numPatches)
{
    // Note: We read ALL patches -> localSize == globalSize
    reader.getFieldReader()(&values.front(), 1,
            hdf5::makeSplashSize<1>(numPatches),
            hdf5::makeSplashDomain<1>(0, numPatches));
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
