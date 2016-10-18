/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include "plugins/openPMD/ParticlePatches.hpp"

namespace parataxis {
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
}  // namespace parataxis
