/**
 * Copyright 2015-2016 Axel Huebl, Alexander Grund
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

#include <vector>
#include <array>

namespace parataxis {
namespace plugins {
namespace openPMD {

    /** Struct for a list of particle patches
     *
     * Object for all particle patches.
     *
     * Class based on work by Axel Huebl
     * @see https://github.com/openPMD/openPMD-standard/blob/1.0.0/STANDARD.md#sub-group-for-each-particle-species
     */
    class ParticlePatches
    {
    public:
        std::vector<uint64_t> numParticles;
        std::vector<uint64_t> numParticlesOffset;

        std::array<std::vector<uint64_t>, 3> offsets;
        std::array<std::vector<uint64_t>, 3> extents;

        /** Fill-Constructor with n empty-sized patches
         *
         * @param n number of patches to store
         */
        ParticlePatches( const size_t n );

        /** Returns the number of patches
         */
        size_t size() const { return numParticles.size(); }

        std::string toString() const;
    };

    ParticlePatches::ParticlePatches( const size_t n ):
            numParticles(n), numParticlesOffset(n)
    {
        for(auto& offset: offsets)
            offset.resize(n);
        for(auto& extent: extents)
            extent.resize(n);
    }

    std::string ParticlePatches::toString() const
    {
        std::stringstream result;
        for(size_t i = 0; i < size(); i++)
        {
            result << "Patch #" << i << ": "
                   << numParticles[i] << ":" << numParticlesOffset[i] << " ("
                   << Space3D(extents[0][i], extents[1][i], extents[2][i]).toString() <<":"
                   << Space3D(offsets[0][i], offsets[1][i], offsets[2][i]).toString()
                   << ")\n";
        }
        return result.str();
    }

} // namespace openPMD
} // namespace plugins
} // namespace parataxis
