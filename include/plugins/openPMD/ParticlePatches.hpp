#pragma once

#include <vector>
#include <array>

namespace xrt {
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
} // namespace xrt
