/**
 * Copyright 2013-2016 Axel Huebl, Felix Schmitt, Rene Widera, Alexander Grund
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

#include "xrtTypes.hpp"
#include "traits/PICToOpenPMD.hpp"

namespace xrt {
namespace plugins {
namespace hdf5 {

/** write attribute of a particle to hdf5 file
 *
 * @tparam T_Identifier identifier of a particle record
 */
template<typename T_Identifier>
struct WriteParticleAttribute
{
    /** write attribute to hdf5 file
     *
     * @param frame frame with all particles
     * @param numParticles number of particles in this patch
     * @param localParticlesOffset number of particles in this patch
     * @param numParticlesGlobal number of particles globally
     */
    template<class T_SplashWriter, typename T_Frame>
    HINLINE void operator()(T_SplashWriter& inWriter,
                            T_Frame& frame,
                            const uint64_t numParticles,
                            const uint64_t localParticlesOffset,
                            const uint64_t numParticlesGlobal
    )
    {
        typedef typename Resolve_t<T_Identifier>::type ValueType;
        constexpr uint32_t numComponents = PMacc::traits::GetNComponents<ValueType>::value;

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:   (begin) write species attribute: %1%") % T_Identifier::getName();

        T_SplashWriter writer = inWriter[traits::OpenPMDName<T_Identifier>::get()];

        // get the SI scaling, dimensionality of the attribute
        std::vector<float_64> unit = traits::OpenPMDUnit<T_Identifier, T_Frame>::get();
        std::vector<float_64> unitDimension = traits::OpenPMDUnit<T_Identifier, T_Frame>::getDimension();

        assert(unit.size() == numComponents); // unitSI for each component
        assert(unitDimension.size() == traits::NUnitDimension); // seven openPMD base units

        const PMacc::Selection<simDim>& globalDomain = Environment::get().SubGrid().getGlobalDomain();

        typedef typename PMacc::traits::GetComponentsType<ValueType>::type ComponentValueType;
        ComponentValueType* tmpArray = new ComponentValueType[numParticles];

        const std::string name_lookup[] = {"x", "y", "z"};
        for (uint32_t d = 0; d < numComponents; d++)
        {
            ValueType* dataPtr = frame.getIdentifier(T_Identifier()).getPointer();
            #pragma omp parallel for
            for(uint64_t i = 0; i < numParticles; ++i)
                tmpArray[i] = ((ComponentValueType*)dataPtr)[i * numComponents + d];

            // Write components only for >1D records
            auto tmpWriter = (numComponents > 1) ? writer[name_lookup[d]] : writer;
            tmpWriter.getPolyDataWriter()(
                tmpArray,
                1,
                makeSplashDomain(globalDomain),
                makeSplashSize<1>(numParticlesGlobal),
                makeSplashDomain<1>(localParticlesOffset, numParticles)
            );
            tmpWriter.getAttributeWriter()("unitSI", unit.at(d));
        }
        __deleteArray(tmpArray);

        auto writeAttribute = writer.getAttributeWriter();

        writeAttribute("unitDimension", unitDimension);
        writeAttribute("timeOffset", float_X(0));
        // Single particles
        writeAttribute("macroWeighted", uint32_t(0));
        writeAttribute("weightingPower", float_64(1));

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5:   ( end ) write species attribute: %1%") % T_Identifier::getName();
    }

};

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
