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

#include "parataxisTypes.hpp"

namespace parataxis {
namespace plugins {
namespace hdf5 {

/** Load attribute of a species from HDF5 checkpoint file
 *
 * @tparam T_Identifier identifier of species attribute
 */
template<typename T_Identifier>
struct LoadParticleAttribute
{

    /** read attributes from hdf5 file
     *
     * @param frame frame with all particles
     * @param numParticles number of particles in this patch
     * @param localParticlesOffset number of particles in this patch
     * @param numParticlesGlobal number of particles globally
     */
    template<class T_SplashReader, typename T_Frame>
    HINLINE void operator()(T_SplashReader& inReader,
                            T_Frame& frame,
                            const uint64_t numParticles,
                            const uint64_t localParticlesOffset,
                            const uint64_t numParticlesGlobal
    )
    {
        typedef typename Resolve_t<T_Identifier>::type ValueType;
        constexpr uint32_t numComponents = PMacc::traits::GetNComponents<ValueType>::value;

        PMacc::log<PARATAXISLogLvl::IN_OUT>("HDF5:  (begin) load species attribute: %1%") % T_Identifier::getName();
        T_SplashReader reader = inReader[traits::OpenPMDName<T_Identifier>::get()];

        typedef typename PMacc::traits::GetComponentsType<ValueType>::type ComponentValueType;
        ComponentValueType* tmpArray = numParticles ? new ComponentValueType[numParticles] : nullptr;

        const std::string name_lookup[] = {"x", "y", "z"};
        for (uint32_t d = 0; d < numComponents; d++)
        {
             // Read components only for >1D records
             auto tmpReader = (numComponents > 1) ? reader[name_lookup[d]] : reader;
             tmpReader.getFieldReader()(
                 tmpArray,
                 1,
                 makeSplashSize<1>(numParticlesGlobal),
                 makeSplashDomain<1>(localParticlesOffset, numParticles)
             );

             /* copy component from temporary array to array of structs */
             ValueType* dataPtr = frame.getIdentifier(T_Identifier()).getPointer();
             #pragma omp parallel for
             for(uint64_t i = 0; i < numParticles; ++i)
                 ((ComponentValueType*)dataPtr)[i * numComponents + d] = tmpArray[i];
        }
        __deleteArray(tmpArray);

        PMacc::log<PARATAXISLogLvl::IN_OUT>("HDF5:  ( end ) load species attribute: %1%") % T_Identifier::getName();
    }

};

}  // namespace hdf5
}  // namespace plugins
}  // namespace parataxis
