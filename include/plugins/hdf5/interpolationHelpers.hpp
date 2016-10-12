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

#include "plugins/hdf5/SplashWriter.hpp"
#include "plugins/openPMD/helpers.hpp"

namespace xrt {
namespace plugins {
namespace hdf5 {

    /**
     * Find the smallest timestep in a set of HDF5 files so that the HDF5-time is at least \ref curTime
     * @param dataCollector Initialized dataCollector
     * @param curTime Time to look for
     * @param maxTimestep Optional output of maximum timestep of the HDF5 file set
     * @param startTimestep Optional HDF5 timestep to start searching. Might increase search speed
     * @return HDF5 timestep/ID so that its time is <= curTime and the next time (if it exists) is > curTime
     */
    template<class T_DataCollector, typename T_TimeType>
    uint32_t findTimestep(T_DataCollector& dataCollector, T_TimeType curTime, uint32_t* maxTimestep = nullptr, uint32_t startTimestep = 0)
    {
        namespace openPMD = plugins::openPMD;
        // Open at first timestep and validate file
        auto reader = plugins::hdf5::makeSplashWriter(dataCollector, startTimestep);
        uint32_t maxHDF5Timestep = dataCollector.getMaxID();
        openPMD::validate(reader, false);
        // Get the current delta t
        auto readAttr = reader(openPMD::getBasePath(reader)).getAttributeReader();
        float_64 timeUnitSI;
        float_X dt;
        readAttr("dt", dt);
        readAttr("timeUnitSI", timeUnitSI);
        dt *= UNIT_TIME/timeUnitSI;
        // Get first time
        float_X time = openPMD::getTime(reader);
        // This time should be smaller, than our time.
        // We still consider it valid, if the difference is at most 1 dt (in which case we extrapolate)
        if(time > curTime + dt)
            throw std::runtime_error(std::string("Cannot load field as first entry is at ") +
                    std::to_string(time / UNIT_TIME) + "s but requested time is " +
                    std::to_string(curTime / UNIT_TIME) + "s");
        else if(time < curTime)
        {
            // We increase the HDF5 time till it is >= our time.
            // Then stepping back takes as to the very last timestep before curTime
            // First fast forward assuming equal distance times
            float_X  diff = curTime - time;
            reader.setId(std::min<uint32_t>(ceil(diff / dt), maxHDF5Timestep));
            // Now single steps
            while(openPMD::getTime(reader) < curTime && reader.getId() < maxHDF5Timestep)
                reader.setId(reader.getId() + 1);
            // And now backward till H5time <= curTime (guaranteed to terminate as for id=0 'time < curTime' is checked
            while(openPMD::getTime(reader) > curTime)
                reader.setId(reader.getId() - 1);
        }
        if(maxTimestep)
            *maxTimestep = maxHDF5Timestep;
        return reader.getId();
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
