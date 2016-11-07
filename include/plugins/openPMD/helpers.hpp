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

#include <stdexcept>
#include <string>
#include <cstdio>

namespace parataxis {
namespace plugins {
namespace openPMD {

struct Version{
    Version(const std::string& version){
        if(sscanf(version.c_str(), "%u.%u.%u", &data[0], &data[1], &data[2]) != static_cast<int>(data.size()))
            throw std::invalid_argument(version + " is not a valid version string");
    }
    Version(uint32_t major, uint32_t minor, uint32_t revision): data{major, minor, revision}
    {}

    std::string toString() const
    {
        return std::to_string(data[0]) + "." + std::to_string(data[1]) + "." +  std::to_string(data[2]);
    }

    std::array<uint32_t, 3> data;
};

bool operator<(const Version& lhs, const Version& rhs)
{
    for(unsigned i=0; i<lhs.data.size(); i++)
    {
        if(lhs.data[i] < rhs.data[i])
            return true;
    }
    return false;
}


/** Perform basic checks that the file is in correct openPMD format */
template<class T_SplashReader>
void validate(T_SplashReader&& reader, bool usePicExtension)
{
    auto readAttr = reader.getGlobalAttributeReader();
    std::string version, iterEnc;
    uint32_t ext;
    readAttr("openPMD", version);
    readAttr("openPMDextension", ext);
    readAttr("iterationEncoding", iterEnc);
    Version reqVersion(1, 0, 0);
    if(version < reqVersion)
        throw std::runtime_error(std::string("Invalid version. Expected ") + reqVersion.toString() + ", found " + version);
    if(usePicExtension && (ext & 1) != 1)
        throw std::runtime_error(std::string("ED-PIC extension not found, found ") + std::to_string(ext));
    if(iterEnc != "fileBased")
        throw std::runtime_error(std::string("Only file based iterationEncoding is supported. Found ") + iterEnc);
}

/** Return the base path for the current iteration */
template<class T_SplashReader>
std::string getBasePath(T_SplashReader&& reader)
{
    // ATM libSplash already uses 'data/<id>' as the base path. No way out of this...
    if(true)
        return "";
    else{
        std::string basePath = reader.getGlobalAttributeReader().readString("basePath");
        size_t placeholderPos;
        while((placeholderPos = basePath.find("%T")) != std::string::npos)
        {
            basePath.replace(placeholderPos, 2, std::to_string(reader.getId()));
        }
        return basePath;
    }
}

/** Return the meshes path for the current iteration */
template<class T_SplashReader>
std::string getMeshesPath(T_SplashReader&& reader)
{
    return getBasePath(reader) + reader.getGlobalAttributeReader().readString("meshesPath");
}

/** Return the particles path for the current iteration */
template<class T_SplashReader>
std::string getParticlesPath(T_SplashReader&& reader)
{
    return getBasePath(reader) + reader.getGlobalAttributeReader().readString("particlesPath");
}

/** Converts the time from HDF5 in simulation units */
template<class T_SplashReader, typename T_Type>
T_Type convertTime(T_SplashReader&& reader, const T_Type time)
{
    auto readAttr = reader(getBasePath(reader)).getAttributeReader();
    float_64 timeUnitSI;
    readAttr("timeUnitSI", timeUnitSI);
    return time * timeUnitSI / UNIT_TIME;
}

/** Return the time at which the current timestep is set */
template<class T_SplashReader>
float_X getTime(T_SplashReader&& reader)
{
    auto readAttr = reader(getBasePath(reader)).getAttributeReader();
    float_X time;
    readAttr("time", time);
    return convertTime(reader, time);
}

/** Return the length of the timestep */
template<class T_SplashReader>
float_X getTimestepLength(T_SplashReader&& reader)
{
    auto readAttr = reader(getBasePath(reader)).getAttributeReader();
    float_X dt;
    readAttr("dt", dt);
    return convertTime(reader, dt);
}

/** Returns the axis labels (fastest varying dimension first) of a record */
template<size_t T_dims, class T_SplashReader>
std::array<char, T_dims> getAxisLabels(T_SplashReader&& reader)
{
    const std::string dataOrder = reader.getAttributeReader().readString("dataOrder");
    // 1 Char label, 1 Char NULL terminator
    char axisLabels[T_dims * 2];
    reader.getAttributeReader()("axisLabels", T_dims, axisLabels, sizeof(axisLabels));
    std::array<char, T_dims> result;
    if(dataOrder == "C")
    {
        for(size_t i=0; i<T_dims; i++)
        {
            result[T_dims - i - 1] = axisLabels[i * 2];
            assert(axisLabels[i * 2 + 1] == '\0');
        }
    } else if(dataOrder == "F")
    {
        for(size_t i=0; i<T_dims; i++)
        {
            result[i] = axisLabels[i * 2];
            assert(axisLabels[i * 2 + 1] == '\0');
        }
    } else
        throw std::runtime_error(std::string("Unknown data order: ") + dataOrder);
    return result;
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace parataxis
