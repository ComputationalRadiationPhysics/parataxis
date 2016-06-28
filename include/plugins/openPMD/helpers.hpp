#pragma once

#include <stdexcept>
#include <string>

namespace xrt {
namespace plugins {
namespace openPMD {

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
    if(version != "1.0.0")
        throw std::runtime_error(std::string("Invalid version. Expected 1.0.0, found ") + version);
    if(usePicExtension && (ext & 1) != 1)
        throw std::runtime_error(std::string("ED-PIC extension not found, found ") + std::to_string(ext));
    if(iterEnc != "fileBased")
        throw std::runtime_error(std::string("Only file based iterationEncoding is supported. Found ") + iterEnc);
}

/** Return the base path for the current iteration */
template<class T_SplashReader>
std::string getBasePath(T_SplashReader&& reader)
{
    std::string basePath = reader.getGlobalAttributeReader().readString("basePath");
    size_t placeholderPos;
    while((placeholderPos = basePath.find("%T")) != std::string::npos)
    {
        basePath.replace(placeholderPos, 2, std::to_string(reader.getId()));
    }
    return basePath;
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

/** Return the time at which the current timestep is set */
template<class T_SplashReader>
float_X getTime(T_SplashReader&& reader)
{
    auto readAttr = reader(getBasePath(reader)).getAttributeReader();
    float_64 timeUnitSI;
    float_X time;
    readAttr("time", time);
    readAttr("timeUnitSI", timeUnitSI);
    return time * UNIT_TIME/timeUnitSI;
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
