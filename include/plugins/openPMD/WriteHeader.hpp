#pragma once

#include "xrtTypes.hpp"
#include "version.hpp"
#include "plugins/common/stringHelpers.hpp"
#include "plugins/hdf5/SplashWriter.hpp"

namespace xrt {
namespace plugins {
namespace openPMD {

struct WriteHeader
{
    WriteHeader(hdf5::SplashWriter& writer): writer_(writer){}

    void operator()(const std::string& fileNameBase, bool usePIC_ED_Ext = false)
    {
        auto writeGlobalAttribute = writer_.GetGlobalAttributeWriter();
        /* openPMD attributes */
        /*   required */
    	writeGlobalAttribute("openPMD", "1.0.0");
        writeGlobalAttribute("openPMDextension", uint32_t(usePIC_ED_Ext ? 1 : 0)); // ED-PIC ID
        writeGlobalAttribute("basePath", "/data/%T/");
        writeGlobalAttribute("meshesPath", "meshes/");
        writeGlobalAttribute("particlesPath", "particles/");
        writeGlobalAttribute("iterationEncoding", "fileBased");
        writeGlobalAttribute("iterationFormat", fileNameBase + std::string("_%T.h5"));

        /*   recommended */
        std::string author = Environment::get().SimulationDescription().getAuthor();
        if(!author.empty())
        	writeGlobalAttribute("author", author);
        writeGlobalAttribute("software", "XRT");
        std::stringstream softwareVersion;
        softwareVersion << XRT_VERSION_MAJOR << "."
                        << XRT_VERSION_MINOR << "."
                        << XRT_VERSION_PATCH;
        writeGlobalAttribute("softwareVersion", softwareVersion.str());
        writeGlobalAttribute("date", common::getDateString("%F %T %z"));

        /* openPMD: required time attributes */
        auto writeAttribute = writer_.GetAttributeWriter("");
        writeAttribute("dt", DELTA_T);
        writeAttribute("time", float_X(Environment::get().SimulationDescription().getCurrentStep()) * DELTA_T);
        writeAttribute("timeUnitSI", UNIT_TIME);
    }
private:

    hdf5::SplashWriter& writer_;
};

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
