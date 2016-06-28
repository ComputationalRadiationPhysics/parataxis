#pragma once

#include "xrtTypes.hpp"
#include "version.hpp"
#include "plugins/common/stringHelpers.hpp"
#include "plugins/hdf5/SplashWriter.hpp"

namespace xrt {
namespace plugins {
namespace openPMD {

template<class T_DataCollector>
struct WriteHeader
{
    WriteHeader(hdf5::SplashWriter<T_DataCollector>& writer): writer_(writer){}

    void operator()(const std::string& fileNameBase, bool usePIC_ED_Ext = false)
    {
        auto writeGlobalAttribute = writer_.getGlobalAttributeWriter();
        /* openPMD attributes */
        /*   required */
    	writeGlobalAttribute("openPMD", "1.0.0");
        writeGlobalAttribute("openPMDextension", uint32_t(usePIC_ED_Ext ? 1 : 0)); // ED-PIC ID
        writeGlobalAttribute("basePath", "/data/%T/");
        writeGlobalAttribute("meshesPath", usePIC_ED_Ext ? "fields/" : "meshes/");
        writeGlobalAttribute("particlesPath", "particles/");
        writeGlobalAttribute("iterationEncoding", "fileBased");
        writeGlobalAttribute("iterationFormat", boost::filesystem::basename(fileNameBase) + std::string("_%T.h5"));

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
        auto writeAttribute = writer_("").getAttributeWriter();
        writeAttribute("dt", DELTA_T);
        writeAttribute("time", float_X(Environment::get().SimulationDescription().getCurrentStep()) * DELTA_T);
        writeAttribute("timeUnitSI", UNIT_TIME);

        if(usePIC_ED_Ext)
        {
            writeAttribute = writer_("fields").getAttributeWriter();
            writeAttribute("fieldSolver", "none");
            writeAttribute("fieldBoundary", "open\0open\0open\0open\0open\0open\0", 2 * simDim);
            writeAttribute("particleBoundary", "absorbing\0absorbing\0absorbing\0absorbing\0absorbing\0absorbing\0", 2 * simDim);
            writeAttribute("currentSmoothing", "none");
            writeAttribute("chargeCorrection", "none");
        }
    }
private:

    hdf5::SplashWriter<T_DataCollector>& writer_;
};

template<class T_DataCollector>
void writeHeader(hdf5::SplashWriter<T_DataCollector>& writer, const std::string& fileNameBase, bool usePIC_ED_Ext = false)
{
    WriteHeader<T_DataCollector> writeHeader(writer);
    writeHeader(fileNameBase, usePIC_ED_Ext);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
