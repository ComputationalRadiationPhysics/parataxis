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

#include "parataxisTypes.hpp"
#include "version.hpp"
#include "plugins/common/stringHelpers.hpp"
#include "plugins/hdf5/SplashWriter.hpp"

namespace parataxis {
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
        writeGlobalAttribute("software", "PARATAXIS");
        std::stringstream softwareVersion;
        softwareVersion << PARATAXIS_VERSION_MAJOR << "."
                        << PARATAXIS_VERSION_MINOR << "."
                        << PARATAXIS_VERSION_PATCH;
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
}  // namespace parataxis
