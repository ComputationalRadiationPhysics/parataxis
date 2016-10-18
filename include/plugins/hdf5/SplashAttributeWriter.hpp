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
#include "plugins/hdf5/SplashBaseAttributeWriter.hpp"
#include <splash/splash.h>

namespace parataxis {
namespace plugins {
namespace hdf5 {

/** Functor for writing an attribute for a given dataset or (when dataSetName is empty) the current iteration */
class SplashAttributeWriter: public detail::SplashBaseAttributeWriter<SplashAttributeWriter>
{
public:
    SplashAttributeWriter(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(&hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeWriter<SplashAttributeWriter>;
    void writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value);
    void writeImpl(const splash::CollectionType& colType, const std::string& name,
            unsigned numDims, const splash::Dimensions& dims, const void* value);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

/** Functor for writing a global attribute */
class SplashGlobalAttributeWriter: public detail::SplashBaseAttributeWriter<SplashGlobalAttributeWriter>
{
public:
    SplashGlobalAttributeWriter(splash::IParallelDataCollector& hdfFile, int32_t id):
        isParallelWriter(true), hdfFile_(&hdfFile), id_(id){}

    SplashGlobalAttributeWriter(splash::SerialDataCollector& hdfFile, int32_t id):
        isParallelWriter(false), hdfFile_(&hdfFile), id_(id){}

private:
    friend struct detail::SplashBaseAttributeWriter<SplashGlobalAttributeWriter>;
    void writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value);
    void writeImpl(const splash::CollectionType& colType, const std::string& name,
            unsigned numDims, const splash::Dimensions& dims, const void* value);

    const bool isParallelWriter;
    splash::DataCollector* hdfFile_;
    int32_t id_;

    void write(const splash::CollectionType& type, const char *name, const void* buf);
};

void SplashAttributeWriter::writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value)
{
    hdfFile_->writeAttribute(id_, colType, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(), value);
}

void SplashAttributeWriter::writeImpl(const splash::CollectionType& colType, const std::string& name,
        unsigned numDims, const splash::Dimensions& dims, const void* value)
{
    hdfFile_->writeAttribute(id_, colType, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(),
                                        numDims, dims, value);
}

void SplashGlobalAttributeWriter::writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value)
{
    write(colType, name.c_str(), value);
}

void SplashGlobalAttributeWriter::write(const splash::CollectionType& colType, const char *name, const void* buf)
{
    // Wrapper due to non-uniform interface of libsplash
    if(isParallelWriter)
        static_cast<splash::IParallelDataCollector&>(*hdfFile_).writeGlobalAttribute(id_, colType, name, buf);
    else
        static_cast<splash::SerialDataCollector&>(*hdfFile_).writeGlobalAttribute(colType, name, buf);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace parataxis
