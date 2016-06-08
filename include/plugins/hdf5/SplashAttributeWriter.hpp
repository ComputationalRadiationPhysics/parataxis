#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashBaseAttributeWriter.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Functor for writing an attribute for a given dataset or (when dataSetName is empty) the current iteration */
class SplashAttributeWriter: public detail::SplashBaseAttributeWriter<SplashAttributeWriter>
{
public:
    SplashAttributeWriter(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeWriter<SplashAttributeWriter>;
    void writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value);
    void writeImpl(const splash::CollectionType& colType, const std::string& name,
            unsigned numDims, const splash::Dimensions& dims, const void* value);

    splash::DataCollector& hdfFile_;
    const int32_t id_;
    std::string dataSetName_;
};

/** Functor for writing a global attribute */
class SplashGlobalAttributeWriter: public detail::SplashBaseAttributeWriter<SplashGlobalAttributeWriter>
{
public:
    SplashGlobalAttributeWriter(splash::IParallelDataCollector& hdfFile, int32_t id):
        isParallelWriter(true), hdfFile_(hdfFile), id_(id){}

    SplashGlobalAttributeWriter(splash::SerialDataCollector& hdfFile, int32_t id):
        isParallelWriter(false), hdfFile_(hdfFile), id_(id){}

private:
    friend struct detail::SplashBaseAttributeWriter<SplashGlobalAttributeWriter>;
    void writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value);
    void writeImpl(const splash::CollectionType& colType, const std::string& name,
            unsigned numDims, const splash::Dimensions& dims, const void* value);

    const bool isParallelWriter;
    splash::DataCollector& hdfFile_;
    const int32_t id_;

    void write(const splash::CollectionType& type, const char *name, const void* buf);
};

void SplashAttributeWriter::writeImpl(const splash::CollectionType& colType, const std::string& name, const void* value)
{
    hdfFile_.writeAttribute(id_, colType, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(), value);
}

void SplashAttributeWriter::writeImpl(const splash::CollectionType& colType, const std::string& name,
        unsigned numDims, const splash::Dimensions& dims, const void* value)
{
    hdfFile_.writeAttribute(id_, colType, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(),
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
        static_cast<splash::IParallelDataCollector&>(hdfFile_).writeGlobalAttribute(id_, colType, name, buf);
    else
        static_cast<splash::SerialDataCollector&>(hdfFile_).writeGlobalAttribute(colType, name, buf);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
