#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashBaseAttributeReader.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Functor for reading an attribute for a given dataset or (when dataSetName is empty) the current iteration */
class SplashAttributeReader: public detail::SplashBaseAttributeReader<SplashAttributeReader>
{
public:
    SplashAttributeReader(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(&hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeReader<SplashAttributeReader>;
    void readImpl(const splash::CollectionType& colType, const std::string& name, void* value);
    void readImpl(const splash::CollectionType& colType, const std::string& name,
            unsigned numDims, const splash::Dimensions& dims, void* value);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

void SplashAttributeReader::readImpl(const splash::CollectionType& colType, const std::string& name, void* value)
{
    // TODO: check type (currently impossible)
    hdfFile_->readAttribute(id_, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(), value);
}

void SplashAttributeReader::readImpl(const splash::CollectionType& colType, const std::string& name,
        unsigned numDims, const splash::Dimensions& dims, void* value)
{
    // TODO: check type/dimensions (currently impossible)
    hdfFile_->readAttribute(id_, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str(), value);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
