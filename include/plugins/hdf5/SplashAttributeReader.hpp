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
    splash::DCAttributeInfo* readImpl(const std::string& name);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

class SplashGlobalAttributeReader: public detail::SplashBaseAttributeReader<SplashGlobalAttributeReader>
{
public:
    SplashGlobalAttributeReader(splash::DataCollector& hdfFile, int32_t id, const std::string& dataSetName):
        hdfFile_(&hdfFile), id_(id), dataSetName_(dataSetName){}

private:
    friend struct detail::SplashBaseAttributeReader<SplashGlobalAttributeReader>;
    splash::DCAttributeInfo* readImpl(const std::string& name);

    splash::DataCollector* hdfFile_;
    int32_t id_;
    std::string dataSetName_;
};

splash::DCAttributeInfo* SplashAttributeReader::readImpl(const std::string& name)
{
    return hdfFile_->readAttributeMeta(id_, dataSetName_.empty() ? nullptr : dataSetName_.c_str(), name.c_str());
}

splash::DCAttributeInfo* SplashGlobalAttributeReader::readImpl(const std::string& name)
{
    return hdfFile_->readGlobalAttributeMeta(id_, name.c_str());
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
