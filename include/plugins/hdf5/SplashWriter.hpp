#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashAttributeWriter.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Wrapper for splash data collectors to store state and simply writing */
class SplashWriter
{
public:
    SplashWriter(splash::IParallelDataCollector& hdfFile, int32_t id):
        isParallelWriter(true), hdfFile_(hdfFile), id_(id){}

    SplashWriter(splash::SerialDataCollector& hdfFile, int32_t id):
        isParallelWriter(false), hdfFile_(hdfFile), id_(id){}

    void SetCurrentDataset(const std::string& name);

    SplashGlobalAttributeWriter GetGlobalAttributeWriter();
    SplashAttributeWriter GetAttributeWriter();
    SplashAttributeWriter GetAttributeWriter(const std::string& datasetName);

private:
    const bool isParallelWriter;
    splash::DataCollector& hdfFile_;
    const int32_t id_;
    std::string curDatasetName_;
};

void SplashWriter::SetCurrentDataset(const std::string& name)
{
    curDatasetName_ = name;
}

SplashGlobalAttributeWriter SplashWriter::GetGlobalAttributeWriter()
{
    return isParallelWriter ?
            SplashGlobalAttributeWriter(static_cast<splash::ParallelDataCollector&>(hdfFile_), id_) :
            SplashGlobalAttributeWriter(static_cast<splash::SerialDataCollector&>(hdfFile_), id_);
}

SplashAttributeWriter SplashWriter::GetAttributeWriter()
{
    return GetAttributeWriter(curDatasetName_);
}

SplashAttributeWriter SplashWriter::GetAttributeWriter(const std::string& datasetName)
{
    return SplashAttributeWriter(hdfFile_, id_, datasetName);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
