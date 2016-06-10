#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashAttributeWriter.hpp"
#include "plugins/hdf5/SplashFieldWriter.hpp"

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Wrapper for splash data collectors to store state and simplify writing */
template<class T_DataCollector>
class SplashWriter
{
public:
    SplashWriter(T_DataCollector& hdfFile, int32_t id):
        hdfFile_(hdfFile), id_(id){}

    void SetCurrentDataset(const std::string& name);
    std::string GetCurrentDataset() const { return curDatasetName_; }

    SplashGlobalAttributeWriter GetGlobalAttributeWriter();
    SplashAttributeWriter GetAttributeWriter();
    SplashAttributeWriter GetAttributeWriter(const std::string& datasetName);
    SplashFieldWriter GetFieldWriter();

private:
    T_DataCollector& hdfFile_;
    const int32_t id_;
    std::string curDatasetName_;
};

template<class T_DataCollector>
SplashWriter<T_DataCollector> makeSplashWriter(T_DataCollector& hdfFile, int32_t id)
{
    return SplashWriter<T_DataCollector>(hdfFile, id);
}

template<class T_DataCollector>
void SplashWriter<T_DataCollector>::SetCurrentDataset(const std::string& name)
{
    curDatasetName_ = name;
}

template<class T_DataCollector>
SplashGlobalAttributeWriter SplashWriter<T_DataCollector>::GetGlobalAttributeWriter()
{
    return SplashGlobalAttributeWriter(hdfFile_, id_);
}

template<class T_DataCollector>
SplashAttributeWriter SplashWriter<T_DataCollector>::GetAttributeWriter()
{
    return GetAttributeWriter(curDatasetName_);
}

template<class T_DataCollector>
SplashAttributeWriter SplashWriter<T_DataCollector>::GetAttributeWriter(const std::string& datasetName)
{
    return SplashAttributeWriter(hdfFile_, id_, datasetName);
}

template<class T_DataCollector>
SplashFieldWriter SplashWriter<T_DataCollector>::GetFieldWriter()
{
    return SplashFieldWriter(hdfFile_, id_, curDatasetName_);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
