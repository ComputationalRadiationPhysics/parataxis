#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashAttributeWriter.hpp"
#include "plugins/hdf5/SplashFieldWriter.hpp"
#include "plugins/hdf5/SplashDomainWriter.hpp"
#include "plugins/hdf5/SplashPolyDataWriter.hpp"
#include "plugins/hdf5/SplashAttributeReader.hpp"
#include "plugins/hdf5/SplashFieldReader.hpp"
#include "plugins/hdf5/SplashDomainReader.hpp"

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Wrapper for splash data collectors to store state and simplify writing/reading */
template<class T_DataCollector>
class SplashWriter
{
public:
    SplashWriter(T_DataCollector& hdfFile, int32_t id):
        hdfFile_(&hdfFile), id_(id){}
    SplashWriter(const SplashWriter&) = default;
    SplashWriter() = delete;
    ~SplashWriter(){}

    void setCurrentDataset(const std::string& name);
    std::string getCurrentDataset() const { return curDatasetName_; }

    // Create a copy with a given dataset
    SplashWriter operator()(const std::string& name);
    // Create a copy with the dataset appended (separated by "/")
    SplashWriter operator[](const std::string& name);

    SplashGlobalAttributeWriter getGlobalAttributeWriter();
    SplashAttributeWriter getAttributeWriter();
    SplashFieldWriter getFieldWriter();
    SplashDomainWriter getDomainWriter();
    SplashPolyDataWriter getPolyDataWriter();

    SplashAttributeReader getAttributeReader();
    SplashGlobalAttributeReader getGlobalAttributeReader();
    SplashFieldReader getFieldReader();
    SplashDomainReader getDomainReader();

    T_DataCollector& getDC(){ return *hdfFile_; }
    int32_t getId() const { return id_; }
    void setId(int32_t id) { id_ = id; }
private:

    T_DataCollector* hdfFile_;
    int32_t id_;
    std::string curDatasetName_;
};

template<class T_DataCollector>
SplashWriter<T_DataCollector> makeSplashWriter(T_DataCollector& hdfFile, int32_t id)
{
    return SplashWriter<T_DataCollector>(hdfFile, id);
}

template<class T_DataCollector>
void SplashWriter<T_DataCollector>::setCurrentDataset(const std::string& name)
{
    curDatasetName_ = name;
}

template<class T_DataCollector>
SplashGlobalAttributeWriter SplashWriter<T_DataCollector>::getGlobalAttributeWriter()
{
    return SplashGlobalAttributeWriter(*hdfFile_, id_);
}

template<class T_DataCollector>
SplashWriter<T_DataCollector> SplashWriter<T_DataCollector>::operator()(const std::string& name)
{
    SplashWriter<T_DataCollector> result(*hdfFile_, id_);
    result.setCurrentDataset(name);
    return result;
}

template<class T_DataCollector>
SplashWriter<T_DataCollector> SplashWriter<T_DataCollector>::operator[](const std::string& name)
{
    SplashWriter<T_DataCollector> result(*hdfFile_, id_);
    if(curDatasetName_.empty())
        result.curDatasetName_ = name;
    else
        result.curDatasetName_ = curDatasetName_ + "/" + name;
    return result;
}

////////////////////////////////////////////////////////////////////////////////
/// Write methods
////////////////////////////////////////////////////////////////////////////////

template<class T_DataCollector>
SplashAttributeWriter SplashWriter<T_DataCollector>::getAttributeWriter()
{
    return SplashAttributeWriter(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashFieldWriter SplashWriter<T_DataCollector>::getFieldWriter()
{
    return SplashFieldWriter(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashDomainWriter SplashWriter<T_DataCollector>::getDomainWriter()
{
    return SplashDomainWriter(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashPolyDataWriter SplashWriter<T_DataCollector>::getPolyDataWriter()
{
    return SplashPolyDataWriter(*hdfFile_, id_, curDatasetName_);
}

////////////////////////////////////////////////////////////////////////////////
/// Read methods
////////////////////////////////////////////////////////////////////////////////

template<class T_DataCollector>
SplashGlobalAttributeReader SplashWriter<T_DataCollector>::getGlobalAttributeReader()
{
    return SplashGlobalAttributeReader(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashAttributeReader SplashWriter<T_DataCollector>::getAttributeReader()
{
    return SplashAttributeReader(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashFieldReader SplashWriter<T_DataCollector>::getFieldReader()
{
    return SplashFieldReader(*hdfFile_, id_, curDatasetName_);
}

template<class T_DataCollector>
SplashDomainReader SplashWriter<T_DataCollector>::getDomainReader()
{
    return SplashDomainReader(*hdfFile_, id_, curDatasetName_);
}

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
