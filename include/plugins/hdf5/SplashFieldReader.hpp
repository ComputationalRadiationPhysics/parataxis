#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

    /** Functor for writing a domain to hdf5 */
    class SplashFieldReader
    {
    public:
        SplashFieldReader(splash::ParallelDataCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Write the field
        /// @param data         Pointer to contiguous data
        /// @param globalDomain Offset and Size of the field over all processes
        /// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data)
        template<typename T>
        void operator()(T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain);
    private:
        splash::ParallelDataCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    template<typename T>
    void SplashFieldReader::operator()(T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: reading %4%D record %1% (globalSize: %2%, localDomain: %3%")
                % datasetName_ % globalSize.toString() % localDomain.toString() % numDims;

        typename traits::PICToSplash<T>::type splashType;
        assert(isSizeValid(globalSize, numDims));
        assert(isDomainValid(localDomain, numDims));

        // Validate dataset
        // sizeRead will be set
        splash::Dimensions sizeRead;
        splash::CollectionType* colType = hdfFile_.readMeta(
            id_,
            datasetName_.c_str(),
            globalSize,
            splash::Dimensions(0, 0, 0),
            sizeRead);

        if(sizeRead != globalSize)
            throw std::runtime_error("Invalid size read");

        if(colType->getDataType() != splashType.getDataType())
            throw std::runtime_error("Invalid data type");

        __delete(colType);

        // Read dataset
        hdfFile_.read(
            id_,
            localDomain.getSize(),
            localDomain.getOffset(),
            datasetName_.c_str(),
            sizeRead,
            data);

        if(sizeRead != localDomain.getSize())
             throw std::runtime_error("Invalid local size read");
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
