#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

    /** Functor for writing a field to hdf5 */
    class SplashFieldWriter
    {
    public:
        SplashFieldWriter(splash::IParallelDataCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Write the field
        /// @param data         Pointer to contiguous data
        /// @param numDims      Dimensionality of the data
        /// @param globalSize   Size of the data over all processes
        /// @param localDomain  Offset and Size of the data on the current process (Size must match extents of data)
        template<typename T>
        void operator()(const T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain);
        /// Write a scalar field (1D)
        /// @param data         Pointer to contiguous data
        /// @param globalSize   Size of the data over all processes
        /// @param localDomain  Offset and Size of the data on the current process (Size must match extents of data)
        template<typename T>
        void operator()(const T data, const splash::Dimensions& globalSize, const splash::Domain& localDomain);
    private:
        splash::IParallelDataCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    template<typename T>
    void SplashFieldWriter::operator()(const T* data, unsigned numDims, const splash::Dimensions& globalSize, const splash::Domain& localDomain)
    {
        PMacc::log<XRTLogLvl::DEBUG>("HDF5: writing %4%D record %1% (globalSize: %2%, localDomain: %3%")
                % datasetName_ % globalSize.toString() % localDomain.toString() % numDims;

        typename traits::PICToSplash<T>::type splashType;
        assert(isSizeValid(globalSize, numDims));
        assert(isDomainValid(localDomain, numDims));

        hdfFile_.write(id_,                     /* id == time step */
                       globalSize,              /* total size of dataset over all processes */
                       localDomain.getOffset(), /* write offset for this process */
                       splashType,              /* data type */
                       numDims,                 /* NDims spatial dimensionality of the field */
                       localDomain.getSize(),   /* data size of this process */
                       datasetName_.c_str(),
                       data);
    }

    template<typename T>
    void SplashFieldWriter::operator()(const T data, const splash::Dimensions& globalSize, const splash::Domain& localDomain)
    {
        (*this)(&data, 1, globalSize, localDomain);
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
