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
        SplashFieldWriter(splash::IParallelDomainCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Write the field
        /// @param data         Pointer to contiguous data
        /// @param globalDomain Offset and Size of the field over all processes
        /// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data)
        template<typename T>
        void operator()(const T* data, unsigned numDims, const splash::Domain& globalDomain, const splash::Domain& localDomain);
    private:
        splash::IParallelDomainCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    template<typename T>
    void SplashFieldWriter::operator()(const T* data, unsigned numDims, const splash::Domain& globalDomain, const splash::Domain& localDomain)
    {
        typename traits::PICToSplash<T>::type splashType;
        assert(isDomainValid(globalDomain, numDims));
        assert(isDomainValid(localDomain, numDims));

        hdfFile_.writeDomain(  id_,                                       /* id == time step */
                               globalDomain.getSize(),                    /* total size of dataset over all processes */
                               localDomain.getOffset(),                   /* write offset for this process */
                               splashType,                                /* data type */
                               numDims,                                   /* NDims spatial dimensionality of the field */
                               splash::Selection(localDomain.getSize()),  /* data size of this process */
                               datasetName_.c_str(),
                               globalDomain,
                               splash::DomainCollector::GridType,
                               data);
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
