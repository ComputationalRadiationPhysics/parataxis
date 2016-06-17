#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

    /** Functor for writing splash::PolyType data to hdf5 */
    class SplashPolyDataWriter
    {
    public:
        SplashPolyDataWriter(splash::IParallelDomainCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Write the field
        /// @param data          Pointer to contiguous data
        /// @param numDims       Number of dimensions of the data
        /// @param globalDomain  Offset and Size of the global domain (TODO: What is this?)
        /// @param globaDataSize Size of the data over all processes
        /// @param localDomain   Size of data to be written and offset into global data (Size must match extents of data)
        template<typename T>
        void operator()(const T* data, unsigned numDims,
                const splash::Domain& globalDomain,
                const splash::Dimensions& globalSize,
                const splash::Domain& localDomain);
    private:
        splash::IParallelDomainCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    template<typename T>
    void SplashPolyDataWriter::operator()(const T* data, unsigned numDims,
            const splash::Domain& globalDomain,
            const splash::Dimensions& globalSize,
            const splash::Domain& localDomain)
    {
        PMacc::log<XRTLogLvl::DEBUG>("HDF5: writing %5%D record %1% (globalDomain: %2%, globalSize: %3%, localDomain: %4%")
                % datasetName_ % globalDomain.toString() % globalSize.toString() % localDomain.toString() % numDims;

        typename traits::PICToSplash<T>::type splashType;
        assert(isSizeValid(globalSize, numDims));
        assert(isDomainValid(localDomain, numDims));

        hdfFile_.writeDomain(  id_,                                       /* id == time step */
                               globalSize,                                /* total size of dataset over all processes */
                               localDomain.getOffset(),                   /* write offset for this process */
                               splashType,                                /* data type */
                               numDims,                                   /* NDims spatial dimensionality of the field */
                               splash::Selection(localDomain.getSize()),  /* data size of this process */
                               datasetName_.c_str(),
                               globalDomain,
                               splash::DomainCollector::PolyType,
                               data);
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
