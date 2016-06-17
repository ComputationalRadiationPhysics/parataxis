#pragma once

#include "xrtTypes.hpp"
#include "traits/PICToSplash.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

    namespace detail {

        template<typename T>
        struct GetValueType
        {
            using type = typename T::ValueType;
        };

        template<typename T>
        struct GetValueType<T*>
        {
            using type = T;
        };

    }  // namespace detail

    /** Functor for writing a domain to hdf5 */
    class SplashDomainReader
    {
    public:
        SplashDomainReader(splash::ParallelDomainCollector& hdfFile, int32_t id, const std::string& datasetName):
            hdfFile_(hdfFile), id_(id), datasetName_(datasetName){}
        /// Write the field
        /// @param data         Data that can be accessed via operator[unsigned]
        /// @param globalDomain Offset and Size of the field over all processes
        /// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data)
        template<typename T, typename T_ValueType = typename detail::GetValueType<T>::type>
        void operator()(T data, unsigned numDims, const splash::Domain& globalDomain, const splash::Domain& localDomain);
    private:
        splash::ParallelDomainCollector& hdfFile_;
        const int32_t id_;
        std::string datasetName_;
    };

    template<typename T, typename T_ValueType>
    void SplashDomainReader::operator()(const T data, unsigned numDims, const splash::Domain& globalDomain, const splash::Domain& localDomain)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: reading %4%D record %1% (globalDomain: %2%, localDomain: %3%")
                % datasetName_ % globalDomain.toString() % localDomain.toString() % numDims;

        typename traits::PICToSplash<T_ValueType>::type splashType;
        assert(isDomainValid(globalDomain, numDims));
        assert(isDomainValid(localDomain, numDims));

        // Validate dataset
        // sizeRead will be set
        splash::Dimensions sizeRead;
        splash::CollectionType* colType = hdfFile_.readMeta(
            id_,
            datasetName_.c_str(),
            globalDomain.getSize(),
            globalDomain.getOffset(),
            sizeRead);

        if(sizeRead != globalDomain.getSize())
            throw std::runtime_error("Invalid size read");

        if(colType->getDataType() != splashType.getDataType())
            throw std::runtime_error("Invalid data type");

        __delete(colType);

        // Read dataset
        splash::DomainCollector::DomDataClass data_class;
        splash::Domain readDomain = localDomain;
        readDomain.getOffset() += globalDomain.getOffset();
        splash::DataContainer *field_container = hdfFile_.readDomain(
            id_,
            datasetName_.c_str(),
            readDomain,
            &data_class);

        const T_ValueType* srcData = reinterpret_cast<const T_ValueType*>(field_container->getIndex(0)->getData());
        const size_t linearSize = localDomain.getSize().getScalarSize();
        for (size_t i = 0; i < linearSize; ++i)
            data[i] = srcData[i];

        delete field_container;
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
