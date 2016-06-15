#pragma once

#include "xrtTypes.hpp"
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Reader that can read DataBoxes and can be specialized over the value type of the data box */
template<class T_ValueType>
struct DataBoxReader
{
    /// Write a field from a databox
    /// @param writer       Instance of SplashWriter
    /// @param dataBox      Databox containing the data
    /// @param globalDomain Offset and Size of the field over all processes
    /// @param localSize    Size of the field on the current process (must match extents of data in the box)
    /// @param localOffset  Offset of the field of the current process
    template<class T_SplashWriter, class T_DataBox, unsigned T_globalDims>
    void operator()(T_SplashWriter& writer, const T_DataBox& dataBox,
            const PMacc::Selection<T_globalDims>& globalDomain,
            const PMacc::DataSpace<T_DataBox::Dim>& localSize,
            const PMacc::DataSpace<T_globalDims>& localOffset)
    {
        using ValueType = typename T_DataBox::ValueType;
        static_assert(std::is_same<T_ValueType, ValueType>::value, "Wrong type in dataBox");

        // sizeRead will be set
        splash::Dimensions sizeRead;
        splash::CollectionType* colType = writer.GetDC().readMeta(
            writer.GetId(),
            writer.GetCurrentDataset().c_str(),
            makeSplashSize(globalDomain.size),
            makeSplashDomain(globalDomain).getOffset(),
            sizeRead);

        if(sizeRead != makeSplashSize(globalDomain.size))
            throw std::runtime_error("Invalid size read");

        if(colType->getDataType() != typename traits::PICToSplash<ValueType>::type().getDataType())
            throw std::runtime_error("Invalid data type");

        __delete(colType);

        auto fullLocalSize = PMacc::DataSpace<T_globalDims>::create(1);
        for(unsigned i = 0; i < localSize.getDim(); i++)
            fullLocalSize[i] = localSize[i];

        splash::DomainCollector::DomDataClass data_class;
        splash::DataContainer *field_container =
            writer.GetDC().readDomain(writer.GetId(),
                                      writer.GetCurrentDataset().c_str(),
                                      makeSplashDomain<T_globalDims>(globalDomain.offset + localOffset, fullLocalSize),
                                      &data_class);

        const size_t linearSize = localSize.productOfComponents();
        typedef PMacc::DataBoxDim1Access<T_DataBox> D1Box;
        D1Box d1Access(dataBox, localSize);

        const ValueType* data = reinterpret_cast<const ValueType*>(field_container->getIndex(0)->getData());
        for (size_t i = 0; i < linearSize; ++i)
            d1Access[i] = data[i];
        delete field_container;
    }
};

/// Write a field from a databox
/// @param writer       Instance of SplashWriter
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data in the box)
template<class T_SplashWriter, class T_DataBox, unsigned T_dim>
void readDataBox(T_SplashWriter&& writer, const T_DataBox& dataBox, const PMacc::Selection<T_dim>& globalDomain, const PMacc::Selection<T_dim>& localDomain)
{
    DataBoxReader<typename T_DataBox::ValueType> boxReader;
    boxReader(writer, dataBox, globalDomain, localDomain.size, localDomain.offset);
}

/// Write a field from a databox
/// @param writer       Instance of SplashWriter
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localSize    Size of the field on the current process (must match extents of data in the box)
/// @param localOffset  Offset of the field of the current process
template<class T_SplashWriter, class T_DataBox, unsigned T_globalDims>
void readDataBox(T_SplashWriter&& writer, const T_DataBox& dataBox,
        const PMacc::Selection<T_globalDims>& globalDomain,
        const PMacc::DataSpace<T_DataBox::Dim>& localSize,
        const PMacc::DataSpace<T_globalDims>& localOffset)
{
    DataBoxReader<typename T_DataBox::ValueType> boxReader;
    boxReader(writer, dataBox, globalDomain, localSize, localOffset);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt

#include "plugins/hdf5/ComplexBoxReader.hpp"
