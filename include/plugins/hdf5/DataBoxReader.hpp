#pragma once

#include "xrtTypes.hpp"
#include <memory/boxes/DataBoxDim1Access.hpp>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Reader that can read DataBoxes and can be specialized over the value type of the data box */
template<class T_ValueType>
struct DataBoxReader
{
    /// Read a field from a databox
    /// @param reader       Instance of SplashReader
    /// @param dataBox      Databox containing the data
    /// @param globalDomain Offset and Size of the field over all processes
    /// @param localSize    Size of the field on the current process (must match extents of data in the box)
    /// @param localOffset  Offset of the field of the current process
    template<class T_SplashReader, class T_DataBox, unsigned T_globalDims>
    void operator()(T_SplashReader& reader, const T_DataBox& dataBox,
            const PMacc::Selection<T_globalDims>& globalDomain,
            const PMacc::DataSpace<T_DataBox::Dim>& localSize,
            const PMacc::DataSpace<T_globalDims>& localOffset)
    {
        using ValueType = typename T_DataBox::ValueType;
        static_assert(std::is_same<T_ValueType, ValueType>::value, "Wrong type in dataBox");

        auto fullLocalSize = PMacc::DataSpace<T_globalDims>::create(1);
        for(unsigned i = 0; i < localSize.getDim(); i++)
            fullLocalSize[i] = localSize[i];

        typedef PMacc::DataBoxDim1Access<T_DataBox> D1Box;
        D1Box d1Access(dataBox, localSize);

        reader.GetDomainReader()(d1Access, T_globalDims, makeSplashDomain(globalDomain), makeSplashDomain(localOffset, fullLocalSize));
    }
};

/// Read a field to a databox
/// @param reader       Instance of SplashReader
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data in the box)
template<class T_SplashReader, class T_DataBox, unsigned T_dim>
void readDataBox(T_SplashReader&& reader, const T_DataBox& dataBox, const PMacc::Selection<T_dim>& globalDomain, const PMacc::Selection<T_dim>& localDomain)
{
    DataBoxReader<typename T_DataBox::ValueType> boxReader;
    boxReader(reader, dataBox, globalDomain, localDomain.size, localDomain.offset);
}

/// Read a field to a databox
/// @param reader       Instance of SplashReader
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localSize    Size of the field on the current process (must match extents of data in the box)
/// @param localOffset  Offset of the field of the current process
template<class T_SplashReader, class T_DataBox, unsigned T_globalDims>
void readDataBox(T_SplashReader&& reader, const T_DataBox& dataBox,
        const PMacc::Selection<T_globalDims>& globalDomain,
        const PMacc::DataSpace<T_DataBox::Dim>& localSize,
        const PMacc::DataSpace<T_globalDims>& localOffset)
{
    DataBoxReader<typename T_DataBox::ValueType> boxReader;
    boxReader(reader, dataBox, globalDomain, localSize, localOffset);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt

#include "plugins/hdf5/ComplexBoxReader.hpp"
