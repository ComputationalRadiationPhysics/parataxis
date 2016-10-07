/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#pragma once

#include <memory/boxes/DataBoxDim1Access.hpp>

namespace xrt {
namespace plugins {
namespace hdf5 {

/** Writer that can write DataBoxes and can be specialized over the value type of the data box */
template<class T_ValueType>
struct DataBoxWriter
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
        const size_t tmpArraySize = localSize.productOfComponents();
        std::unique_ptr<ValueType[]> tmpArray(new ValueType[tmpArraySize]);

        typedef PMacc::DataBoxDim1Access<T_DataBox> D1Box;
        D1Box d1Access(dataBox, localSize);

        /* copy data to temp array
         * tmpArray has the size of the data without any offsets
         */
        for (size_t i = 0; i < tmpArraySize; ++i)
            tmpArray[i] = d1Access[i];

        auto fullLocalSize = PMacc::DataSpace<T_globalDims>::create(1);
        for(unsigned i = 0; i < localSize.getDim(); i++)
            fullLocalSize[i] = localSize[i];

        writer.getDomainWriter()(tmpArray.get(), T_globalDims, makeSplashDomain(globalDomain), makeSplashDomain(localOffset, fullLocalSize));
    }
};

/// Write a field from a databox
/// @param writer       Instance of SplashWriter
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data in the box)
template<class T_SplashWriter, class T_DataBox, unsigned T_dim>
void writeDataBox(T_SplashWriter&& writer, const T_DataBox& dataBox, const PMacc::Selection<T_dim>& globalDomain, const PMacc::Selection<T_dim>& localDomain)
{
    DataBoxWriter<typename T_DataBox::ValueType> boxWriter;
    boxWriter(writer, dataBox, globalDomain, localDomain.size, localDomain.offset);
}

/// Write a field from a databox
/// @param writer       Instance of SplashWriter
/// @param dataBox      Databox containing the data
/// @param globalDomain Offset and Size of the field over all processes
/// @param localSize    Size of the field on the current process (must match extents of data in the box)
/// @param localOffset  Offset of the field of the current process
template<class T_SplashWriter, class T_DataBox, unsigned T_globalDims>
void writeDataBox(T_SplashWriter&& writer, const T_DataBox& dataBox,
        const PMacc::Selection<T_globalDims>& globalDomain,
        const PMacc::DataSpace<T_DataBox::Dim>& localSize,
        const PMacc::DataSpace<T_globalDims>& localOffset)
{
    DataBoxWriter<typename T_DataBox::ValueType> boxWriter;
    boxWriter(writer, dataBox, globalDomain, localSize, localOffset);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt

#include "plugins/hdf5/ComplexBoxWriter.hpp"
