#pragma once

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
    /// @param localDomain  Offset and Size of the field on the current process (Size must match extents of data in the box)
    template<class T_SplashWriter, class T_DataBox>
    void operator()(T_SplashWriter& writer, const T_DataBox& dataBox, const PMacc::Selection<simDim>& globalDomain, const PMacc::Selection<simDim>& localDomain)
    {
        using ValueType = typename T_DataBox::ValueType;
        static_assert(std::is_same<T_ValueType, ValueType>::value, "Wrong type in dataBox");
        const size_t tmpArraySize = localDomain.size.productOfComponents();
        std::unique_ptr<ValueType[]> tmpArray(new ValueType[tmpArraySize]);

        typedef PMacc::DataBoxDim1Access<T_DataBox> D1Box;
        D1Box d1Access(dataBox, localDomain.size);

        /* copy data to temp array
         * tmpArray has the size of the data without any offsets
         */
        for (size_t i = 0; i < tmpArraySize; ++i)
            tmpArray[i] = d1Access[i];

        writer.GetFieldWriter()(tmpArray.get(), makeSplashDomain(globalDomain), makeSplashDomain(localDomain));
    }
};

template<class T_SplashWriter, class T_DataBox>
void writeDataBox(T_SplashWriter& writer, const T_DataBox& dataBox, const PMacc::Selection<simDim>& globalDomain, const PMacc::Selection<simDim>& localDomain)
{
    DataBoxWriter<typename T_DataBox::ValueType> boxWriter;
    boxWriter(writer, dataBox, globalDomain, localDomain);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
