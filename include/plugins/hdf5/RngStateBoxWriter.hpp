#pragma once

namespace xrt {
namespace plugins {
namespace hdf5 {

#include "xrtTypes.hpp"
#include "plugins/hdf5/DataBoxWriter.hpp"
#include <random/methods/Xor.hpp>

template<>
struct DataBoxWriter<PMacc::random::methods::Xor::StateType>
{
    template<class T_SplashWriter, class T_DataBox>
    void operator()(T_SplashWriter& writer, const T_DataBox& dataBox, const PMacc::Selection<simDim>& globalDomain, const PMacc::Selection<simDim>& localDomain)
    {
        using ValueType = typename T_DataBox::ValueType;
        static_assert(std::is_same<PMacc::random::methods::Xor::StateType, ValueType>::value, "Wrong type in dataBox");

        const std::string datasetName = writer.GetCurrentDataset();
        writer.SetCurrentDataset(datasetName + "/d");
        writeDataBox(writer, makeHostTransformBox(dataBox, [](const ValueType& state) { return state.d; }), globalDomain, localDomain);
        for(unsigned i=0; i<5; i++)
        {
            writer.SetCurrentDataset(datasetName + "/v" + char('0' + i));
            writeDataBox(writer, makeHostTransformBox(dataBox, [i](const ValueType& state) { return state.v[i]; }), globalDomain, localDomain);
        }

        writer.SetCurrentDataset(datasetName + "/boxmuller_flag");
        writeDataBox(writer, makeHostTransformBox(dataBox, [](const ValueType& state) { return state.boxmuller_flag; }), globalDomain, localDomain);
        writer.SetCurrentDataset(datasetName + "/boxmuller_flag_double");
        writeDataBox(writer, makeHostTransformBox(dataBox, [](const ValueType& state) { return state.boxmuller_flag_double; }), globalDomain, localDomain);
        writer.SetCurrentDataset(datasetName + "/boxmuller_extra");
        writeDataBox(writer, makeHostTransformBox(dataBox, [](const ValueType& state) { return state.boxmuller_extra; }), globalDomain, localDomain);
        writer.SetCurrentDataset(datasetName + "/boxmuller_extra_double");
        writeDataBox(writer, makeHostTransformBox(dataBox, [](const ValueType& state) { return state.boxmuller_extra_double; }), globalDomain, localDomain);

        writer.SetCurrentDataset(datasetName);
    }
};

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
