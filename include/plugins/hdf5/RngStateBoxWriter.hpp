#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/DataBoxReader.hpp"
#include <random/methods/Xor.hpp>

namespace xrt {
namespace plugins {
namespace hdf5 {

template<>
struct DataBoxWriter<PMacc::random::methods::Xor::StateType>
{
    template<class T_SplashWriter, class T_DataBox>
    void operator()(T_SplashWriter& writer, const T_DataBox& dataBox,
            const PMacc::Selection<simDim>& globalDomain,
            const PMacc::DataSpace<simDim>& localSize,
            const PMacc::DataSpace<simDim>& localOffset)
    {
        using ValueType = typename T_DataBox::ValueType;
        static_assert(std::is_same<PMacc::random::methods::Xor::StateType, ValueType>::value, "Wrong type in dataBox");
        const PMacc::Selection<simDim> localDomain(localSize, localOffset);

        writeDataBox(writer["d"], makeHostTransformBox(dataBox, [](const ValueType& state) { return state.d; }), globalDomain, localDomain);
        std::array<char, 3> vNames = {'v', '0', '\0'};
        for(unsigned i=0; i<5; i++)
        {
            vNames[1] = char('0' + i);
            writeDataBox(writer[&vNames.front()], makeHostTransformBox(dataBox, [i](const ValueType& state)  { return state.v[i]; }), globalDomain, localDomain);
        }

        writeDataBox(writer["boxmuller_flag"], makeHostTransformBox(dataBox, [](const ValueType& state)  { return state.boxmuller_flag; }), globalDomain, localDomain);
        writeDataBox(writer["boxmuller_flag_double"], makeHostTransformBox(dataBox, [](const ValueType& state)  { return state.boxmuller_flag_double; }), globalDomain, localDomain);
        writeDataBox(writer["boxmuller_extra"], makeHostTransformBox(dataBox, [](const ValueType& state)  { return state.boxmuller_extra; }), globalDomain, localDomain);
        writeDataBox(writer["boxmuller_extra_double"], makeHostTransformBox(dataBox, [](const ValueType& state)  { return state.boxmuller_extra_double; }), globalDomain, localDomain);
    }
};

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
