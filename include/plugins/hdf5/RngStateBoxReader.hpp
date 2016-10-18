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

#include "xrtTypes.hpp"
#include "plugins/hdf5/DataBoxReader.hpp"
#include <random/methods/Xor.hpp>
#include <random/methods/XorMin.hpp>

namespace xrt {
namespace plugins {
namespace hdf5 {

    template<>
    struct DataBoxReader<PMacc::random::methods::XorMin::StateType>
    {
        template<class T_SplashReader, class T_DataBox>
        void operator()(T_SplashReader& reader, const T_DataBox& dataBox,
                const PMacc::Selection<simDim>& globalDomain,
                const PMacc::DataSpace<simDim>& localSize,
                const PMacc::DataSpace<simDim>& localOffset)
        {
            using ValueType = typename T_DataBox::ValueType;
            static_assert(std::is_same<
                    PMacc::random::methods::Xor::StateType, ValueType>::value || // Due to same naming we can also write the common part of Xor
                    std::is_same<PMacc::random::methods::XorMin::StateType, ValueType>::value
                    , "Wrong type in dataBox");
            const PMacc::Selection<simDim> localDomain(localSize, localOffset);

            readDataBox(reader["d"], makeHostTransformBox(dataBox, [](ValueType& state) -> unsigned& { return state.d; }), globalDomain, localDomain);
            std::array<char, 3> vNames = {'v', '0', '\0'};
            for(unsigned i=0; i<5; i++)
            {
                vNames[1] = char('0' + i);
                readDataBox(reader[&vNames.front()], makeHostTransformBox(dataBox, [i](ValueType& state) -> unsigned&  { return state.v[i]; }), globalDomain, localDomain);
            }
        }
    };

    template<>
    struct DataBoxReader<PMacc::random::methods::Xor::StateType>
    {
        template<class T_SplashReader, class T_DataBox>
        void operator()(T_SplashReader& reader, const T_DataBox& dataBox,
                const PMacc::Selection<simDim>& globalDomain,
                const PMacc::DataSpace<simDim>& localSize,
                const PMacc::DataSpace<simDim>& localOffset)
        {
            using ValueType = typename T_DataBox::ValueType;
            static_assert(std::is_same<PMacc::random::methods::Xor::StateType, ValueType>::value, "Wrong type in dataBox");
            const PMacc::Selection<simDim> localDomain(localSize, localOffset);

            // Trick: Read common part of Xor and XorMin
            DataBoxReader<PMacc::random::methods::XorMin::StateType>()(reader, dataBox, globalDomain, localSize, localOffset);

            readDataBox(reader["boxmuller_flag"], makeHostTransformBox(dataBox, [](ValueType& state) -> int&  { return state.boxmuller_flag; }), globalDomain, localDomain);
            readDataBox(reader["boxmuller_flag_double"], makeHostTransformBox(dataBox, [](ValueType& state) -> int&  { return state.boxmuller_flag_double; }), globalDomain, localDomain);
            readDataBox(reader["boxmuller_extra"], makeHostTransformBox(dataBox, [](ValueType& state) -> float&  { return state.boxmuller_extra; }), globalDomain, localDomain);
            readDataBox(reader["boxmuller_extra_double"], makeHostTransformBox(dataBox, [](ValueType& state) -> double&  { return state.boxmuller_extra_double; }), globalDomain, localDomain);
        }
    };

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
