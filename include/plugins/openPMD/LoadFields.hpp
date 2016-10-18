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

namespace xrt {
namespace plugins {
namespace openPMD {

template<class T_SplashReader>
struct LoadField
{
    T_SplashReader& reader_;

    LoadField(T_SplashReader& reader): reader_(reader){}

    template<class T_DataBox>
    void operator()(const std::string& name, const GridLayout& fieldLayout, T_DataBox fieldBox)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("Begin loading field '%1%'") % name;

        const auto& subGrid = Environment::get().SubGrid();

        reader_.setCurrentDataset(std::string("fields/") + name);

        hdf5::readDataBox(
                    reader_,
                    fieldBox.shift(fieldLayout.getGuard()),
                    subGrid.getGlobalDomain(),
                    PMacc::Selection<simDim>(
                            fieldLayout.getDataSpaceWithoutGuarding(),
                            subGrid.getLocalDomain().offset
                    )
                );

        PMacc::log<XRTLogLvl::IN_OUT>("Finished loading field '%1%'") % name;
    }
};

/**
 * Functor for loading a field
 */
template<typename T_Field>
struct LoadFields
{

    template<class T_SplashReader>
    void operator()(T_SplashReader& reader)
    {
        auto& dc = Environment::get().DataConnector();

        /* load field without copying data to host */
        T_Field& field = dc.getData<T_Field>(T_Field::getName(), true);

        LoadField<T_SplashReader> loadField(reader);
        loadField(
                  traits::OpenPMDName<T_Field>::get(),
                  field.getGridBuffer().getGridLayout(),
                  field.getHostDataBox());

        field.getGridBuffer().hostToDevice();

        dc.releaseData(T_Field::getName());
    }
};

} //namespace openPMD
} //namespace plugins
} //namespace xrt
