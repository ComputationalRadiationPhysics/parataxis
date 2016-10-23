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

#include "parataxisTypes.hpp"
#include "plugins/hdf5/DataBoxWriter.hpp"
#include "debug/LogLevels.hpp"
#include <debug/VerboseLog.hpp>

namespace parataxis {
namespace plugins {
namespace openPMD {

template<class T_SplashWriter>
struct WriteField
{

    T_SplashWriter& writer_;

    WriteField(T_SplashWriter& writer): writer_(writer){}

    template<typename T_DataBox>
    void operator()(const std::string name,
               const GridLayout& fieldLayout,
               float_64 unit,
               std::vector<float_64> unitDimension,
               float_X timeOffset,
               const T_DataBox& fieldBox
               )
    {
        using ValueType = typename T_DataBox::ValueType;

        PMacc::log<PARATAXISLogLvl::IN_OUT>("HDF5: write field: %1%") % name;

        const SubGrid& subGrid = Environment::get().SubGrid();
        /* parameter checking */
        assert(unitDimension.size() == 7); // seven openPMD base units
        assert(subGrid.getLocalDomain().size == fieldLayout.getDataSpaceWithoutGuarding());

        /* Change dataset */
        writer_.setCurrentDataset(std::string("fields/") + name);

        hdf5::writeDataBox(
                    writer_,
                    fieldBox.shift(fieldLayout.getGuard()),
                    subGrid.getGlobalDomain(),
                    PMacc::Selection<simDim>(
                            fieldLayout.getDataSpaceWithoutGuarding(),
                            subGrid.getLocalDomain().offset
                    )
                );

        /* attributes */
        auto writeAttribute = writer_.getAttributeWriter();

        std::array<float_X, simDim> positions;
        std::fill_n(positions.begin(), simDim, 0.5);
        writeAttribute("position", positions);

        writeAttribute("unitSI", unit);
        writeAttribute("unitDimension", unitDimension);
        writeAttribute("timeOffset", timeOffset);
        writeAttribute("geometry", "cartesian");
        writeAttribute("dataOrder", "C");

        char axisLabels[simDim][2];
        for(uint32_t d = 0; d < simDim; ++d)
        {
            axisLabels[simDim-1-d][0] = char('x' + d); // 3D: F[z][y][x], 2D: F[y][x]
            axisLabels[simDim-1-d][1] = '\0';          // terminator is important!
        }
        writeAttribute("axisLabels", static_cast<const char*>(&axisLabels[0][0]), simDim);

        std::array<float_X, simDim> gridSpacing;
        for( uint32_t d = 0; d < simDim; ++d )
            gridSpacing[d] = cellSize[d];
        writeAttribute("gridSpacing", gridSpacing);

        std::array<float_64, simDim> gridGlobalOffset;
        for( uint32_t d = 0; d < simDim; ++d )
            gridGlobalOffset[d] = float_64(cellSize[d]) * float_64(subGrid.getGlobalDomain().offset[d]);
        writeAttribute("gridGlobalOffset", gridGlobalOffset);

        writeAttribute("gridUnitSI", float_64(UNIT_LENGTH));
        writeAttribute("fieldSmoothing", "none");
    }

};

}  // namespace openPMD
}  // namespace plugins
}  // namespace parataxis
