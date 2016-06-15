#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/DataBoxWriter.hpp"
#include "debug/LogLevels.hpp"
#include <debug/VerboseLog.hpp>

namespace xrt {
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

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5 write field: %1%") % name;

        const SubGrid& subGrid = Environment::get().SubGrid();
        /* parameter checking */
        assert(unitDimension.size() == 7); // seven openPMD base units
        assert(subGrid.getLocalDomain().size == fieldLayout.getDataSpaceWithoutGuarding());

        /* Change dataset */
        writer_.SetCurrentDataset(std::string("fields/") + name);

        hdf5::writeDataBox(
                    writer_,
                    fieldBox.shift(fieldLayout.getGuard()),
                    subGrid.getGlobalDomain(),
                    PMacc::Selection<simDim>(
                            subGrid.getLocalDomain().offset,
                            fieldLayout.getDataSpaceWithoutGuarding()
                    )
                );

        /* attributes */
        auto writeAttribute = writer_.GetAttributeWriter();

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
}  // namespace xrt
