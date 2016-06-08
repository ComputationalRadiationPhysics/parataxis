#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/SplashWriter.hpp"
#include "debug/LogLevels.hpp"
#include <debug/VerboseLog.hpp>

namespace xrt {
namespace plugins {
namespace openPMD {

struct WriteField
{

    hdf5::SplashWriter& writer_;

    WriteField(hdf5::SplashWriter& writer): writer_(writer){}

    template<typename T_Buffer>
    void operator()(const std::string name,
               const GridLayout& fieldLayout,
               float_64 unit,
               std::vector<float_64> unitDimension,
               float_X timeOffset,
               T_Buffer fieldBuffer
               )
    {
        using DataBoxType = typename T_Buffer::DataBoxType;
        using ValueType = typename DataBoxType::ValueType;

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5 write field: %1%") % name;

        /* parameter checking */
        assert(unitDimension.size() == 7); // seven openPMD base units

        /* Change dataset */
        writer_.SetCurrentDataset(std::string("fields/") + name);

        const SubGrid& subGrid = Environment::get().SubGrid();

        splash::Dimensions splashGlobalDomainOffset(0, 0, 0);
        splash::Dimensions splashLocalOffset(0, 0, 0);
        splash::Dimensions splashGlobalDomainSize(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
        {
            splashLocalOffset[d] = subGrid.getLocalDomain().offset[d];
            splashGlobalDomainOffset[d] = subGrid.getGlobalDomain().offset[d];
            splashGlobalDomainSize[d] = subGrid.getGlobalDomain().size[d];
        }

        size_t tmpArraySize = subGrid.getLocalDomain().size.productOfComponents();
        std::unique_ptr<ValueType[]> tmpArray = new ValueType[tmpArraySize];

        typedef PMacc::DataBoxDim1Access<DataBoxType> D1Box;
        D1Box d1Access(fieldBuffer.getDataBox().shift(fieldLayout.getGuard()), fieldLayout.getDataSpaceWithoutGuarding());

        /* copy data to temp array
         * tmpArray has the size of the data without any offsets
         */
        for (size_t i = 0; i < tmpArraySize; ++i)
            tmpArray[i] = d1Access[i];

        splash::Dimensions sizeSrcData(1, 1, 1);

        for (uint32_t d = 0; d < simDim; ++d)
            sizeSrcData[d] = fieldLayout.getDataSpaceWithoutGuarding()[d];

        writer_.GetFieldWriter()(tmpArray,
                                 splash::Domain(
                                         splashGlobalDomainOffset, /* offset of the global domain */
                                         splashGlobalDomainSize    /* size of the global domain */
                                 ),
                                 splash::Domain(
                                         splashLocalOffset,   /* write offset for this process */
                                         sizeSrcData               /* data size of this process */
                                 ));

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
        writeAttribute("axisLabels", axisLabels);

        std::array<float_X, simDim> gridSpacing;
        for( uint32_t d = 0; d < simDim; ++d )
            gridSpacing[d] = cellSize[d];
        writeAttribute("gridSpacing", gridSpacing);

        std::array<float_64, simDim> gridGlobalOffset;
        for( uint32_t d = 0; d < simDim; ++d )
            gridGlobalOffset[d] = float_64(cellSize[d]) * float_64(splashGlobalDomainOffset[d]);
        writeAttribute("gridGlobalOffset", gridGlobalOffset);

        writeAttribute("gridUnitSI", float_64(UNIT_LENGTH));
    }

};

}  // namespace openPMD
}  // namespace plugins
}  // namespace xrt
