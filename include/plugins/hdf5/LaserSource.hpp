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
#include "plugins/ISimulationPlugin.hpp"
#include "plugins/hdf5/BasePlugin.hpp"
#include "plugins/hdf5/SplashWriter.hpp"
#include "plugins/TaggedBuffer.hpp"
#include "plugins/openPMD/helpers.hpp"
#include <memory/buffers/HostDeviceBuffer.hpp>
#include <memory/buffers/GridBuffer.hpp>
#include <dataManagement/ISimulationData.hpp>

namespace parataxis {
namespace plugins {
namespace hdf5 {

    template<class T_LaserSrc>
    struct GetDistribution
    {
        GetDistribution(): numPhotons(0)
        {
            auto& dc = Environment::get().DataConnector();
            T_LaserSrc& laser = dc.getData(T_LaserSrc::getName(), true);
            dataBox = laser.getBuffer().getDeviceBuffer().getDataBox();
            size = laser.getBuffer().getGridLayout().getDataspaceWithoutGuarding();
            // Skip guards. This means idx "-1" is valid
            dataBox.shift(laser.getBuffer().getGridLayout().getGuard());
            dc.releaseData(T_LaserSrc::getName());
        }

        DINLINE void
        init(Space localCellIdx) const
        {
            // Calculate the float (cell) index on the (HDF5) grid
            float2_X idx = float2_X(localCellIdx.z(), localCellIdx.y()) * float2_X(cellSize.z(), cellSize.y()) - gridLocalOffset;
            idx /= gridCellSize;
            // Get the integer index of the surroundings
            Space2D idxLow = Space2D(PMaccMath::float2int_rd(idx.x()), PMaccMath::float2int_rd(idx.y()));
            Space2D idxHigh = Space2D(PMaccMath::float2int_ru(idx.x()), PMaccMath::float2int_ru(idx.y()));
            // And do a bilinear interpolation between them:
            // First check if we are outside the region:
            if(idxHigh.x() < 0 || idxHigh.y() < 0 || idxLow.x() >= size.x() || idxLow.y() >= size.y())
                numPhotons = 0;
            else
            {
                // Now the following holds: idxLow and idxHigh are indices to valid memory.
                // One of them could be outside the hdf5 grid, but we have a guard around that so that would point to zeroed memory
                // With additional checks for that, we could avoid usage of that memory but probably this is faster.
                // TODO: Evaluate this. Those additional checks might not cause divergence and hence
                // are better than the additional memory usage and transfer

                // Interpolate in X
                float2_X interpolatedX;
                // Special case: We are on the connection between the points
                if(idxLow.x() == idxHigh.x())
                    interpolatedX = float2_X(dataBox(idxLow), dataBox(idxHigh));
                else
                {
                    // Regular case "idxHigh.x - idxLow.x == 1": interpolate
                    interpolatedX[0] = (idxHigh.x() - idx.x()) * dataBox(idxLow) + (idx.x() - idxLow.x()) * dataBox(Space2D(idxHigh.x(), idxLow.y()));
                    interpolatedX[1] = (idxHigh.x() - idx.x()) * dataBox(Space2D(idxLow.x(), idxHigh.y()) + (idx.x() - idxLow.x()) * dataBox(idxHigh.x()));
                }
                // Interpolate in Y
                if(idxLow.y() == idxHigh.y())
                    numPhotons = interpolatedX[0]; // Take any, they are the same value
                else
                    numPhotons = (idxHigh.y() - idx.y()) * interpolatedX[0] + (idx.y() - idxLow.y()) * interpolatedX[1];
            }
        }

        DINLINE float_X
        operator()(uint32_t timeStep) const
        {
            return numPhotons;
        }
        PMACC_ALIGN(numPhotons, float_X);
        PMACC_ALIGN(gridLocalOffset, float2_X);
        PMACC_ALIGN(gridCellSize, float2_X);
        PMACC_ALIGN(size, Space2D);
        PMACC_ALIGN(dataBox, typename T_LaserSrc::DeviceDataBox);
    };

    /** Read the laser specification from source file */
    template<class T_Species>
    class LaserSource: public ISimulationPlugin, private BasePlugin, private PMacc::ISimulationData
    {
        using HostBuffer = PMacc::HostBufferIntern<float_X, 2>;
        using GridBuffer = PMacc::GridBuffer<float_X, 2>;
    public:
        using DeviceDataBox = typename GridBuffer::DeviceBufferType::DataBoxType;

        LaserSource(): lastHDF5Timestep(0), swapAxis(false)
        {
            Environment::get().PluginConnector().registerPlugin(this);
            Environment::get().DataConnector().registerData(*this);
        }

        void synchronize() override{}
        static std::string getName()
        {
            return std::string("laserSrc.") + T_Species::getName();
        }
        PMacc::SimulationDataId getUniqueId()
        {
            return getName();
        }

        void notify(uint32_t currentStep) override {}
        // No checkpoint/restart for now. Data is loaded at every timestep
        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override {}
        void restart(uint32_t restartStep, const std::string restartDirectory) override
        {
            cuBuffer->getDeviceBuffer().reset(false);
            lastHDF5Timestep = -1;
        }

        void pluginRegisterHelp(po::options_description& desc) override {
            const std::string prefix = getPrefix();
            desc.add_options()
                ((prefix + ".hdf5File").c_str(), po::value<std::string>(&baseFilename), "base file name used for loading the laser distribution (extension will be appended)")
                ((prefix + ".yAxisName").c_str(), po::value<std::string>(&yAxisName)->default_value("y"), "HDF5 axis name that is mapped on simulation y axis")
                ((prefix + ".zAxisName").c_str(), po::value<std::string>(&zAxisName)->default_value("x"), "HDF5 axis name that is mapped on simulation z axis")
                ;
        }

        std::string pluginGetName() const override {
            return std::string("Laser Source - ") + T_Species::getName();
        }

        std::string getPrefix() const override {
            return getName();
        }

        void update(uint32_t currentStep) override
        {
            loadBuffer(currentStep * DELTA_T);
        }

        void pluginLoad() override;
        void pluginUnload() override;

        PMacc::GridBuffer<float_X, 2>& getBuffer()
        {
            return *cuBuffer;
        }

    private:

        /** Interpolates the field for the given time, uploading it to the device when finished.
         *  Loads from HDF5 if necessary */
        void loadBuffer(float_X curTime);
        /** Initializes the buffers from HDF5 */
        void initBuffers();
        template<class T_Reader>
        void loadBufferFromHDF5(T_Reader& reader, HostBuffer& buffer);
        /** Check if the HDF5 meta-data is consistent with the expected data */
        template<class T_Reader>
        void validateHDF5(T_Reader& reader) const;


        std::string baseFilename;
        std::string yAxisName, zAxisName;
        /** last used timestep */
        int32_t lastHDF5Timestep;
        std::unique_ptr<HostBuffer> tmpBuffer;
        /** Size of HDF5 cells, offset of the global HDF5 grid (posGrid - simOrigin), offset of local grid in buffers (posGrid - localOrigin) */
        float2_X hdf5CellSize, gridGlobalOffset, gridLocalOffset;
        /** Size of the HDF5 grid in cells, offset to local domain in cells (rounded down, positive) */
        Space2D hdf5GridSize, hdf5LocalCellOffset;
        bool swapAxis;
        std::unique_ptr<GridBuffer> cuBuffer;
        TaggedBuffer<HostBuffer> buffers;
    };

    template<class T_Species>
    void LaserSource<T_Species>::pluginLoad()
    {
        namespace openPMD = plugins::openPMD;

        plugins::hdf5::BasePlugin::pluginLoad();
        lastHDF5Timestep = -1;
        openH5File(baseFilename, true);
        auto reader = plugins::hdf5::makeSplashWriter(*dataCollector, 0);
        openPMD::validate(reader, false);

        reader = reader(openPMD::getMeshesPath(reader))["Nph"];
        const std::array<char, 2> axisLabels = openPMD::getAxisLabels<2>(reader);
        if(axisLabels[0] == yAxisName && axisLabels[1] == zAxisName)
            swapAxis = true;
        else if(axisLabels[0] == zAxisName && axisLabels[1] == yAxisName)
            swapAxis = false;
        else
        {
            throw std::runtime_error(std::string("Invalid axis labels: ") +
                    std::string(axisLabels[0]) + ", " + std::string(axisLabels[1]));
        }
        auto getAttribute = reader.getAttributeReader();
        if(getAttribute.readString("geometry") != "cartesian")
            throw std::runtime_error("Only cartesian geometry is supported");
        float_X gridUnitSI;
        getAttribute("gridSpacing", hdf5CellSize);
        getAttribute("gridUnitSI", gridUnitSI);
        if(swapAxis)
            std::swap(hdf5CellSize[0], hdf5CellSize[1]);
        hdf5CellSize *= gridUnitSI / UNIT_LENGTH;

        splash::Dimensions hdf5FieldSize = reader["x"].getFieldReader().getGlobalSize();
        hdf5GridSize.x() = hdf5FieldSize[0];
        hdf5GridSize.y() = hdf5FieldSize[1];
        if(swapAxis)
            std::swap(hdf5GridSize.x(), hdf5GridSize.y());

        getAttribute("gridGlobalOffset", gridGlobalOffset);
        if(swapAxis)
            std::swap(gridGlobalOffset.x(), gridGlobalOffset.y());
        gridGlobalOffset *= gridUnitSI / UNIT_LENGTH;
        // Read where the values are defined in the cell
        float2_X incellPosition;
        reader["x"].getAttributeReader()("position", incellPosition);
        // For now just add it to the global offset
        gridGlobalOffset += incellPosition * hdf5CellSize;
        float2_X hdf5GlobalSize = hdf5CellSize * float2_X(hdf5GridSize) - gridGlobalOffset;
        const SubGrid& subGrid = Environment::get().SubGrid();
        floatD_X globalSize = subGrid.getGlobalDomain().size() * cellSize;
        if(gridGlobalOffset.x() > 0 || gridGlobalOffset.y() > 0 || hdf5GlobalSize.x() < globalSize.z() || hdf5GlobalSize.y())
            PMacc::log<PARATAXISLogLvl::DOMAINS>("WARNING: Gridsize for %1% in HDF5 file is to small. Will pad with zeroes!") % pluginGetName();

        // This now only applies to data read, so swap after all other uses
        if(getAttribute.readString("dataOrder") != "C")
            swapAxis = !swapAxis;

        const Space localSize = subGrid.getLocalDomain().size;
        Space2D bufferSize;
        bufferSize.x() = PMaccMath::min(PMaccMath::Float2int_rd(localSize.z() * cellSize.z() / hdf5CellSize.x()), hdf5GridSize.x());
        bufferSize.y() = PMaccMath::min(PMaccMath::Float2int_rd(localSize.y() * cellSize.y() / hdf5CellSize.y()), hdf5GridSize.y());
        floatD_X localDomainOffset = subGrid.getLocalDomain().offset * cellSize;
        float2_X localDomainOffset2D = float2_X(localDomainOffset.z(), localDomainOffset.y());
        gridLocalOffset = localDomainOffset2D - gridGlobalOffset;
        hdf5LocalCellOffset = Space2D(PMaccMath::floor(gridLocalOffset.x() / hdf5CellSize.x()), PMaccMath::floor(gridLocalOffset.y() / hdf5CellSize.y()));
        if(hdf5LocalCellOffset.x() < 0)
            hdf5LocalCellOffset.x() = 0;
        if(hdf5LocalCellOffset.y() < 0)
            hdf5LocalCellOffset.y() = 0;
        localDomainOffset2D -= hdf5LocalCellOffset * hdf5CellSize;
        // Restrict buffer size to maximum available size
        bufferSize.x() = PMaccMath::max(0, PMaccMath::min(bufferSize.x(), hdf5GridSize.x() - hdf5LocalCellOffset.x()));
        bufferSize.y() = PMaccMath::max(0, PMaccMath::min(bufferSize.y(), hdf5GridSize.y() - hdf5LocalCellOffset.y()));
        if(swapAxis)
            tmpBuffer.reset(new HostBuffer(bufferSize.revert()));
        // Keep one guard cell (zeroed) to avoid to many cases in kernel
        cuBuffer.reset(new GridBuffer(PMacc::GridLayout<2>(bufferSize, Space2D(1, 1))));

        initBuffers();
        closeH5File();
    }

    template<class T_Species>
    template<class T_Reader>
    void LaserSource<T_Species>::validateHDF5(T_Reader& reader) const
    {
        openPMD::validate(reader, false);

        // Validate data
        reader = reader(openPMD::getMeshesPath(reader))["Nph"];
        auto getAttribute = reader.getAttributeReader();
        bool tmpSwapAxis = swapAxis;
        if(getAttribute.readString("dataOrder") != "C")
            tmpSwapAxis = !tmpSwapAxis;

        const std::array<char, 2> axisLabels = openPMD::getAxisLabels<2>(reader);
        if(tmpSwapAxis)
            std::swap(axisLabels[0], axisLabels[1]);
        if(axisLabels[0] != zAxisName || axisLabels[1] == yAxisName)
        {
            throw std::runtime_error(std::string("Invalid axis labels: ") +
                    std::string(axisLabels[0]) + ", " + std::string(axisLabels[1]) +
                    " at HDF5-ID: " + std::to_string(reader.getId()));
        }
        if(getAttribute.readString("geometry") != "cartesian")
            throw std::runtime_error("Only cartesian geometry is supported");
        float2_X gridSpacing;
        float_X gridUnitSI;
        getAttribute("gridSpacing", gridSpacing);
        getAttribute("gridUnitSI", gridUnitSI);
        if(tmpSwapAxis)
            std::swap(gridSpacing[0], gridSpacing[1]);
        gridSpacing *= gridUnitSI / UNIT_LENGTH;
        if(PMaccMath::abs(hdf5CellSize[0] - gridSpacing[0]) >= hdf5CellSize[0] * 1e-5 ||
                PMaccMath::abs(hdf5CellSize[1] - gridSpacing[1]) >= hdf5CellSize[1] * 1e-5)
        {
            throw std::runtime_error(std::string("Changed gridSpacing at HDF5-ID: ") +
                    std::to_string(reader.getId()));
        }

        splash::Dimensions hdf5FieldSize = reader["x"].getFieldReader().getGlobalSize();
        if(tmpSwapAxis)
            std::swap(hdf5FieldSize[0], hdf5FieldSize[1]);
        if(hdf5FieldSize[0] != hdf5GridSize[0] || hdf5FieldSize[1] == hdf5GridSize[1])
        {
            throw std::runtime_error(std::string("Changed HDF5 field size at HDF5-ID: ") +
                    std::to_string(reader.getId()));
        }

        float2_X hdf5GridGlobalOffset;
        getAttribute("gridGlobalOffset", hdf5GridGlobalOffset);
        if(swapAxis)
            std::swap(hdf5GridGlobalOffset.x(), hdf5GridGlobalOffset.y());
        hdf5GridGlobalOffset *= gridUnitSI / UNIT_LENGTH;
        // Read where the values are defined in the cell
        float2_X incellPosition;
        reader["x"].getAttributeReader()("position", incellPosition);
        // For now just add it to the global offset
        hdf5GridGlobalOffset += incellPosition * hdf5CellSize;

        if(PMaccMath::abs(gridGlobalOffset[0] - hdf5GridGlobalOffset[0]) >= gridGlobalOffset[0] * 1e-5 ||
                PMaccMath::abs(gridGlobalOffset[1] - hdf5GridGlobalOffset[1]) >= gridGlobalOffset[1] * 1e-5)
        {
            throw std::runtime_error(std::string("IChanged gridGlobalOffset or (incell-)position at HDF5-ID: ") +
                    std::to_string(reader.getId()));
        }
    }

    template<class T_Species>
    void LaserSource<T_Species>::pluginUnload()
    {
        plugins::hdf5::BasePlugin::pluginUnload();
    }

    template<class T_Species>
    template<class T_Reader>
    void LaserSource<T_Species>::loadBufferFromHDF5(T_Reader& reader, HostBuffer& buffer)
    {
        // TODO: y-polarized photons are currently not there and hence not used
        reader = reader(openPMD::getMeshesPath(reader))["Nph/x"];

        if(swapAxis)
        {
            // Read data with reverted sizes
            reader.getFieldReader()(tmpBuffer->getBasePointer(),
                    2,
                    plugins::hdf5::makeSplashSize(Space2D(hdf5GridSize.revert())),
                    plugins::hdf5::makeSplashDomain(Space2D(hdf5LocalCellOffset.revert()), Space2D(tmpBuffer->getDataSpace())));
            // Transpose into destination
            // TODO: This adds a synchronization point into the event system. We might want to run this in parallel to kernels
            auto tmpBox = tmpBuffer->getDataBox();
            auto destBox = buffer.getDataBox();
            for(int y=0; y<buffer.getDataSpace().y(); ++y)
            {
                for(int x=0; x<buffer.getDataSpace().x(); ++x)
                    destBox(Space2D(x, y)) = tmpBox(Space2D(y, x));
            }
        }else
        {
            reader.getFieldReader()(buffer.getBasePointer(),
                    2,
                    plugins::hdf5::makeSplashSize(hdf5GridSize),
                    plugins::hdf5::makeSplashDomain(hdf5LocalCellOffset, buffer.getDataSpace()));
        }
    }

    template<class T_Species>
    void LaserSource<T_Species>::initBuffers()
    {
        namespace openPMD = plugins::openPMD;
        buffers.clear();
        const Space2D bufferSize = cuBuffer->getGridLayout().getDataSpace();
        // No elements -> Nothing to do
        if(bufferSize.productOfComponents() == 0)
            return;
        const int32_t maxHDF5Timestep = dataCollector->getMaxID();
        if(maxHDF5Timestep < 2)
            throw std::runtime_error("Found less than 2 timesteps in the HDF dataset. Cannot continue.");
        for(int32_t i = 0; i<maxHDF5Timestep; i++)
        {
            auto reader = plugins::hdf5::makeSplashWriter(*dataCollector, i);
            validateHDF5(reader);

            auto readAttr = reader(openPMD::getBasePath(reader)).getAttributeReader();
            float_64 timeUnitSI;
            float_X hdf5Timestep, hdf5Time;
            readAttr("dt", hdf5Timestep);
            readAttr("time", hdf5Time);
            readAttr("timeUnitSI", timeUnitSI);
            hdf5Timestep *= UNIT_TIME/timeUnitSI;
            hdf5Time *= UNIT_TIME/timeUnitSI;

            readAttr = reader(openPMD::getMeshesPath(reader))["Nph"].getAttributeReader();
            float_X hdf5TimeOffset;
            readAttr("timeOffset", hdf5TimeOffset);
            hdf5Time += hdf5TimeOffset * UNIT_TIME/timeUnitSI;

            std::unique_ptr<HostBuffer> buffer(new HostBuffer(bufferSize));
            loadBufferFromHDF5(reader, *buffer);
            buffers.push_back(std::move(buffer), hdf5Time, hdf5Timestep);
        }
        PMacc::log<PARATAXISLogLvl::IN_OUT>("(Re-)Initialized fields from HDF5. Time %1% - %2%")
                % buffers.getFirstTime() % buffers.getLastTime();
    }


    template<class T_Species>
    void LaserSource<T_Species>::loadBuffer(float_X curTime)
    {
        // No elements -> Nothing to do
        if(cuBuffer->getHostBuffer().getDataSpace().productOfComponents() == 0)
            return;
        // Check if completely outside HDF time interval
        if(curTime < buffers.getFirstTime() || curTime >= buffers.getLastTime())
        {
            if(lastHDF5Timestep >= 0)
            {
                // Reset buffer
                cuBuffer->getDeviceBuffer().reset(false);
                lastHDF5Timestep = -1;
            }
            return;
        }
        // We interpolate fields for curTime in [lastTime, nextTime)
        lastHDF5Timestep = buffers.findTimestep(curTime, lastHDF5Timestep);
        assert(lastHDF5Timestep <= nextIdx);
        assert(lastHDF5Timestep >= 0 && lastHDF5Timestep + 1 < buffers.size());
        const float_X lastTime = buffers[lastHDF5Timestep].time;
        const float_X nextTime = buffers[lastHDF5Timestep + 1].time;
        assert(lastTime < nextTime);
        // Linear interpolation
        const float_X interpolationFactor = (curTime - lastTime) / (nextTime - lastTime);
        // Weighting factor: Get HDF5 value as photons per time, then multiply by timestep length,
        // to get the photons to spawn in this timestep
        float_X weighting = DELTA_T / buffers[lastHDF5Timestep + 1].dt;
        // And basically the same for the volume
        weighting *= cellSize.productOfComponents() / hdf5CellSize.productOfComponents();
        const Space2D size = buffers[lastHDF5Timestep].buffer->getDataSpace();
        auto lastBox = buffers[lastHDF5Timestep].buffer->getDataBox();
        auto nextBox = buffers[lastHDF5Timestep + 1].buffer->getDataBox();
        auto destBox = cuBuffer->getHostBuffer().getDataBox();
        // Skip guard
        destBox.shift(cuBuffer->getGridLayout().getGuard());
        #pragma omp parallel for
        for(int y = 0; y < size.y(); ++y)
        {
            #pragma omp simd
            for(Space2D idx(0, y); idx.x()<size.x(); ++idx.x())
                destBox(idx) = (lastBox(idx) + (nextBox(idx) - lastBox(idx)) * interpolationFactor) * weighting;
        }
        cuBuffer->hostToDevice();
    }

}  // namespace hdf5
}  // namespace plugins
}  // namespace parataxis
