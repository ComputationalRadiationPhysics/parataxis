#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"
#include "ComplexTraits.hpp"
#include "math/abs.hpp"
#include "plugins/imaging/TiffCreator.hpp"
#if (ENABLE_HDF5 == 1)
#   include "plugins/openPMD/WriteHeader.hpp"
#   include "plugins/hdf5/DataBoxWriter.hpp"
#   include "plugins/hdf5/DataBoxReader.hpp"
#endif
#include "plugins/hdf5/BasePlugin.hpp"
#include "TransformBox.hpp"
#include "traits/PICToSplash.hpp"
#include "traits/SplashToPIC.hpp"

#include <dataManagement/DataConnector.hpp>
#include <debug/VerboseLog.hpp>
#include <mpi/MPIReduce.hpp>
#include <mpi/reduceMethods/Reduce.hpp>
#include <nvidia/functors/Add.hpp>
#include <memory/buffers/HostBufferIntern.hpp>
#include <splash/splash.h>
#include <string>
#include <sstream>
#include <array>

namespace xrt {
namespace plugins {

    template<class T_Detector>
    class PrintDetector : public ISimulationPlugin, hdf5::BasePlugin
    {
        using Detector = T_Detector;

        std::string name;
        std::string prefix;

        uint32_t notifyPeriod;
        std::string fileName;
        bool noBeamstop;

        PMacc::mpi::MPIReduce reduce_;
        using ReduceMethod = PMacc::mpi::reduceMethods::Reduce;

        using Type = typename Detector::Type;
        std::unique_ptr< PMacc::HostBuffer<Type, 2> > masterBuffer_;

    public:
        PrintDetector():
            name("PrintDetector: Prints the detector to a TIFF"),
            prefix(Detector::getName() + std::string("_print")),
            notifyPeriod(0),
            noBeamstop(false)
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((prefix + ".period").c_str(), po::value<uint32_t>(&notifyPeriod), "enable analyzer [for each n-th step]")
                ((prefix + ".fileName").c_str(), po::value<std::string>(&fileName)->default_value("detector"), "base file name (_step.tif will be appended)")
                ((prefix + ".noBeamstop").c_str(), po::bool_switch(&noBeamstop)->default_value(false), "Do not delete 'shadow' of target")
                ;
            hdf5::BasePlugin::pluginRegisterHelp(desc);
        }

        std::string pluginGetName() const override
        {
            return name;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Outputting detector at timestep %1%") % currentStep;
            PMacc::DataConnector& dc = Environment::get().DataConnector();

            Detector& detector = dc.getData<Detector>(Detector::getName());

            bool isMaster = reduce_.hasResult(ReduceMethod());
            Space2D size = detector.getSize();
            if(isMaster && !masterBuffer_)
                masterBuffer_.reset(new PMacc::HostBufferIntern<Type, 2>(size));
            reduce_(PMacc::nvidia::functors::Add(),
                   isMaster ? masterBuffer_->getDataBox().getPointer() : nullptr,
                   detector.getHostDataBox().getPointer(),
                   size.productOfComponents(),
                   ReduceMethod()
                   );

            if (isMaster){
                if(!noBeamstop)
                    doBeamstop(masterBuffer_->getDataBox(), size);
                std::stringstream fileName;
                fileName << this->fileName
                         << "_" << std::setw(6) << std::setfill('0') << currentStep
                         << ".tif";

                imaging::TiffCreator tiff;
                auto transformedBox = makeTransformBox(masterBuffer_->getDataBox(), typename Detector::OutputTransformer());
                tiff(fileName.str(), transformedBox, size);
#if (ENABLE_HDF5==1)
                writeHDF5(transformedBox, size, currentStep);
#endif
                }

            dc.releaseData(Detector::getName());
        }

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override;
        void restart(uint32_t restartStep, const std::string restartDirectory) override;

    protected:
        void pluginLoad() override
        {
            Environment::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
            hdf5::BasePlugin::pluginLoad();
        }
        void pluginUnload() override
        {
            hdf5::BasePlugin::pluginUnload();
            masterBuffer_.reset();
        }
    private:
        std::string getPrefix() const override
        {
            return Detector::getName();
        }

        template<class T_DataBox>
        void doBeamstop(T_DataBox&& data, const Space2D& size)
        {
            const Space simSize = Environment::get().SubGrid().getTotalDomain().size;
            float_X numBeamCellsX = CELL_DEPTH * simSize.z() / Detector::cellWidth;
            float_X numBeamCellsY = CELL_HEIGHT * simSize.y() / Detector::cellHeight;
            const Space2D start((size.x() - numBeamCellsX) / 2, (size.y() - numBeamCellsY) / 2);
            const Space2D end((size.x() + numBeamCellsX) / 2, (size.y() + numBeamCellsY) / 2);
            PMacc::log< XRTLogLvl::IN_OUT >("Applying beamstop in range %1%-%2%/%3%-%4%") % start.x() % end.x() % start.y() % end.y();
            for(unsigned y = start.y(); y <= end.y(); y++)
            {
                for(unsigned x = start.x(); x <= end.x(); x++)
                    data(Space2D(x, y)) = 0;
            }
        }
        template<class T_DataBox>
        void writeHDF5(T_DataBox&& data, const Space2D& size, uint32_t currentStep) const;
    };

#if (ENABLE_HDF5==1)

    template<class T_Detector>
    template<class T_DataBox>
    void PrintDetector<T_Detector>::writeHDF5(T_DataBox&& data, const Space2D& size, uint32_t currentStep) const
    {
    	using OutputType = typename std::remove_reference_t<T_DataBox>::ValueType;
        PMacc::HostBufferIntern<OutputType, 2> buffer(size);
        auto bufferBox = buffer.getDataBox();
        for (int32_t y = 0; y < size.y(); ++y)
        {
            for (int32_t x = 0; x < size.x(); ++x)
                bufferBox(Space2D(x, y)) = data(Space2D(x, y));
        }

        const uint32_t maxOpenFilesPerNode = 1;
        PMacc::GridController<simDim> &gc = Environment::get().GridController();
        splash::ParallelDataCollector hdf5DataFile(MPI_COMM_SELF,
                                                   gc.getCommunicator().getMPIInfo(),
                                                   splash::Dimensions(),
                                                   maxOpenFilesPerNode);

        splash::DataCollector::FileCreationAttr fAttr;
        splash::DataCollector::initFileCreationAttr(fAttr);

        hdf5DataFile.open(fileName.c_str(), fAttr);
        auto writer = hdf5::makeSplashWriter(hdf5DataFile, currentStep);
        openPMD::writeHeader(writer, this->fileName);

        writer.setCurrentDataset(std::string("meshes/") + Detector::getName());
        writer.getFieldWriter()(buffer.getPointer(), 2,
                                hdf5::makeSplashSize(size),
                                hdf5::makeSplashDomain(Space2D::create(0), size)
                               );

        auto writeAttribute = writer.getAttributeWriter();
        writeAttribute("geometry", "cartesian");;
        writeAttribute("dataOrder", "C");
        writeAttribute("axisLabels", "y\0x\0", 2); //[y][x]

        std::array<float_X, 2> gridSpacing = {Detector::cellWidth, Detector::cellHeight};
        writeAttribute("gridSpacing", gridSpacing);

        std::array<float_64, 2> gridGlobalOffset = {0, 0};
        writeAttribute("gridGlobalOffset", gridGlobalOffset);

        writeAttribute("gridUnitSI", float_64(UNIT_LENGTH));

        std::array<float_64, 2> position = {0.5, 0.5};
        writeAttribute("position", position);

        std::array<float_64, 7> unitDimension = {0., 0., 0., 0., 0., 0., 0.};
        // Current unit scale is arbitrary. Correct unit would be: {0., 1.,-3., 0., 0., 0., 0.};
        writeAttribute("unitDimension", unitDimension);

        writeAttribute("unitSI", float_64(1));
        writeAttribute("timeOffset", float_X(0));

        hdf5DataFile.close();
    }

    template<class T_Detector>
    void PrintDetector<T_Detector>::checkpoint(uint32_t currentStep, const std::string checkpointDirectory)
    {
        std::string fname;
        if (boost::filesystem::path(checkpointFilename).is_relative())
            fname = checkpointDirectory + "/" + checkpointFilename;
        else
            fname = checkpointFilename;
        openH5File(fname, false);
        auto writer = hdf5::makeSplashWriter(*dataCollector, currentStep);
        openPMD::writeHeader(writer, this->fileName);

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: write detector: %1%") % Detector::getName();

        /* Change dataset */
        writer.setCurrentDataset(std::string("meshes/") + Detector::getName());

        auto& dc = Environment::get().DataConnector();
        Detector& detector = dc.getData<Detector>(Detector::getName());

        const SubGrid& subGrid = Environment::get().SubGrid();
        hdf5::writeDataBox(
                    writer,
                    detector.getHostDataBox(),
                    PMacc::Selection<3>(
                        Space3D(
                            detector.getSize().x(),
                            detector.getSize().y(),
                            Environment::get().GridController().getGpuNodes().productOfComponents()
                        )
                    ),
                    detector.getSize(),
                    Space3D(
                            0,
                            0,
                            Environment::get().GridController().getScalarPosition()
                    )
                );

        /* attributes */
        // Is type integral? Otherwise it is assumed to be complex
        const bool isIntegral = std::is_integral<Type>::value;
        std::array<float_X, simDim> positions;
        positions.fill(0.5);
        auto writeAttribute = (isIntegral ? writer : writer["real"]).getAttributeWriter();
        writeAttribute("position", positions);
        writeAttribute("unitSI", float_64(1));
        if(!isIntegral)
        {
            writeAttribute = writer["imag"].getAttributeWriter();
            writeAttribute("position", positions);
            writeAttribute("unitSI", float_64(1));
        }

        writeAttribute = writer.getAttributeWriter();
        writeAttribute("unitDimension", std::vector<float_64>(7, 0));
        writeAttribute("timeOffset", float_X(0));
        writeAttribute("geometry", "cartesian");
        writeAttribute("dataOrder", "C");

        const char* axisLabels = "z\0y\0x\0";
        writeAttribute("axisLabels", axisLabels, 3);

        std::array<float_X, 3> gridSpacing = {Detector::cellWidth, Detector::cellHeight, 0};
        writeAttribute("gridSpacing", gridSpacing);
        writeAttribute("gridGlobalOffset", std::vector<float_64>(3, 0));
        writeAttribute("gridUnitSI", float_64(UNIT_LENGTH));

        dc.releaseData(Detector::getName());

        closeH5File();
    }

    template<class T_Detector>
    void PrintDetector<T_Detector>::restart(uint32_t currentStep, const std::string restartDirectory)
    {
        std::string fname;
        if (boost::filesystem::path(restartFilename).is_relative())
            fname = restartDirectory + "/" + restartFilename;
        else
            fname = restartFilename;
        openH5File(fname, true);
        auto writer = hdf5::makeSplashWriter(*dataCollector, currentStep);

        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: read detector: %1%") % Detector::getName();

        /* Change dataset */
        writer.setCurrentDataset(std::string("meshes/") + Detector::getName());

        auto& dc = Environment::get().DataConnector();
        Detector& detector = dc.getData<Detector>(Detector::getName(), true);

        const SubGrid& subGrid = Environment::get().SubGrid();
        hdf5::readDataBox(
                    writer,
                    detector.getHostDataBox(),
                    PMacc::Selection<3>(
                        Space3D(
                            detector.getSize().x(),
                            detector.getSize().y(),
                            Environment::get().GridController().getGpuNodes().productOfComponents()
                        )
                    ),
                    detector.getSize(),
                    Space3D(
                            0,
                            0,
                            Environment::get().GridController().getScalarPosition()
                    )
                );

        detector.hostToDevice();

        dc.releaseData(Detector::getName());

        closeH5File();
    }
#endif // ENABLE_HDF5

}  // namespace plugins
}  // namespace xrt
