#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "debug/LogLevels.hpp"
#include "ComplexTraits.hpp"
#include "math/abs.hpp"
#include "plugins/imaging/TiffCreator.hpp"
#include "plugins/openPMD/WriteHeader.hpp"
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
    class PrintDetector : public ISimulationPlugin
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

        void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override{}
        void restart(uint32_t restartStep, const std::string restartDirectory) override{}

    protected:
        void pluginLoad() override
        {
            Environment::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);
        }
        void pluginUnload() override
        {
            masterBuffer_.reset();
        }
    private:
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


        splash::Dimensions bufferSize(size.x(), size.y(), 1);

        const char* dataSetName = "meshes/detector";
        writer.SetCurrentDataset(dataSetName);
        hdf5DataFile.write(currentStep,
        				   typename traits::PICToSplash<OutputType>::type(),
                           2,
                           splash::Selection(bufferSize),
                           dataSetName,
                           buffer.getPointer());

        auto writeAttribute = writer.GetAttributeWriter();
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
#endif // ENABLE_HDF5

}  // namespace plugins
}  // namespace xrt
