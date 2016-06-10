#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "plugins/hdf5/SplashWriter.hpp"
#include "plugins/hdf5/DataBoxWriter.hpp"
#include "plugins/hdf5/RngStateBoxWriter.hpp"
#include "plugins/hdf5/splashUtils.hpp"
#include "plugins/openPMD/WriteFields.hpp"
#include "DensityField.hpp"
#include "debug/LogLevels.hpp"
#include <debug/VerboseLog.hpp>
#include <particles/IdProvider.def>
#include <splash/splash.h>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <string>

namespace xrt {
namespace plugins {
namespace hdf5 {

class HDF5Output: public ISimulationPlugin
{
public:
    HDF5Output():
        notifyPeriod(0),
        filename("h5_data"),
        checkpointFilename("h5_checkpoint"),
        dataCollector(nullptr)
    {
        Environment::get().PluginConnector().registerPlugin(this);
    }

    void pluginRegisterHelp(po::options_description& desc) override
    {
        desc.add_options()
            ("hdf5.period", po::value<uint32_t > (&notifyPeriod)->default_value(0), "enable HDF5 IO [for each n-th step]")
            ("hdf5.file", po::value<std::string > (&filename), "HDF5 output filename (prefix)")
            ("hdf5.checkpoint-file", po::value<std::string > (&checkpointFilename), "Optional HDF5 checkpoint filename (prefix)")
            ("hdf5.restart-file", po::value<std::string > (&restartFilename), "HDF5 restart filename (prefix)")
            ;
    }

    std::string pluginGetName() const override
    {
        return "HDF5Output";
    }

    void notify(uint32_t currentStep) override
    {
        writeHDF5(currentStep, false);
    }

    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
    {
        this->checkpointDirectory = checkpointDirectory;
        writeHDF5(currentStep, true);
    }

    void restart(uint32_t restartStep, const std::string restartDirectory) override{}
private:
    void writeHDF5(uint32_t currentStep, bool isCheckpoint);
    void openH5File(const std::string& filename);
    void closeH5File();
    void pluginLoad() override;
    void pluginUnload() override;

    uint32_t notifyPeriod;
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;
    std::string checkpointDirectory;

    splash::ParallelDomainCollector* dataCollector;
    splash::Dimensions mpiPos, mpiSize;
};

void HDF5Output::writeHDF5(uint32_t currentStep, bool isCheckpoint)
{
    std::string fname;
    if (isCheckpoint)
    {
        if (boost::filesystem::path(checkpointFilename).is_relative())
            fname = checkpointDirectory + "/" + checkpointFilename;
        else
            fname = checkpointFilename;
    }else
        fname = filename;
    openH5File(fname);
    auto writer = makeSplashWriter(*dataCollector, currentStep);
    openPMD::writeHeader(writer, fname, true);

    __getTransactionEvent().waitForFinished();

    // Write IdProvider
    const auto idProviderState = PMacc::IdProvider<simDim>::getState();
    PMacc::log<XRTLogLvl::IN_OUT>("HDF5: Writing IdProvider state (StartId: %1%, NextId: %2%, maxNumProc: %3%)")
            % idProviderState.startId % idProviderState.nextId % idProviderState.maxNumProc;
    writer.SetCurrentDataset("picongpu/idProvider/startId");
    splash::Domain globalDomain =
            makeSplashDomain(
                    Environment::get().GridController().getPosition(),
                    Environment::get().GridController().getGpuNodes()
            );

    // For localDomain we use the default: No offset, single element
    writer.GetFieldWriter()(&idProviderState.startId, globalDomain, splash::Domain());
    writer.GetAttributeWriter()("maxNumProc", idProviderState.maxNumProc);

    writer.SetCurrentDataset("picongpu/idProvider/nextId");
    writer.GetFieldWriter()(&idProviderState.nextId, globalDomain, splash::Domain());

    auto& dc = Environment::get().DataConnector();
    // Write Random field
    auto& rngBuffer = dc.getData<RNGProvider>(RNGProvider::getName()).getStateBuffer().getHostBuffer();
    if(rngBuffer.getCurrentDataSpace().productOfComponents() > 0)
    {
        writer.SetCurrentDataset("picongpu/rngProvider");
        hdf5::writeDataBox(
                writer,
                rngBuffer.getDataBox(),
                Environment::get().SubGrid().getGlobalDomain(),
                Environment::get().SubGrid().getLocalDomain()
                );
    }
    Environment::get().DataConnector().releaseData(RNGProvider::getName());

    openPMD::WriteFields<DensityField>()(writer);

    closeH5File();
}

void HDF5Output::closeH5File()
{
    if (dataCollector != NULL)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5 close DataCollector");
        dataCollector->close();
    }
}

void HDF5Output::openH5File(const std::string& filename)
{
    const uint32_t maxOpenFilesPerNode = 4;
    if (dataCollector == nullptr)
    {
        auto& gc = Environment::get().GridController();
        dataCollector = new splash::ParallelDomainCollector(
                                                              gc.getCommunicator().getMPIComm(),
                                                              gc.getCommunicator().getMPIInfo(),
                                                              mpiSize,
                                                              maxOpenFilesPerNode);
    }
    // set attributes for datacollector files
    splash::DataCollector::FileCreationAttr attr;
    attr.enableCompression = false;
    attr.fileAccType = splash::DataCollector::FAT_CREATE;
    attr.mpiPosition = mpiPos;
    attr.mpiSize = mpiSize;

    // open datacollector
    try
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5 open DataCollector with file: %1%") % filename;
        dataCollector->open(filename.c_str(), attr);
    }
    catch (const splash::DCException& e)
    {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("HDF5 failed to open DataCollector");
    }
}

void HDF5Output::pluginLoad()
{
    auto& gc = Environment::get().GridController();
    /* It is important that we never change the mpi_pos after this point
     * because we get problems with the restart.
     * Otherwise we do not know which gpu must load the ghost parts around
     * the sliding window.
     */

    mpiPos.set(0, 0, 0);
    mpiSize.set(1, 1, 1);

    for (uint32_t i = 0; i < simDim; ++i)
    {
        mpiPos[i] = gc.getPosition()[i];
        mpiSize[i] = gc.getGpuNodes()[i];
    }

    if (notifyPeriod > 0)
        Environment::get().PluginConnector().setNotificationPeriod(this, notifyPeriod);

    if (restartFilename.empty())
        restartFilename = checkpointFilename;
}

void HDF5Output::pluginUnload()
{
    if (dataCollector)
        dataCollector->finalize();

    __delete(dataCollector);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
