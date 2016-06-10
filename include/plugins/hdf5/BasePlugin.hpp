#pragma once

namespace xrt {
namespace plugins {
namespace hdf5 {

class BasePlugin
{
protected:
    BasePlugin():
        filename("h5_data"),
        checkpointFilename("h5_checkpoint"),
        dataCollector(nullptr)
    {}

    virtual ~BasePlugin(){}

    virtual std::string getPrefix() const = 0;

    void pluginRegisterHelp(po::options_description& desc)
    {
        desc.add_options()
            ((getPrefix() + ".file").c_str(), po::value<std::string > (&filename), "HDF5 output filename (prefix)")
            ((getPrefix() + ".checkpoint-file").c_str(), po::value<std::string > (&checkpointFilename), "Optional HDF5 checkpoint filename (prefix)")
            ((getPrefix() + ".restart-file").c_str(), po::value<std::string > (&restartFilename), "HDF5 restart filename (prefix)")
            ;
    }

    void openH5File(const std::string& filename);
    void closeH5File();
    void pluginLoad();
    void pluginUnload();

    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;

    splash::ParallelDomainCollector* dataCollector;
    splash::Dimensions mpiPos, mpiSize;
};

void BasePlugin::closeH5File()
{
    if (dataCollector)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5 close DataCollector");
        dataCollector->close();
    }
}

void BasePlugin::openH5File(const std::string& filename)
{
    const uint32_t maxOpenFilesPerNode = 4;
    if (!dataCollector)
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

void BasePlugin::pluginLoad()
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

    if (restartFilename.empty())
        restartFilename = checkpointFilename;
}

void BasePlugin::pluginUnload()
{
    if (dataCollector)
        dataCollector->finalize();

    __delete(dataCollector);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
