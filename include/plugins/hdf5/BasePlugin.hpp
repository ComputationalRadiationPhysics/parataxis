#pragma once

#include "xrtTypes.hpp"
#if (ENABLE_HDF5 == 1)
#   include <splash/splash.h>
#endif

namespace po = boost::program_options;

namespace xrt {
namespace plugins {
namespace hdf5 {

class BasePlugin
{
protected:
    BasePlugin()
#if (ENABLE_HDF5 == 1)
     : dataCollector(nullptr)
#endif
    {}

    virtual ~BasePlugin(){}

    virtual std::string getPrefix() const = 0;

    void pluginRegisterHelp(po::options_description& desc)
    {
#if (ENABLE_HDF5 == 1)
        restartFilename.clear();
        desc.add_options()
            ((getPrefix() + ".file").c_str(), po::value<std::string>(&filename)->default_value(getPrefix() + "_data"), "HDF5 output filename (prefix)")
            ((getPrefix() + ".checkpoint-file").c_str(), po::value<std::string>(&checkpointFilename)->default_value(getPrefix() + "_checkpoint"), "HDF5 checkpoint filename (prefix)")
            ((getPrefix() + ".restart-file").c_str(), po::value<std::string>(&restartFilename), "HDF5 restart filename (prefix)")
            ;
#endif
    }

    void openH5File(const std::string& filename, bool openRead);
    void closeH5File();
    void pluginLoad();
    void pluginUnload();

#if (ENABLE_HDF5 == 1)
    std::string filename;
    std::string checkpointFilename;
    std::string restartFilename;

    splash::ParallelDomainCollector* dataCollector;
    splash::Dimensions mpiPos, mpiSize;
#endif
};

void BasePlugin::closeH5File()
{
#if (ENABLE_HDF5 == 1)
    if (dataCollector)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: close DataCollector");
        dataCollector->close();
    }
#endif
}

void BasePlugin::openH5File(const std::string& filename, bool openRead)
{
#if (ENABLE_HDF5 == 1)
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
    attr.fileAccType = openRead ? splash::DataCollector::FAT_READ : splash::DataCollector::FAT_CREATE;
    attr.mpiPosition = mpiPos;
    attr.mpiSize = mpiSize;

    // open datacollector
    try
    {
        PMacc::log<XRTLogLvl::IN_OUT>("HDF5: open DataCollector with file: %1%") % filename;
        dataCollector->open(filename.c_str(), attr);
    }
    catch (const splash::DCException& e)
    {
        std::cerr << e.what() << std::endl;
        throw std::runtime_error("HDF5: failed to open DataCollector");
    }
#endif
}

void BasePlugin::pluginLoad()
{
#if (ENABLE_HDF5 == 1)
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
#endif
}

void BasePlugin::pluginUnload()
{
#if (ENABLE_HDF5 == 1)
   if (dataCollector)
        dataCollector->finalize();

    __delete(dataCollector);
#endif
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
