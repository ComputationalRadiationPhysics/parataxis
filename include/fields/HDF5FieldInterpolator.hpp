#pragma once

#include "xrtTypes.hpp"
#include "fields/IFieldManipulator.hpp"
#include "plugins/ISimulationPlugin.hpp"
#include "plugins/hdf5/BasePlugin.hpp"
#include "plugins/hdf5/SplashWriter.hpp"
#include "plugins/openPMD/helpers.hpp"
#include <cuSTL/container/HostBuffer.hpp>

namespace xrt {
namespace fields {

template<class T_Field>
class HDF5FieldInterpolator: public IFieldManipulator, public ISimulationPlugin, private plugins::hdf5::BasePlugin
{
public:

    HDF5FieldInterpolator(): lastTime(-1), nextTime(-1), curHDF5Timestep(0), maxHDF5Timestep(0)
    {
        Environment::get().PluginConnector().registerPlugin(this);
    }

    void notify(uint32_t currentStep) override {}
    void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override {}
    void restart(uint32_t restartStep, const std::string restartDirectory) override {
        curHDF5Timestep = 0;
        reinitFields(restartStep * DELTA_T, true);
    }

    void pluginRegisterHelp(po::options_description& desc) override {
        const std::string prefix = T_Field::getName();
        desc.add_options()
            ((prefix + ".hdf5File").c_str(), po::value<std::string>(&baseFilename), "base file name used for loading the field (extension will be appended)")
            ;
    }

    std::string pluginGetName() const override {
        return T_Field::getName() + "-Manipulator";
    }

    std::string getPrefix() const override {
        return T_Field::getName();
    }

    void update(uint32_t currentStep) override
    {
        loadField(currentStep * DELTA_T);
    }

    void pluginLoad() override;
    void pluginUnload() override;

private:
    using Type = typename T_Field::Type;
    using HostBuffer = PMacc::container::HostBuffer<Type, simDim>;

    /** Interpolates the field for the given time, uploading it to the device when finished.
     *  Loads from HDF5 if necessary */
    void loadField(Type curTime);
    /** Return the biggest HDF5 timestep for which h5Time<=curTime is valid
     *  Requires open HDF5 file*/
    uint32_t findTimestep(Type curTime, uint32_t startTimestep = 0);
    /** Reinitializes the fields from HDF5 starting at curHDF5Timestep as the lower bound
     *  If force is true, data is loaded even when the timestep did not change */
    void reinitFields(Type curTime, bool force);


    std::string baseFilename;
    std::unique_ptr<HostBuffer> lastField, nextField;
    Type lastTime, nextTime;
    /** Highest open HDF5 id (from nextField), highest possible HDF5 id */
    uint32_t curHDF5Timestep, maxHDF5Timestep;
};

template<class T_Field>
void HDF5FieldInterpolator<T_Field>::pluginLoad()
{
    plugins::hdf5::BasePlugin::pluginLoad();
    const Space localSize = Environment::get().SubGrid().getLocalDomain().size;
    std::cout << "Size is " << localSize.toString() << std::endl;
    lastField.reset(new HostBuffer(localSize));
    nextField.reset(new HostBuffer(localSize));

    lastTime = nextTime = -1;
    curHDF5Timestep = maxHDF5Timestep = 0;
    openH5File(baseFilename, true);
    reinitFields(0, true);
    closeH5File();
}

template<class T_Field>
void HDF5FieldInterpolator<T_Field>::pluginUnload()
{
    plugins::hdf5::BasePlugin::pluginUnload();
    lastField.reset();
    nextField.reset();
}

template<class T_Field>
uint32_t HDF5FieldInterpolator<T_Field>::findTimestep(Type curTime, uint32_t startTimestep)
{
    namespace openPMD = plugins::openPMD;
    // Open at first timestep and validate file
    auto reader = plugins::hdf5::makeSplashWriter(*dataCollector, startTimestep);
    maxHDF5Timestep = dataCollector->getMaxID();
    openPMD::validate(reader, false);
    // Get the current delta t
    auto readAttr = reader(openPMD::getBasePath(reader)).getAttributeReader();
    float_64 timeUnitSI;
    float_X dt;
    readAttr("dt", dt);
    readAttr("timeUnitSI", timeUnitSI);
    dt *= UNIT_TIME/timeUnitSI;
    // Get first time
    float_X time = openPMD::getTime(reader);
    // This time should be smaller, than our time.
    // We still consider it valid, if the difference is at most 1 dt (in which case we extrapolate)
    if(time > curTime + dt)
        throw std::runtime_error(std::string("Cannot load field as first entry is at ") +
                std::to_string(time / UNIT_TIME) + "s but requested time is " +
                std::to_string(curTime / UNIT_TIME) + "s");
    else if(time < curTime)
    {
        // We increase the HDF5 time till it is >= our time.
        // Then stepping back takes as to the very last timestep before curTime
        // First fast forward assuming equal distance times
        float_X  diff = curTime - time;
        reader.setId(std::min<uint32_t>(ceil(diff / dt), maxHDF5Timestep));
        // Now single steps
        while(openPMD::getTime(reader) < curTime && reader.getId() < maxHDF5Timestep)
            reader.setId(reader.getId() + 1);
        // And now backward till H5time <= curTime (guaranteed to terminate as for id=0 'time < curTime' is checked
        while(openPMD::getTime(reader) > curTime)
            reader.setId(reader.getId() - 1);
    }
    return reader.getId();
}

template<class T_Field>
void HDF5FieldInterpolator<T_Field>::reinitFields(Type curTime, bool force)
{
    namespace openPMD = plugins::openPMD;
    uint32_t newHDF5Timestep = findTimestep(curTime, curHDF5Timestep);
    // No new step
    if(!force && newHDF5Timestep != curHDF5Timestep)
        return;
    curHDF5Timestep = newHDF5Timestep;
    const auto& subGrid = Environment::get().SubGrid();
    auto reader = plugins::hdf5::makeSplashWriter(*dataCollector, newHDF5Timestep);
    lastTime = openPMD::getTime(reader);
    reader = reader(openPMD::getMeshesPath(reader))["electron_density"];
    const auto axisLabels = openPMD::getAxisLabels<simDim>(reader);
    for(uint32_t i=0; i<simDim; i++)
    {
        if(axisLabels[i] != 'x'+i)
            throw std::runtime_error("Invalid axis labels");
    }
    reader.getFieldReader()(lastField->getDataPointer(),
            simDim,
            plugins::hdf5::makeSplashSize(subGrid.getGlobalDomain().size),
            plugins::hdf5::makeSplashDomain(subGrid.getLocalDomain()));
    if(curHDF5Timestep < maxHDF5Timestep)
    {
        curHDF5Timestep++;
        reader.setId(curHDF5Timestep);
        nextTime = openPMD::getTime(reader);
        reader = reader(openPMD::getMeshesPath(reader))["electron_density"];
        reader.getFieldReader()(nextField->getDataPointer(),
                simDim,
                plugins::hdf5::makeSplashSize(subGrid.getGlobalDomain().size),
                plugins::hdf5::makeSplashDomain(subGrid.getLocalDomain()));
    }else
    {
        nextTime = lastTime;
        *nextField = *lastField;
    }
    PMacc::log<XRTLogLvl::IN_OUT>("(Re-)Initialized fields from HDF5. Record %1%/%2%, time %3% - %4%")
            % curHDF5Timestep % maxHDF5Timestep % lastTime % nextTime;
}


template<class T_Field>
void HDF5FieldInterpolator<T_Field>::loadField(Type curTime)
{
    namespace openPMD = plugins::openPMD;
    // We interpolate fields for curTime in [lastTime, nextTime)
    // So we need to switch of we are out of that timespan and there are times left in the HDF5 dataset
    if(curTime >= nextTime && curHDF5Timestep < maxHDF5Timestep)
    {
        // Advance in time
        curHDF5Timestep++;
        // Last = next, next = invalid data (overwritten below)
        lastTime = nextTime;
        std::swap(lastField, nextField);

        openH5File(baseFilename, true);
        auto reader = plugins::hdf5::makeSplashWriter(*dataCollector, curHDF5Timestep);
        nextTime = openPMD::getTime(reader);
        // Normally we just go one step forward (sim timestep < hdf5 timestep)
        // If that fails we reinit both fields
        if(nextTime > curTime)
        {
            // regular case
            const auto& subGrid = Environment::get().SubGrid();
            reader = reader(openPMD::getMeshesPath(reader))["electron_density"];
            reader.getFieldReader()(nextField->getDataPointer(),
                    simDim,
                    plugins::hdf5::makeSplashSize(subGrid.getGlobalDomain().size),
                    plugins::hdf5::makeSplashDomain(subGrid.getLocalDomain()));
        }else
            reinitFields(curTime, false);
        closeH5File();
    }
    // If nextTime == lastTime we just take the field,
    // else we linearly interpolate (extrapolate for outside) between them
    Type interpolationFactor = (nextTime == lastTime) ? 0 : (curTime - lastTime) / (nextTime - lastTime);
    const size_t size = lastField->size().productOfComponents();
    const Type* last = lastField->getDataPointer();
    const Type* next = nextField->getDataPointer();
    auto& dc = Environment::get().DataConnector();
    T_Field& field = dc.getData<T_Field>(T_Field::getName(), true);
    auto fieldBox = field.getHostDataBox().shift(field.getGridBuffer().getGridLayout().getGuard());
    const Space localSize = Environment::get().SubGrid().getLocalDomain().size;
    assert(localSize == field.getGridBuffer().getGridLayout().getDataSpaceWithoutGuarding());
    for(size_t i = 0; i < size; ++i)
    {
        Space idx = PMacc::DataSpaceOperations<simDim>::map(localSize, i);
        fieldBox(idx) = last[i] + (next[i] - last[i]) * interpolationFactor;
    }
    field.getGridBuffer().hostToDevice();
    dc.releaseData(T_Field::getName());
}

}  // namespace fields
}  // namespace xrt
