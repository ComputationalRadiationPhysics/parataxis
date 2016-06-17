#pragma once

#include "xrtTypes.hpp"
#include "plugins/hdf5/DataBoxReader.hpp"

namespace xrt {
namespace plugins {
namespace openPMD {

template<class T_SplashWriter>
struct LoadField
{
    T_SplashWriter& writer_;

    LoadField(T_SplashWriter& writer): writer_(writer){}

    template<class T_DataBox>
    void operator()(const std::string& name, const GridLayout& fieldLayout, T_DataBox fieldBox)
    {
        PMacc::log<XRTLogLvl::IN_OUT>("Begin loading field '%1%'") % name;

        const auto& subGrid = Environment::get().SubGrid();

        writer_.SetCurrentDataset(std::string("fields/") + name);

        hdf5::readDataBox(
                    writer_,
                    fieldBox.shift(fieldLayout.getGuard()),
                    subGrid.getGlobalDomain(),
                    PMacc::Selection<simDim>(
                            fieldLayout.getDataSpaceWithoutGuarding(),
                            subGrid.getLocalDomain().offset
                    )
                );

        PMacc::log<XRTLogLvl::IN_OUT>("Finished loading field '%1%'") % name;
    }
};

/**
 * Functor for loading a field
 */
template<typename T_Field>
struct LoadFields
{

    template<class T_SplashWriter>
    void operator()(T_SplashWriter& writer)
    {
        auto& dc = Environment::get().DataConnector();

        /* load field without copying data to host */
        T_Field& field = dc.getData<T_Field>(T_Field::getName(), true);

        LoadField<T_SplashWriter> loadField(writer);
        loadField(
                  traits::OpenPMDName<T_Field>::get(),
                  field.getGridBuffer().getGridLayout(),
                  field.getHostDataBox());

        field.getGridBuffer().hostToDevice();

        dc.releaseData(T_Field::getName());
    }
};

} //namespace openPMD
} //namespace plugins
} //namespace xrt
