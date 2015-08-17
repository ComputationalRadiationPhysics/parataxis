#pragma once

#include "xrtTypes.hpp"
#include <dataManagement/DataConnector.hpp>

namespace xrt {
namespace particles {
namespace functors {

    /** copy species to host memory
     *
     * use `DataConnector::getData<...>()` to copy data
     */
    template<typename T_SpeciesType>
    struct CopySpeciesToHost
    {
        typedef T_SpeciesType SpeciesType;

        HINLINE void operator()() const
        {
            /* DataConnector copies data to host */
            PMacc::DataConnector &dc = Environment::get().DataConnector();
            dc.getData<SpeciesType>(SpeciesType::FrameType::getName());
            dc.releaseData(SpeciesType::FrameType::getName());
        }
    };

}  // namespace functors
}  // namespace particles
}  // namespace xrt
