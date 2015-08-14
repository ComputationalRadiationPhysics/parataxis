#pragma once

#include "xrtTypes.hpp"
#include <pluginSystem/IPlugin.hpp>

namespace xrt {

    class ISimulationPlugin: public PMacc::IPlugin
    {
    public:
        ISimulationPlugin(): cellDescription_(nullptr)
        {}

        void
        setMappingDesc(const MappingDesc* cellDescription)
        {
            cellDescription_ = cellDescription;
        }
    protected:
        const MappingDesc* cellDescription_;
    };

}  // namespace xrt
