#pragma once

#include "xrtTypes.hpp"
#include <simulationControl/TimeInterval.hpp>

namespace xrt {

    class TimeIntervallExt: public PMacc::TimeIntervall
    {
    public:
        /**
         * Convenience function for calling \ref toggleEnd, \ref printInterval and \ref toggleStart
         * @return Result string from \ref printInterval
         */
        std::string
        printCurIntervallRestart()
        {
            this->toggleEnd();
            std::string result = this->printInterval();
            this->toggleStart();
            return result;
        }
    };

}  // namespace xrt
