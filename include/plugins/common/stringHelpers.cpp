#include "plugins/common/stringHelpers.hpp"
#include <ctime>

namespace xrt {
namespace plugins {
namespace common {

    /** Return the current date as string
     *
     * \param format, \see http://www.cplusplus.com/reference/ctime/strftime/
     * \return std::string with formatted date
     */
    std::string getDateString(const std::string& format)
    {
        time_t rawtime;
        struct tm* timeinfo;
        const size_t maxLen = 30;
        char buffer [maxLen];

        time( &rawtime );
        timeinfo = localtime( &rawtime );

        strftime( buffer, maxLen, format.c_str(), timeinfo );

        return buffer;
    }

} // namespace common
} // namespace plugins
} // namespace xrt
