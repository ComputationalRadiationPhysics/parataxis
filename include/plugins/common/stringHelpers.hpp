#pragma once

#include <string>

namespace xrt {
namespace plugins {
namespace common {

    /** Return the current date as string
     *
     * \param format, \see http://www.cplusplus.com/reference/ctime/strftime/
     * \return std::string with formatted date
     */
    std::string getDateString(const std::string& format);

} // namespace common
} // namespace plugins
} // namespace xrt
