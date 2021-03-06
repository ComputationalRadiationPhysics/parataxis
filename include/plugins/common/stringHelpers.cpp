/**
 * Copyright 2015-2016 Axel Huebl, Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
#include "plugins/common/stringHelpers.hpp"
#include <ctime>

namespace parataxis {
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
} // namespace parataxis
