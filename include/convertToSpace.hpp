/**
 * Copyright 2015-2016 Alexander Grund
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
 
#pragma once

#include "xrtTypes.hpp"
#include <string>

namespace xrt {

    /**
     * Copies values from a vector-like container (size(), push_back(..), []) to a Space instance
     * Fills missing values with @ref defaultValue and shows an error if superfluous values !=
     * @ref defaultValue are found
     *
     * @param container    input values
     * @param defaultValue default value
     * @param description  description of the container shown in the error message
     *                     if this is empty, no checking is performed
     * @return Space instance with values from container
     */
    template< class T_Container, typename T_Value >
    Space
    convertToSpace(T_Container&& container, T_Value&& defaultValue, const std::string& description)
    {
        /* fill with default values */
        while ( container.size() < simDim )
            container.push_back(defaultValue);

        if(!description.empty()){
            for(uint32_t i=simDim; i<container.size(); ++i)
            {
                if(container[i] != defaultValue)
                {
                    std::cerr << "Warning for " << description << ": "
                              << container[i] << " for dimension " << (i+1) << " specified "
                              << " but simulation has only " << simDim << " dimensions.\n"
                              << "Value will be ignored!" << std::endl;
                }
            }
        }

        Space result;
        for(uint32_t i=0; i<simDim; ++i)
            result[i] = container[i];

        return result;
    }


}  // namespace xrt
