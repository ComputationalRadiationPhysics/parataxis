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
            for(unsigned i=simDim; i<container.size(); ++i)
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
        for(unsigned i=0; i<simDim; ++i)
            result[i] = container[i];

        return result;
    }


}  // namespace xrt
