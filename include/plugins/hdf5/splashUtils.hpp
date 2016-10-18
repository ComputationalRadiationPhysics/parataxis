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
#include <splash/splash.h>

namespace xrt {
namespace plugins {
namespace hdf5 {

template<unsigned T_dim>
splash::Dimensions makeSplashSize(const PMacc::DataSpace<T_dim>& size)
{
    splash::Dimensions splashSize;

    for (uint32_t d = 0; d < T_dim; ++d)
        splashSize[d] = size[d];
    return splashSize;
}

/** Convert a PMacc::Selection instance into a splash::Domain */
template<unsigned T_dim>
splash::Domain makeSplashDomain(const PMacc::Selection<T_dim>& selection)
{
    splash::Domain splashDomain;

    for (uint32_t d = 0; d < T_dim; ++d)
        splashDomain.getOffset()[d] = selection.offset[d];
    splashDomain.getSize() = makeSplashSize(selection.size);
    return splashDomain;
}

/** Convert offset and size as PMacc::DataSpace instances into a splash::Domain */
template<unsigned T_dim>
splash::Domain makeSplashDomain(const PMacc::DataSpace<T_dim>& offset, const PMacc::DataSpace<T_dim>& size)
{
    splash::Domain splashDomain;

    for (uint32_t d = 0; d < T_dim; ++d)
        splashDomain.getOffset()[d] = offset[d];
    splashDomain.getSize() = makeSplashSize(size);
    return splashDomain;
}

/** Check if a size contains numDims dimensions (excess=1) */
bool isSizeValid(const splash::Dimensions& size, unsigned numDims)
{
    assert(numDims > 0);
    for(unsigned d = numDims; d < 3; d++)
    {
        if(size[d] != 1)
            return false;
    }
    return true;
}

/** Check if an offset contains numDims dimensions (excess=0) */
bool isOffsetValid(const splash::Dimensions& offset, unsigned numDims)
{
    assert(numDims > 0);
    for(unsigned d = numDims; d < 3; d++)
    {
        if(offset[d] != 0)
            return false;
    }
    return true;
}

/** Check if a domain containing numDims dimensions is valid (excess offsets=0, sizes=1) */
bool isDomainValid(const splash::Domain& domain, unsigned numDims)
{
    return isSizeValid(domain.getSize(), numDims) &&
           isOffsetValid(domain.getOffset(), numDims);
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
