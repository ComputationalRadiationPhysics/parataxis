#pragma once

namespace xrt {
namespace plugins {
namespace hdf5 {

#include "xrtTypes.hpp"
#include <splash/splash.h>

/** Convert a PMacc::Selection instance into a splash::Domain */
template<unsigned T_dim>
splash::Domain makeSplashDomain(const PMacc::Selection<T_dim>& selection)
{
    splash::Domain splashDomain;

    for (uint32_t d = 0; d < T_dim; ++d)
    {
        splashDomain.getOffset()[d] = selection.offset[d];
        splashDomain.getSize()[d] = selection.size[d];
    }
    return splashDomain;
}

/** Convert offset and size as PMacc::DataSpace instances into a splash::Domain */
template<unsigned T_dim>
splash::Domain makeSplashDomain(const PMacc::DataSpace<T_dim>& offset, const PMacc::DataSpace<T_dim>& size)
{
    splash::Domain splashDomain;

    for (uint32_t d = 0; d < T_dim; ++d)
    {
        splashDomain.getOffset()[d] = offset[d];
        splashDomain.getSize()[d] = size[d];
    }
    return splashDomain;
}

/** Check if a domain containing numDims dimensions is valid (excess offsets=0, sizes=1) */
bool isDomainValid(const splash::Domain domain, unsigned numDims)
{
    assert(numDims > 0);
    for(unsigned d = numDims; d <= 3; d++)
    {
        if(domain.getOffset()[d - 1] != 0)
            return false;
        if(domain.getSize()[d - 1] != 1)
            return false;
    }
    return true;
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
