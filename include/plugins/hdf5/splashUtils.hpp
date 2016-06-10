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

    for (uint32_t d = 0; d < simDim; ++d)
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

    for (uint32_t d = 0; d < simDim; ++d)
    {
        splashDomain.getOffset()[d] = offset[d];
        splashDomain.getSize()[d] = size[d];
    }
    return splashDomain;
}

}  // namespace hdf5
}  // namespace plugins
}  // namespace xrt
