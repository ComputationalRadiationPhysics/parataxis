#pragma once

#if (XRT_ENABLE_HDF5==1)
#include <splash/splash.h>

#include "simulation_defines.hpp"

namespace xrt {
namespace traits {
    template<>
    struct SplashToPIC<splash::ColTypeBool>
    {
        typedef bool type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeFloat>
    {
        typedef float_32 type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeDouble>
    {
        typedef float_64 type;
    };

    /** Native int */
    template<>
    struct SplashToPIC<splash::ColTypeInt>
    {
        typedef int type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeInt32>
    {
        typedef int32_t type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeUInt32>
    {
        typedef uint32_t type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeInt64>
    {
        typedef int64_t type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeUInt64>
    {
        typedef uint64_t type;
    };

    template<>
    struct SplashToPIC<splash::ColTypeDim>
    {
        typedef splash::Dimensions type;
    };

} //namespace traits
}// namespace xrt

#endif // (XRT_ENABLE_HDF5==1)
