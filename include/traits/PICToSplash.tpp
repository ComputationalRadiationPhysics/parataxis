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

#if (PARATAXIS_ENABLE_HDF5==1)
#include <splash/splash.h>

#include "simulation_defines.hpp"
#include <boost/mpl/if.hpp>
#include <boost/type_traits.hpp>

namespace parataxis {
namespace traits {

    template<>
    struct PICToSplash<bool>
    {
        typedef splash::ColTypeBool type;
    };

    template<>
    struct PICToSplash<float_32>
    {
        typedef splash::ColTypeFloat type;
    };

    template<>
    struct PICToSplash<float_64>
    {
        typedef splash::ColTypeDouble type;
    };

    template<>
    struct PICToSplash<int32_t>
    {
        typedef splash::ColTypeInt32 type;
    };

    template<>
    struct PICToSplash<uint32_t>
    {
        typedef splash::ColTypeUInt32 type;
    };

    template<>
    struct PICToSplash<int64_t>
    {
        typedef splash::ColTypeInt64 type;
    };

    template<>
    struct PICToSplash<uint64_t>
    {
        typedef splash::ColTypeUInt64 type;
    };

    /** Specialization for uint64_cu.
     *  If uint64_cu happens to be the same as uint64_t we use an unused dummy type
     *  to avoid duplicate specialization
     */
    struct uint64_cu_unused_splash;
    template<>
    struct PICToSplash<
                        typename bmpl::if_<
                            typename bmpl::or_<
                                boost::is_same<uint64_t, PMacc::uint64_cu>,
                                bmpl::bool_<sizeof(PMacc::uint64_cu) != sizeof(uint64_t)>
                            >::type,
                            uint64_cu_unused_splash,
                            PMacc::uint64_cu
                        >::type
                     >: public PICToSplash<uint64_t>
    {};

    template<>
    struct PICToSplash<splash::Dimensions>
    {
        typedef splash::ColTypeDim type;
    };

} //namespace traits
}// namespace parataxis

#endif // (PARATAXIS_ENABLE_HDF5==1)
