#pragma once

namespace xrt {
namespace traits {
    /** Convert a Splash CollectionType to a PIConGPU Type
     *
     * \tparam T_SplashType Splash CollectionType
     * \return \p ::type as public typedef
     */
    template<typename T_SplashType>
    struct SplashToPIC;

} //namespace traits
}// namespace xrt

#include "SplashToPIC.tpp"
