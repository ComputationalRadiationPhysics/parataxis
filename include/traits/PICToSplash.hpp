#pragma once

namespace xrt {
namespace traits {
    /** Convert a PIConGPU Type to a Splash CollectionType
     *
     * \tparam T_Type Typename in PIConGPU
     * \return \p ::type as public typedef of a Splash CollectionType
     */
    template<typename T_Type>
    struct PICToSplash;

} //namespace traits
}// namespace xrt

#include "PICToSplash.tpp"
