#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace particles {
namespace functors {

    namespace detail {

        template<typename T_FrameType, bool hasWavelength = HasFlag_t<T_FrameType, wavelength<> >::value>
        struct GetWavelength
        {
            using FrameType = T_FrameType;
            using Wavelength = GetResolvedFlag_t<FrameType, wavelength<> >;

            HDINLINE float_X
            operator()() const
            {
                return Wavelength::getValue() / UNIT_LENGTH;
            }

        };

        template<typename T_FrameType>
        struct GetWavelength<T_FrameType, false>
        {
            using FrameType = T_FrameType;
            static_assert(HasFlag_t<T_FrameType, energy<> >::value, "Species has no wavelength or energy set");
            using Energy = GetResolvedFlag_t<FrameType, energy<> >;

            HDINLINE float_X
            operator()() const
            {
                return PLANCK_CONSTANT * SPEED_OF_LIGHT / (Energy::getValue() / UNIT_ENERGY);
            }

        };

    }  // namespace detail

    /**
     * Returns the wavelength of a species (unit-less) from wavelength or energy property
     */
    template<typename T_Species, class T_FrameType = typename T_Species::FrameType>
    using GetWavelength = detail::GetWavelength<T_FrameType>;

}  // namespace functors
}  // namespace particles
}  // namespace xrt
