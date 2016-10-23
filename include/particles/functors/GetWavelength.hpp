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

#include "parataxisTypes.hpp"

namespace parataxis {
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
}  // namespace parataxis
