/**
 * Copyright 2016 Axel Huebl
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

namespace parataxis
{
namespace traits
{
    /** Get the openPMD name for an identifier via static get() */
    template<typename T_Identifier>
    struct OpenPMDName;

    /** Get the openPMD Unit via static get() and the dimension via getDimension() */
    template<typename T_Identifier, typename T_Frame>
    struct OpenPMDUnit;

} // namespace traits
} // namespace parataxis

#include "PICToOpenPMD.tpp"
