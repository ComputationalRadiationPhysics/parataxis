/**
 * Copyright 2013-2016 Rene Widera, Alexander Grund
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

namespace parataxis {

    namespace bmpl = boost::mpl;
    enum class CommTag: uint32_t
    {
        NO_COMMUNICATION,
        BUFF,
        SPECIES_FIRSTTAG /* This needs to be the last one! */
    };

}  // namespace parataxis

/* Use #include <> to allow user overrides */
#include <simulation_defines/_defaultParam.loader>
#include <simulation_defines/extensionParam.loader>

#include <simulation_defines/_defaultUnitless.loader>
#include <simulation_defines/extensionUnitless.loader>
//load starter after user extensions and all params are loaded
#include <simulation_defines/unitless/starter.unitless>
