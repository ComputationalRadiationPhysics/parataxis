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

#include "simulation_defines.hpp"
#include "plugins/PrintParticles.hpp"
#include "plugins/DebugHelper.hpp"
#if (PARATAXIS_ENABLE_PNG == 1 || PARATAXIS_ENABLE_TIFF == 1)
#   include "plugins/PrintField.hpp"
#endif
#if (PARATAXIS_ENABLE_TIFF == 1)
#   include "plugins/TiffToDensity.hpp"
#endif
// PrintDetector handles TIFF and/or HDF5 output
#if (PARATAXIS_ENABLE_TIFF == 1 || PARATAXIS_ENABLE_HDF5 == 1)
#   include "plugins/PrintDetector.hpp"
#endif
#if (PARATAXIS_ENABLE_HDF5 == 1)
#   include "plugins/hdf5/HDF5Output.hpp"
#   include "plugins/hdf5/LaserSource.hpp"
#endif
#include <boost/mpl/vector.hpp>

namespace parataxis {

    /* stand alone plugins (no placeholders) */
    typedef bmpl::vector<
#if (PARATAXIS_ENABLE_TIFF == 1)
            plugins::TiffToDensity
#endif
#if (PARATAXIS_ENABLE_HDF5 == 1)
#   if (PARATAXIS_ENABLE_TIFF == 1)
            ,
#   endif
            plugins::hdf5::HDF5Output
#endif
    > StandAlonePlugins;

    /* species plugins (with placeholder replaced by species) */
    typedef bmpl::vector<
#if (PARATAXIS_ENABLE_HDF5 == 1)
            plugins::hdf5::LaserSource<bmpl::_1>,
#endif
#if ENABLE_PRINT_PARTICLES
            plugins::PrintParticles<bmpl::_1>,
#endif
            plugins::DebugHelper<bmpl::_1>
    > SpeciesPlugins;

    /* field plugins (with placeholder replaced by field) */
    typedef bmpl::vector<
#if ENABLE_PRINT_FIELDS && (PARATAXIS_ENABLE_PNG == 1 || PARATAXIS_ENABLE_TIFF == 1)
            plugins::PrintField<bmpl::_1>
#endif
    > FieldPlugins;

    /* detector plugins (with placeholder replaced by detector) */
    typedef bmpl::vector<
#if ENABLE_PRINT_DETECTORS && (PARATAXIS_ENABLE_TIFF == 1 || PARATAXIS_ENABLE_HDF5 == 1)
            plugins::PrintDetector<bmpl::_1>
#endif
    > DetectorPlugins;

}  // namespace parataxis
