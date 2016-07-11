#pragma once

#include "simulation_defines.hpp"
#include "plugins/PrintParticles.hpp"
#if (XRT_ENABLE_PNG == 1)
#   include "plugins/PrintField.hpp"
#endif
#if (XRT_ENABLE_TIFF == 1)
#   include "plugins/TiffToDensity.hpp"
#endif
// PrintDetector handles TIFF and/or HDF5 output
#if (XRT_ENABLE_TIFF == 1 || XRT_ENABLE_HDF5 == 1)
#   include "plugins/PrintDetector.hpp"
#endif
#if (XRT_ENABLE_HDF5 == 1)
#   include "plugins/hdf5/HDF5Output.hpp"
#endif
#include <boost/mpl/vector.hpp>

namespace xrt {

    /* stand alone plugins (no placeholders) */
    typedef bmpl::vector<
#if (XRT_ENABLE_TIFF == 1)
            plugins::TiffToDensity
#endif
#if (XRT_ENABLE_HDF5 == 1)
#   if (XRT_ENABLE_TIFF == 1)
            ,
#   endif
            plugins::hdf5::HDF5Output
#endif
    > StandAlonePlugins;

    /* species plugins (with placeholder replaced by species) */
    typedef bmpl::vector<
#if ENABLE_PRINT_PARTICLES
            plugins::PrintParticles<bmpl::_1>
#endif
    > SpeciesPlugins;

    /* field plugins (with placeholder replaced by field) */
    typedef bmpl::vector<
#if ENABLE_PRINT_FIELDS && XRT_ENABLE_PNG == 1
            plugins::PrintField<bmpl::_1>
#endif
    > FieldPlugins;

    /* detector plugins (with placeholder replaced by detector) */
    typedef bmpl::vector<
#if ENABLE_PRINT_DETECTORS && (XRT_ENABLE_TIFF == 1 || XRT_ENABLE_HDF5 == 1)
            plugins::PrintDetector<bmpl::_1>
#endif
    > DetectorPlugins;

}  // namespace xrt
