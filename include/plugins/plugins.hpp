#pragma once

#include "simulation_defines.hpp"
#include "plugins/PrintParticles.hpp"
#ifdef XRT_ENABLE_PNG
#   include "plugins/PrintField.hpp"
#endif
#ifdef XRT_ENABLE_TIFF
#   include "plugins/PrintDetector.hpp"
#endif
#include <boost/mpl/vector.hpp>

namespace xrt {

    /* stand alone plugins (no placeholders) */
    typedef bmpl::vector<
    > StandAlonePlugins;

    /* species plugins (with placeholder replaced by species) */
    typedef bmpl::vector<
#if ENABLE_PRINT_PARTICLES
            plugins::PrintParticles<bmpl::_1>
#endif
    > SpeciesPlugins;

    /* field plugins (with placeholder replaced by field) */
    typedef bmpl::vector<
#if ENABLE_PRINT_FIELDS && defined(XRT_ENABLE_PNG)
            plugins::PrintField<bmpl::_1>
#endif
    > FieldPlugins;

    /* detector plugins (with placeholder replaced by detector) */
    typedef bmpl::vector<
#if ENABLE_PRINT_DETECTORS && defined(XRT_ENABLE_TIFF)
            plugins::PrintDetector<bmpl::_1>
#endif
    > DetectorPlugins;

}  // namespace xrt
