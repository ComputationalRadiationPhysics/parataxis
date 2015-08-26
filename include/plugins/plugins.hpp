#pragma once

#include "simulation_defines.hpp"
#include "plugins/PrintParticles.hpp"
#include "plugins/PrintField.hpp"
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
#if ENABLE_PRINT_Field
            plugins::PrintParticles<bmpl::_1>
#endif
    > FieldPlugins;

}  // namespace xrt
