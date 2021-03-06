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
namespace initPolicies {

    /**
     * Functor that returns a constant value as the distribution for any param
     */
    template<class T_Cfg>
    struct ConstDistribution
    {
        ConstDistribution(uint32_t /*timestep*/){}

        DINLINE void
        init(Space /*localCellIdx*/) const
        {}

        DINLINE uint32_t
        operator()(float_X numPhotons) const
        {
            return numPhotons > 0 ? T_Cfg::numParts : 0;
        }
    };

}  // namespace initPolicies
}  // namespace particles
}  // namespace parataxis
