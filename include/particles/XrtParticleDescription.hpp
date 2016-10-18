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

#include "particles/ProcessLeavingParticles.hpp"
#include <particles/ParticleDescription.hpp>

namespace xrt {
namespace particles {

    template<
        typename T_Name,
        typename T_SuperCellSize,
        typename T_ValueTypeSeq,
        typename T_Flags = bmpl::vector0<>,
        typename T_MethodsList = bmpl::vector0<>,
        typename T_FrameExtensionList = bmpl::vector0<>
    >
    using ParticleDescription = PMacc::ParticleDescription<
            T_Name,
            T_SuperCellSize,
            T_ValueTypeSeq,
            T_Flags,
            PMacc::HandleGuardRegion<
                PMacc::particles::policies::ExchangeParticles,
                ProcessLeavingParticles
            >,
            T_MethodsList,
            T_FrameExtensionList
    >;

}  // namespace particles
}  // namespace xrt
