/**
 * Copyright 2013-2016 Axel Huebl, Heiko Burau, Rene Widera, Alexander Grund
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
 
/* Our mallocMC config
 * IMPORTANT: Include this one first!
 */
#pragma once

#include <mallocMC/mallocMC.hpp>
#include <type_traits>

// configure the CreationPolicy "Scatter"
struct ScatterConfig
{
    /* 2MiB page can hold around 256 particle frames */
    typedef std::integral_constant<uint32_t, 2*1024*1024> pagesize;
    /* accessBlocks, regionSize and wasteFactor are not finally selected
       and might be performance sensitive*/
    typedef std::integral_constant<uint32_t, 4> accessblocks;
    typedef std::integral_constant<uint32_t, 8> regionsize;
    typedef std::integral_constant<uint32_t, 2> wastefactor;
    /* resetFreedPages is used to minimize memory fragmentation while different
       frame sizes were used*/
    typedef std::integral_constant<bool, true> resetfreedpages;
};

// Define a new allocator and call it ScatterAllocator
// which resembles the behavior of ScatterAlloc
typedef mallocMC::Allocator<
    mallocMC::CreationPolicies::Scatter<ScatterConfig>,
    mallocMC::DistributionPolicies::Noop,
    mallocMC::OOMPolicies::ReturnNull,
    mallocMC::ReservePoolPolicies::SimpleCudaMalloc,
    mallocMC::AlignmentPolicies::Shrink<>
> ScatterAllocator;

//use ScatterAllocator to replace malloc/free
MALLOCMC_SET_ALLOCATOR_TYPE( ScatterAllocator );
