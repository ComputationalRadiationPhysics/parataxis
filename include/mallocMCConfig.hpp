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
    typedef std::integral_constant<int, 2*1024*1024> pagesize;
    /* accessBlocks, regionSize and wasteFactor are not finally selected
       and might be performance sensitive*/
    typedef std::integral_constant<int, 4> accessblocks;
    typedef std::integral_constant<int, 8> regionsize;
    typedef std::integral_constant<int, 2> wastefactor;
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
