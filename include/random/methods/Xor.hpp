#pragma once

#include "xrtTypes.hpp"
#include <curand_kernel.h>

namespace xrt {
namespace random {
namespace methods {

    /** Uses the CUDA XORWOW RNG */
    class Xor
    {
    public:
        typedef curandStateXORWOW_t StateType;

        DINLINE void
        init(StateType& state, uint32_t seed, uint32_t subsequence = 0, uint32_t offset = 0) const
        {
            curand_init(seed, subsequence, offset, &state);
        }

        DINLINE uint32_t
        get32Bits(StateType& state) const
        {
            return curand(&state);
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace xrt
