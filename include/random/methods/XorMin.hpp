#pragma once

#include "xrtTypes.hpp"
#include "random/methods/Xor.hpp"
#include <curand_kernel.h>

namespace xrt {
namespace random {
namespace methods {

    /** Uses the CUDA XORWOW RNG but does not store state members required for normal distribution*/
    class XorMin
    {
    public:
        class StateType
        {
        public:
            unsigned int d, v[5];

            HDINLINE StateType()
            {}

            HDINLINE StateType(const curandStateXORWOW_t& other): d(other.d), v{other.v[0], other.v[1], other.v[2], other.v[3], other.v[4]}
            {
                static_assert(sizeof(v) == sizeof(other.v), "Unexpected sizes");
            }
        };

        DINLINE void
        init(StateType& state, uint32_t seed, uint32_t subsequence = 0, uint32_t offset = 0) const
        {
            curandStateXORWOW_t tmpState;
            curand_init(seed, subsequence, offset, &tmpState);
            state = tmpState;
        }

        HDINLINE uint32_t
        get32Bits(StateType& state) const
        {
            /* This generator uses the xorwow formula of
            www.jstatsoft.org/v08/i14/paper page 5
            Has period 2^192 - 2^32.
            */
            uint32_t t;
            t = (state.v[0] ^ (state.v[0] >> 2));
            state.v[0] = state.v[1];
            state.v[1] = state.v[2];
            state.v[2] = state.v[3];
            state.v[3] = state.v[4];
            state.v[4] = (state.v[4] ^ (state.v[4] <<4)) ^ (t ^ (t << 1));
            state.d += 362437;
            return state.v[4] + state.d;
        }
    };

}  // namespace methods
}  // namespace random
}  // namespace xrt
