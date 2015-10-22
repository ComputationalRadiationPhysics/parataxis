#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace random {
namespace distributions {

    /**
     * Returns a random 32 bit unsigned integer
     */
    template<class T_RNGMethod>
    class Uniform_uint32
    {
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;
    public:
        DINLINE uint32_t
        operator()(StateType& state)
        {
            return static_cast<uint32_t>(RNGMethod().get32Bits(state));
        }
    };

}  // namespace distributions
}  // namespace random
}  // namespace xrt
