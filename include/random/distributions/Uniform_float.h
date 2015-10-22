#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace random {
namespace distributions {

    /**
     * Returns a random float value in [0,1)
     */
    template<class T_RNGMethod>
    class Uniform_float
    {
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;
    public:
        DINLINE float
        operator()(StateType& state)
        {
            /** Simply divide the 32 bit value by the maximum possible value */
            return static_cast<float>(RNGMethod().get32Bits(state)) / float(1 << 31);
        }
    };

}  // namespace distributions
}  // namespace random
}  // namespace xrt
