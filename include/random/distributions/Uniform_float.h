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
            return _curand_uniform(RNGMethod().get32Bits(state));
        }
    };

}  // namespace distributions
}  // namespace random
}  // namespace xrt
