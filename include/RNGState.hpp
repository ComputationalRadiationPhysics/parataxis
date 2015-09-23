#pragma once

#include "xrtTypes.hpp"

namespace xrt {

    /**
     * Class representing the state of a random number generator
     * Requires a policy that provides a state pointer via getStatePtr()
     * and a typedef for StatePtr
     */
    template<class T_RNGMethod>
    class RNGState: public T_RNGMethod
    {
    public:
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StatePtr StatePtr;

        HDINLINE RNGState()
        {}

        HDINLINE RNGState(const RNGMethod& other): RNGMethod(other)
        {}

        DINLINE StatePtr
        getStatePtr()
        {
            return RNGMethod::getStatePtr();
        }
    };

}  // namespace xrt
