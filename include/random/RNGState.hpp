#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace random {

    /**
     * Class representing the state of a random number generator
     * Requires a policy that provides a state pointer via getStatePtr()
     * and a typedef for StatePtr
     */
    template<class T_RNGMethod>
    class RNGState
    {
    public:
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;
        typedef typename RNGMethod::StatePtr StatePtr;

        HDINLINE RNGState()
        {}

        HDINLINE RNGState(const StateType& other): state(other)
        {}

        DINLINE StatePtr
        getStatePtr()
        {
            return &state;
        }
    private:
        StateType state;
    };

}  // namespace random
}  // namespace xrt
