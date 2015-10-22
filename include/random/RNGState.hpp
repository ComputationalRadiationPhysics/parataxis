#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace random {

    /**
     * Class representing the state of a random number generator
     * Requires a policy that provides a typedef for StateType
     */
    template<class T_RNGMethod>
    class RNGState
    {
    public:
        typedef T_RNGMethod RNGMethod;
        typedef typename RNGMethod::StateType StateType;

        HDINLINE RNGState()
        {}

        HDINLINE RNGState(const StateType& other): state(other)
        {}

        HDINLINE StateType&
        getState()
        {
            return state;
        }
    private:
        StateType state;
    };

}  // namespace random
}  // namespace xrt
