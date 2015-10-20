#pragma once

#include "xrtTypes.hpp"

namespace PMacc{namespace algorithms{namespace math{

    template<>
    struct Abs<uint32_t>
    {
        typedef uint32_t result;

        HDINLINE uint32_t operator( )(uint32_t value)
        {
            return value;
        }
    };

    template<>
    struct Abs2<uint32_t>
    {
        typedef uint32_t result;

        HDINLINE uint32_t operator( )(const uint32_t& value )
        {
            return value*value;
        }
    };

}}}
