#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace math {

    /** Rounds the given float value to the nearest integer.
     *  .5 cases are rounded towards +infinity
     *  Usable in constexpr
     */
    constexpr int floatToIntRound(const float_32 val)
    {
        return static_cast<int>(val + float_32(.5));
    }

    /** Rounds the given float value to the nearest integer.
     *  .5 cases are rounded towards +infinity
     *  Usable in constexpr
     */
    constexpr int floatToIntRound(const float_64 val)
    {
        return static_cast<int>(val + float_64(.5));
    }

}  // namespace math
}  // namespace xrt
