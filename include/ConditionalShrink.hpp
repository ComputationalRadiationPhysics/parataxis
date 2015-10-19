#pragma once

#include "xrtTypes.hpp"

namespace xrt {

    namespace detail {

        template<int32_t T_shrinkDim, bool T_doShrink = T_shrinkDim >= 0>
        struct ConditionalShrink
        {
            template<typename T_Space>
            HDINLINE T_Space
            operator()(T_Space&& space) const
            {
                return space;
            }
        };

        template<int32_t T_shrinkDim>
        struct ConditionalShrink<T_shrinkDim, true>
        {
            template<typename T_Space>
            HDINLINE PMacc::DataSpace<(remove_reference_t<T_Space>::Dim) - 1>
            operator()(T_Space&& space) const
            {
                constexpr uint32_t oldDim = remove_reference_t<T_Space>::Dim;
                static_assert(T_shrinkDim < oldDim, "Cannot remove a dimension that is not there");
                return space.template shrink<oldDim - 1>(T_shrinkDim + 1);
            }
        };

    }  // namespace detail

    /**
     * Functor that shrinks the given DataSpace by removing the given dimension
     * If the dimension is <0 the DataSpace returned unchanged
     */
    template<int32_t T_shrinkDim>
    using ConditionalShrink = detail::ConditionalShrink<T_shrinkDim>;

}  // namespace xrt
