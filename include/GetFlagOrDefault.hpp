#pragma once

#include "xrtTypes.hpp"

namespace xrt {

    /**
     * When a flag with the given key exists in the object the resolved
     * flag type is returned. Otherwise the default is resolved and returned
     */
    template<
        class T_Object,
        typename T_Key,
        typename T_Default
    >
    class GetFlagOrDefault
    {
        using Object = T_Object;
        using Key = T_Key;
        using Default = T_Default;

        using HasFlag = HasFlag_t<Object, Key>;
        using Flag    = GetFlagType_t<Object, Key>;
        using Selected =
                conditional_t<
                    HasFlag::value,
                    Flag,
                    Default
                >;

    public:

        using type = Resolve_t<Selected>;
    };

    template<
        class T_Object,
        typename T_Key,
        typename T_Default
    >
    using GetFlagOrDefault_t = typename GetFlagOrDefault<T_Object, T_Key, T_Default>::type;

}  // namespace xrt
