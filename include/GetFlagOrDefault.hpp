/**
 * Copyright 2015-2016 Alexander Grund
 *
 * This file is part of ParaTAXIS.
 *
 * ParaTAXIS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ParaTAXIS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.
 */
 
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
