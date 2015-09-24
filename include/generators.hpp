#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace generators {

    /**
     * Generates a line (in 2D, --> Point in 1D/Area in 3D)
     * Returns value if current index in \tT_fixedDim equals pos, 0 otherwise
     */
    template<typename T, class T_Config>
    struct Line{
        using Config = T_Config;
        /** Dimension which the line/plan cuts */
        static constexpr uint32_t nDim  = Config::nDim;
        /** Offset where the line/plane is drawn */
        static constexpr size_t offset  = Config::offset;
        /** Value used */
        static constexpr float_64 value = Config::value;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            static_assert(nDim < simDim, "Dimension for Line generator must be smaller then total # of dims");
            return idx[nDim] == offset ? value : 0;
        }
    };

    /**
     * Generates a cuboid (in 3D, flat for lower dim)
     * Returns value if the current index is in the cuboid specified by offset and size
     */
    template<typename T, class T_Config>
    struct Cuboid
    {
        using Config = T_Config;
        using Offset = typename Config::Offset;
        using Size   = typename Config::Size;
        using End    = typename PMacc::math::CT::add<Offset, Size>::type;
        /** Value used */
        static constexpr float_64 value = Config::value;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            for(int i = 0; i < simDim; ++i)
                if(idx[i] < Offset::toRT()[i] || idx[i] >= End::toRT()[i])
                    return 0;
            return value;
        }
    };

    /**
     * Creates a double slit
     */
    template<typename T, class T_Config>
    struct DoubleSlit
    {
        using Config = T_Config;
        static constexpr uint32_t roomPos = Config::roomPos;
        static constexpr uint32_t roomWidth = Config::roomWidth;
        static constexpr uint32_t offset = Config::offset;
        static constexpr uint32_t width = Config::width;
        static constexpr uint32_t spacing = Config::spacing;
        static constexpr uint32_t offset2 = offset + width + spacing;
        /** Value used */
        static constexpr float_64 value = Config::value;

        static_assert(simDim == 2 || roomWidth > 0, "RoomWidth must be > 0");
        static_assert(simDim == 2 || simDim == 3, "Only for 2D and 3D defined");

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            if(simDim == 3 && (idx[0] < roomPos || idx[0] >= roomPos + roomWidth))
                return 0;
            auto idxY = idx[simDim - 2];
            if((idxY >= offset  && idxY < offset  + width) ||
               (idxY >= offset2 && idxY < offset2 + width))
                return 0;
            else
                return 1;
        }
    };

}  // namespace generators
}  // namespace xrt
