#pragma once

#include "xrtTypes.hpp"

namespace xrt {
namespace generators {

    /**
     * Sets all cells to the same value
     */
    template<typename T, class T_Config>
    struct Const{
        using Config = T_Config;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            return Config::value;
        }
    };


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
        static constexpr uint32_t offset  = Config::offset;
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
     * Generates a Cylinder (in 3D, circle for 2D)
     * Returns value if the current index is in the cylinder specified by radius and height
     * Center of first side is given by position
     */
    template<typename T, class T_Config>
    struct Cylinder
    {
        using Config = T_Config;
        using Position = typename Config::Position;
        static constexpr uint32_t height = Config::height;
        static constexpr uint32_t radius = Config::radius;
        /** Value used */
        static constexpr float_64 value = Config::value;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            if(simDim == 3 && (idx.x() < Position::x::value || idx.x() >= Position::x::value + height))
                return 0;
            using PMaccMath::abs2;
            uint32_t r2 = abs2(idx[simDim - 2] - Position::toRT()[simDim - 2]) + abs2(idx[simDim - 1] - Position::toRT()[simDim - 1]);
            if(r2 < radius*radius)
                return value;
            else
                return 0;
        }
    };

    /**
     * Create an "edge", that is everything below a linear function (m*x+n) is filled
     */
    template<typename T, class T_Config>
    struct Edge
    {
        using Config = T_Config;
        static constexpr uint32_t roomPos = Config::roomPos;
        static constexpr uint32_t roomWidth = Config::roomWidth;

        static_assert(roomWidth > 0, "RoomWidth must be > 0");
        static_assert(simDim == 3, "Only for 3D defined");
        static constexpr float_32 m = Config::m;
        static constexpr float_32 n = Config::n;
        /** Value used */
        static constexpr float_64 value = Config::value;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            if(idx.x() < roomPos || idx.x() >= roomPos + roomWidth)
                return 0;
            if(idx.y() < m * idx.z() + n)
                return value;
            else
                return 0;
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

        static_assert(simDim == 2 || roomWidth > 0, "RoomWidth must be > 0");
        static_assert(simDim == 2 || simDim == 3, "Only for 2D and 3D defined");

        // Offset is the middle between the slits, spacing is the distance between the slits centers and width is the size of each slit
        static constexpr uint32_t topSlitStart = Config::offset - Config::spacing/2 - Config::width/2;
        static constexpr uint32_t botSlitStart = topSlitStart + Config::spacing;

        // Check for possible underflow
        static_assert(topSlitStart < Config::offset, "Invalid config. offset to small or spacing and/or width to big");
        /** Value used */
        static constexpr float_64 value = Config::value;


        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            if(simDim == 3 && (idx.x() < roomPos || idx.x() >= roomPos + roomWidth))
                return 0;
            // Note: Default (not rotated) this is y in 3D
            auto idxY = idx[Config::rotated ? simDim - 1: simDim - 2];
            if((idxY >= topSlitStart  && idxY < topSlitStart  + Config::width) ||
               (idxY >= botSlitStart && idxY < botSlitStart + Config::width))
                return value;
            else
                return 0;
        }
    };

    /**
     * Creates a line, which value is the cell index in which the line currently is
     * (Mainly for testing purposes)
     */
    template<typename T, class T_Config>
    struct RaisingLine
    {
        using Config = T_Config;
        /** Dimension in which the line extents */
        static constexpr uint32_t nDim  = Config::nDim;
        /** Offset where the line/plane is drawn */
        static constexpr uint32_t offsetX  = Config::offsetX;
        static constexpr uint32_t offsetOther = Config::offsetOther;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            static_assert(nDim < simDim, "Dimension for Line generator must be smaller then total # of dims");
            if(idx.x() != offsetX)
                return 0;
            if(idx[nDim == 1 ? 2 : 1] != offsetOther)
                return 0;
            return idx[nDim == 1 ? 1 : 2] + 1;
        }
    };

    template<typename T, class T_Config>
    struct CombinedDoubleSlit
    {
        DoubleSlit<T, typename T_Config::Cfg1> gen1;
        DoubleSlit<T, typename T_Config::Cfg2> gen2;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            const T val1 = gen1(idx);
            const T val2 = gen2(idx);
            if(T_Config::useMax)
                return PMaccMath::max(val1, val2);
            else
                return PMaccMath::min(val1, val2);
        }
    };

}  // namespace generators
}  // namespace xrt
