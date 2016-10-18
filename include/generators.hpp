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

#include "parataxisTypes.hpp"

namespace parataxis {
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
    struct Strips
    {
        /** Offset till start of first line */
        static constexpr uint32_t offset  = T_Config::offset;
        /** Width of line */
        static constexpr uint32_t size = T_Config::size;
        /** Spacing between lines */
        static constexpr uint32_t spacing = T_Config::spacing;

        static constexpr uint32_t offsetX  = T_Config::offsetX;
        static constexpr uint32_t sizeX = T_Config::sizeX;
        /** Value used */
        static constexpr float_64 value = T_Config::value;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            if(idx.x() < offsetX || idx.x() >= offsetX + sizeX)
                return 0;
            if(idx.y() < offset)
                return 0;
            if((idx.y() - offset) % (size + spacing) < size)
                return value;
            else
                return 0;
        }
    };

    namespace detail{

        template<class T_Generator, class T_NewCfg>
        struct ReplaceCfg;

        template<template<typename, typename> class T_Generator, class T, class T_OldCfg, class T_NewCfg>
        struct ReplaceCfg< T_Generator<T, T_OldCfg>, T_NewCfg >
        {
            using type = T_Generator<T, T_NewCfg>;
        };

    }  // namespace detail

    template<typename T, class T_Cfg>
    struct CombinedGenerator
    {
        using Generator1 = typename detail::ReplaceCfg<Resolve_t<typename T_Cfg::Gen1>, typename T_Cfg::Cfg1>::type;
        using Generator2 = typename detail::ReplaceCfg<Resolve_t<typename T_Cfg::Gen2>, typename T_Cfg::Cfg2>::type;

        Generator1 gen1;
        Generator2 gen2;

        template<class T_Idx>
        HDINLINE T operator()(T_Idx&& idx) const
        {
            const T val1 = gen1(idx);
            const T val2 = gen2(idx);
            if(T_Cfg::useMax)
                return PMaccMath::max(val1, val2);
            else
                return PMaccMath::min(val1, val2);
        }
    };

}  // namespace generators
}  // namespace parataxis
