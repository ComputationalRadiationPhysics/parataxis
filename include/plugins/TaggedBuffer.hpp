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
#include <vector>

namespace parataxis {
namespace plugins {

    template<class T_Buffer>
    class TaggedBuffer
    {
    public:
        using Buffer = T_Buffer;
        struct Entry
        {
            std::shared_ptr<Buffer> buffer;
            /** Time at which the buffer is valid and timestep length used to reach this */
            float_X time, dt;
            Entry(std::unique_ptr<Buffer> buffer, float_X time, float_X dt);
        };
        void push_back(std::unique_ptr<Buffer> buffer, float_X time, float_X dt);
        size_t size() const { return entries.size(); }
        bool empty() const { return entries.empty(); }
        void clear() { entries.clear(); }
        const Entry& operator[](size_t idx) const { return entries[idx]; }
        /** Return the start time of the dataset */
        float_X getFirstTime() const { return entries.front().time; }
        /** Return the end time of the dataset */
        float_X getLastTime() const { return entries.back().time; }
        /** Return the largest index for which entry.time<=time or -1 if no such entry exists */
        int findTimestep(float_X time, uint32_t curTimestep = 0) const;
    private:
        std::vector<Entry> entries;
    };

    template<class T_Buffer>
    TaggedBuffer<T_Buffer>::Entry::Entry(std::unique_ptr<Buffer> buffer, float_X time, float_X dt):
        buffer(buffer.release()), time(time), dt(dt)
    {}

    template<class T_Buffer>
    void TaggedBuffer<T_Buffer>::push_back(std::unique_ptr<Buffer> buffer, float_X time, float_X dt)
    {
        entries.push_back(Entry(std::move(buffer), time, dt));
    }

    template<class T_Buffer>
    int TaggedBuffer<T_Buffer>::findTimestep(float_X time, uint32_t curTimestep) const
    {
        if(empty() || getFirstTime() > time)
            return -1;
        if(time >= entries.back().time)
            return size() - 1;
        if(curTimestep >= size())
            curTimestep = size() - 1;
        // First fast forward assuming equal distance times
        float_X  diff = entries[curTimestep].time - time;
        curTimestep = PMaccMath::float2int_ru(diff / entries[curTimestep].dt);
        if(curTimestep >= size())
            curTimestep = size() - 1;
        // We increase the timestep till entry.time >= time (terminates at last entry due to check above)
        while(entries[curTimestep].time < time)
            ++curTimestep;
        // And now backward until entry.time <= curTime (guaranteed to terminate as for id=0 'time < curTime' is checked)
        while(entries[curTimestep].time > time)
            --curTimestep;
        return curTimestep;
    }
}  // namespace plugins
}  // namespace parataxis
