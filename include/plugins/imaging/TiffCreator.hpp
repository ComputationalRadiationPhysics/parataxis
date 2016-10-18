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
#include <tiffWriter/tiffWriter.hpp>
#include <string>

namespace xrt {
namespace plugins {
namespace imaging {

    struct TiffCreator
    {
        template<class DBox, class T_Space>
        void operator() (const std::string& fileName, DBox data, T_Space dataSize)
        {
            tiffWriter::FloatImage<> img(fileName, dataSize.x(), dataSize.y());

            for (int32_t y = 0; y < dataSize.y(); ++y)
            {
                for (int32_t x = 0; x < dataSize.x(); ++x)
                {
                    /* Top left corner is 0,0*/
                    img(x, y) = data(Space2D(x, y));
                }
            }
            img.save();
        }
    };

}  // namespace imaging
}  // namespace plugins
}  // namespace xrt
