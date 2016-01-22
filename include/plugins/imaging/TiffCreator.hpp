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
