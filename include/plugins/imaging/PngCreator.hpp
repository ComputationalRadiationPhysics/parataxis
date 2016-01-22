#pragma once

#include "xrtTypes.hpp"
#include <pngwriter.h>
#include <string>

namespace xrt {
namespace plugins {
namespace imaging {

    struct PngCreator
    {
        template<class DBox, class T_Space>
        void operator() (const std::string& fileName, DBox data, T_Space dataSize)
        {
            pngwriter png(dataSize.x(), dataSize.y(), 0, fileName.c_str());
            png.setcompressionlevel(9);

            for (int32_t y = 0; y < dataSize.y(); ++y)
            {
                for (int32_t x = 0; x < dataSize.x(); ++x)
                {
                    float p = data(Space2D(x, y));
                    /* Png writer coordinates start at 1, 1 in top left corner */
                    png.plot(x + 1, y + 1, p, p, p);
                }
            }
            png.close();
        }
    };

}  // namespace imaging
}  // namespace plugins
}  // namespace xrt
