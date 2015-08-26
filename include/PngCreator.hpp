#pragma once

#include "xrtTypes.hpp"
#include <pngwriter.h>
#include <string>

namespace xrt
{
    struct PngCreator
    {
        template<class DBox, class T_Space>
        void operator() (const std::string& fileName, DBox data, T_Space dataSize)
        {
            pngwriter png(dataSize.x(), dataSize.y(), 0, fileName.c_str());
            png.setcompressionlevel(9);

            for (int y = 0; y < dataSize.y(); ++y)
            {
                for (int x = 0; x < dataSize.x(); ++x)
                {
                    float p = data(Space2D(x, y));
                    png.plot(x + 1, dataSize.y() - y, p, p, p);
                }
            }
            png.close();
        }
    };

}
