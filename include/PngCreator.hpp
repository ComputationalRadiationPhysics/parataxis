#pragma once

#include "types.h"
#include <pngwriter.h>
#include <sstream>

namespace xrt
{
    struct PngCreator
    {
        template<class DBox>
        void operator() (uint32_t currentStep, DBox data, Space dataSize)
        {
            std::stringstream fileName;
            fileName << "xrt_";
            fileName << std::setw(6) << std::setfill('0') << currentStep;
            fileName << ".png";
            pngwriter png(dataSize.x(), dataSize.y(), 0, fileName.str().c_str());
            png.setcompressionlevel(9);

            for (int y = 0; y < dataSize.y(); ++y)
            {
                for (int x = 0; x < dataSize.x(); ++x)
                {
                    float p = data(Space(y, x));
                    png.plot(x + 1, dataSize.y() - y, p, p, p);
                }
            }
            png.close();
        }
    };

}
