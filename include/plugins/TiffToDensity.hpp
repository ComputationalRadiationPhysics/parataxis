#pragma once

#include "xrtTypes.hpp"
#include "plugins/ISimulationPlugin.hpp"

#include <tiffWriter/tiffWriter.hpp>

namespace xrt {
namespace plugins {

    class TiffToDensity : public ISimulationPlugin
    {
        std::string name;
        std::string prefix;

        std::string filePath;
        unsigned firstIdx, lastIdx, minSizeFiller, x0, y0;
        std::vector<unsigned> simOffset;
        char filler;
        int size;
        bool repeat;

    public:
        TiffToDensity():
            name("TiffToDensity: Converts one or more tiff images to the density field"),
            prefix("tiff2Dens.")
        {
            Environment::get().PluginConnector().registerPlugin(this);
        }

        void pluginRegisterHelp(po::options_description& desc) override
        {
            desc.add_options()
                ((prefix + "inputFile").c_str(), po::value<std::string>(&filePath), "Input file to use, can contain %i as a placeholder for 3D FFTs")
                ((prefix + "firstIdx").c_str(), po::value<unsigned>(&firstIdx)->default_value(0), "first index to use")
                ((prefix + "lastIdx").c_str(), po::value<unsigned>(&lastIdx)->default_value(0), "last index to use")
                ((prefix + "repeat").c_str(), po::value<bool>(&repeat)->default_value(true), "Repeat single image along X-axis of simulation")
                ((prefix + "minSize").c_str(), po::value<unsigned>(&minSizeFiller)->default_value(0), "Minimum size of the string replaced for %i")
                ((prefix + "fillChar").c_str(), po::value<char>(&filler)->default_value('0'), "Char used to fill the string to the minimum size")
                ((prefix + "xStart").c_str(), po::value<unsigned>(&x0)->default_value(0), "Offset in x-Direction of image")
                ((prefix + "yStart").c_str(), po::value<unsigned>(&y0)->default_value(0), "Offset in y-Direction of image")
                ((prefix + "simOff").c_str(), po::value<std::vector<unsigned>>(&simOffset)->multitoken(), "Offset into the simulation")
                ((prefix + "size").c_str(), po::value<int>(&size)->default_value(-1), "Size of the image to use (-1=all)")
                ;
        }

        std::string pluginGetName() const override
        {
            return name;
        }

        void notify(uint32_t currentStep) override
        {
            PMacc::log< XRTLogLvl::IN_OUT >("Loading density field");

            if(firstIdx == lastIdx || filePath.find("%i") == std::string::npos)
                load2D();
            else
                load3D();
        }

       void checkpoint(uint32_t currentStep, const std::string checkpointDirectory) override
       {}
       void restart(uint32_t restartStep, const std::string restartDirectory) override
       {}
    protected:
        void pluginLoad() override
        {
            // If a file is given, notify once
            if(!filePath.empty())
                Environment::get().PluginConnector().setNotificationPeriod(this, std::numeric_limits<uint32>::max());
        }

        std::string
        getFilledNumber(unsigned num)
        {
            std::string s(std::to_string(num));
            while(s.size()<minSizeFiller)
                s = filler + s;
            return s;
        }

        std::string
        replace(std::string str, const std::string& from, const std::string& to) {
            size_t start_pos = str.find(from);
            if(start_pos == std::string::npos)
                return str;
            return str.replace(start_pos, from.length(), to);
        }

        std::string getFilePath(unsigned num)
        {
            return replace(filePath, "%i", getFilledNumber(firstIdx));
        }

        void load2D()
        {
            auto localDomain = Environment::get().SubGrid().getLocalDomain();
            Space localSize = localDomain.size + 2 * cellDescription_ ->getGuardingSuperCells() * SuperCellSize::toRT();
            Space simOffset = convertToSpace(this->simOffset, 0, "");
            Space offset = simOffset - localDomain.offset + cellDescription_ ->getGuardingSuperCells() * SuperCellSize::toRT();
            if(simDim == 3)
            {
                if(offset.x() >= localSize.x())
                    return; // Starts after this domain
                if(offset.x() < 0 && !repeat)
                    return; // Starts before this domain and we don't repeat
            }
            tiffWriter::FloatImage<> img(getFilePath(firstIdx));
            const Space2D imgSize(img.getWidth(), img.getHeight());
            const Space2D offset2D = offset.shrink<2>(1);
            const Space2D localSize2D = localSize.shrink<2>(1);
            const int y0 = std::max(0, offset2D.x());
            const int y1 = std::min(localSize2D.x(), imgSize.x() + offset2D.x());
            const int z0 = std::max(0, offset2D.y());
            const int z1 = std::min(localSize2D.y(), imgSize.y() + offset2D.y());
            if(z0 >= z1 || y0 >= y1)
                return;
            auto& dc = Environment::get().DataConnector();
            auto& densityField = dc.getData<DensityField>(DensityField::getName(), true);
            auto densityBox = densityField.getHostDataBox();
            int x0 = std::max(0, offset.x());
            int x1 = repeat ? localSize.x() : x0 + 1;
            for(int z = z0; z < z1; z++)
            {
                for(int y = y0; y < y1; y++)
                {
                    for(int x = x0; x < x1; x++)
                    {
                        Space idx(x, y, z);
                        Space2D idxImg = idx.shrink<2>(1) - offset2D;
                        densityBox(idx) = img(idxImg.x(), idxImg.y());
                    }
                }
            }
            densityField.getGridBuffer().hostToDevice();
            dc.releaseData(DensityField::getName());
        }

        void load3D()
        {

        }
    };

}  // namespace plugins
}  // namespace xrt
