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
        unsigned firstIdx, lastIdx, minSizeFiller, imgOffsetX, imgOffsetY;
        std::vector<unsigned> simOffset;
        char filler;
        int maxImgSize;
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
                ((prefix + "fillChar").c_str(), po::value<char>(&filler)->default_value('0'), "Char used to fill the string to the minimum maxImgSize")
                ((prefix + "xStart").c_str(), po::value<unsigned>(&imgOffsetX)->default_value(0), "Offset in x-Direction of image")
                ((prefix + "yStart").c_str(), po::value<unsigned>(&imgOffsetY)->default_value(0), "Offset in y-Direction of image")
                ((prefix + "simOff").c_str(), po::value<std::vector<unsigned>>(&simOffset)->multitoken(), "Offset into the simulation")
                ((prefix + "size").c_str(), po::value<int>(&maxImgSize)->default_value(-1), "Size of the image to use (-1=all)")
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
            // If a file is not given, don't register
            if(filePath.empty())
                return;
            // Notify once
            Environment::get().PluginConnector().setNotificationPeriod(this, std::numeric_limits<uint32>::max());
        }

        std::string
        getFilledNumber(unsigned num) const
        {
            std::string s(std::to_string(num));
            while(s.size()<minSizeFiller)
                s = filler + s;
            return s;
        }

        std::string
        replace(std::string str, const std::string& from, const std::string& to) const
        {
            size_t start_pos = str.find(from);
            if(start_pos == std::string::npos)
                return str;
            return str.replace(start_pos, from.length(), to);
        }

        std::string getFilePath(unsigned num) const
        {
            return replace(filePath, "%i", getFilledNumber(firstIdx));
        }

        void load2D()
        {
            auto localDomain = Environment::get().SubGrid().getLocalDomain();
            // Get local size including guards on both sides
            Space localSize = localDomain.size + 2 * cellDescription_ ->getGuardingSuperCells() * SuperCellSize::toRT();
            // Offset into the simulation
            Space simOffset = convertToSpace(this->simOffset, 0, "");
            // Offset from the start of the local area to the start of the image (can be negative, if the image starts in a previous slice:
            // == difference of sim offset and local offset + the guarding cells at the beginning of the sim
            Space offset = simOffset - localDomain.offset + cellDescription_ ->getGuardingSuperCells() * SuperCellSize::toRT();
            if(simDim == 3)
            {
                // For 3D sims check if our "slice" in x direction is in the range that should be filled
                // and exit early if it is not (avoid loading the image altogether)
                if(offset.x() >= localSize.x())
                    return; // Starts after this domain
                if(offset.x() < 0 && !repeat)
                    return; // Starts before this domain and we don't repeat
            }
            tiffWriter::FloatImage<> img(getFilePath(firstIdx));
            Space2D imgSize(img.getWidth(), img.getHeight());
            Space2D imgOffset(this->imgOffsetX, this->imgOffsetY);
            // Because of the offset the image appears smaller
            imgSize -= imgOffset;
            if(maxImgSize >= 0)
            {
                // Use the given the size but stay within the bounds
                imgSize.x() = std::min(imgSize.x(), maxImgSize);
                imgSize.y() = std::min(imgSize.y(), maxImgSize);
            }
            // Get size and offset in the plane
            const Space2D offset2D = offset.shrink<2>(1);
            const Space2D localSize2D = localSize.shrink<2>(1);
            // Get bounds for the local area/volume
            // x >= 0 && x >= offset
            const int x0 = std::max(0, offset.x());
            // x < localSize or for no repeat mode use only 1 slice
            const int x1 = repeat ? localSize.x() : x0 + 1;
            // y >= 0 && y >= offset
            const int y0 = std::max(0, offset2D.x());
            // y < localSize && y - offset < imgSize ( or y < imgSize + offset)
            // Remember: y is in local coords, offset is offset to start of image as seen in local coords
            // --> Start of image is at y = offset -> imgIdx = y - offset
            const int y1 = std::min(localSize2D.x(), imgSize.x() + offset2D.x());
            // Same as in y
            const int z0 = std::max(0, offset2D.y());
            const int z1 = std::min(localSize2D.y(), imgSize.y() + offset2D.y());
            // Check if we need to insert a part of the img (Note: X is already checked above for 3D only)
            // (e.g. false if simulation is split in y and we only insert into the top part)
            if(z0 >= z1 || y0 >= y1)
                return;
            auto& dc = Environment::get().DataConnector();
            auto& densityField = dc.getData<DensityField>(DensityField::getName(), true);
            auto densityBox = densityField.getHostDataBox();
            // Combine both offsets
            imgOffset -= offset2D;
            for(int z = z0; z < z1; z++)
            {
                for(int y = y0; y < y1; y++)
                {
                    for(int x = x0; x < x1; x++)
                    {
                        Space idx(x, y, z);
                        Space2D idxImg = idx.shrink<2>(1) + imgOffset;
                        densityBox(idx) = img(idxImg.x(), idxImg.y());
                    }
                }
            }
            // Sync to device
            densityField.getGridBuffer().hostToDevice();
            dc.releaseData(DensityField::getName());
        }

        void load3D()
        {

        }
    };

}  // namespace plugins
}  // namespace xrt
