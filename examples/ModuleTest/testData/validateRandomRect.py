import sys
import os
from numpy import *
import unittest
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "buildSystem"))
from ParamParser import ParamParser

class TestSingleCell(unittest.TestCase):
    def testDetector(self):
        cmakeFlags = os.environ["TEST_CMAKE_FLAGS"].split(" ")
        paramOverwrites = None
        for flag in cmakeFlags:
            if flag.startswith("-DPARAM_OVERWRITES:LIST"):
                paramOverwrites = flag.split("=", 1)[1].split(";")

        params = ParamParser()
        if paramOverwrites:
            for param in paramOverwrites:
                if param.startswith("-D"):
                    param = param[2:].split("=")
                    params.AddDefine(param[0], param[1])
        params.ParseFolder(os.environ["TEST_OUTPUT_PATH"] + "/simulation_defines/param")

        params.SetCurNamespace("xrt::detector::PhotonDetector")
        DetDist = params.GetNumber("distance")
        DetCellSize = [params.GetNumber("cellWidth"), params.GetNumber("cellHeight")]
        DetType = params.GetValue("IncomingParticleHandler")

        params.SetCurNamespace("xrt::initialDensity")
        assert(params.GetValue("Generator") == "AvailableGenerators::Cuboid"), "Must use cuboid generator"
        DensityPos = params.GetVector("AvailableGenerators::Cuboid::Offset")
        DensityPos2D = array(DensityPos[1:])

        params.SetCurNamespace("xrt")
        ScatterAngle = params.GetNumber("particles::scatterer::direction::DoubleRandDirection::angle")
        ScatterAngle = [ScatterAngle, ScatterAngle]
        SimCellSize = [params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")]

        SimSize = list(map(int, os.environ["TEST_GRID_SIZE"].split(" ")))
        SimSize2D = array(SimSize[1:])

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            DetSize = im.size

            ## Calculation

            ScatterOffsets = tan(ScatterAngle) * DetDist
            ScatterOffsetsBegin = (DensityPos2D - SimSize2D / 2) * SimCellSize[1:] - ScatterOffsets
            ScatterOffsetsEnd = (DensityPos2D - SimSize2D / 2) * SimCellSize[1:] + ScatterOffsets
            # SimZ = DetX, SimY = DetY
            ScatterOffsetsBegin = flipud(ScatterOffsetsBegin)
            ScatterOffsetsEnd = flipud(ScatterOffsetsEnd)
            PosOnDetBegin = (ScatterOffsetsBegin / DetCellSize + array(DetSize) / 2).astype(int)
            PosOnDetEnd   = (ScatterOffsetsEnd   / DetCellSize + array(DetSize) / 2).astype(int)
            
            # Ignore outermost region (half cell hits which might change the count)
            PosOnDetBegin += [1, 1]
            imgData = array(im)
            imgData = imgData[PosOnDetBegin[0]:PosOnDetEnd[0], PosOnDetBegin[1]:PosOnDetEnd[1]]
            avgHitcount = average(imgData)
            minHitcount = amin(imgData)
            maxHitcount = amax(imgData)
            stdDev = std(imgData)
            print("Hitcount:", minHitcount, "-", maxHitcount, "average =", avgHitcount, "stdDev=", stdDev)
            self.assertAlmostEqual(sqrt(avgHitcount), stdDev, delta=0.1)
            for numHits in nditer(imgData):
                self.assertGreater(numHits, 0)

if __name__ == '__main__':
    unittest.main()
