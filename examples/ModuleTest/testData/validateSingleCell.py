import sys
import os
from numpy import *
import unittest
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "buildSystem"))
from ParamParser import ParamParser

class TestSingleCell(unittest.TestCase):
    def checkCoordinate(self, imgCoord, shouldPos):
        """Check if shouldPos (float value) is the same as imgCoord (int value) using floor and ceil rounding"""
        # Note: Image coordinates are y,x
        if(imgCoord[1] != floor(shouldPos[0])):
            self.assertEqual(imgCoord[1],ceil(shouldPos[0]))
        if(imgCoord[0] != floor(shouldPos[1])):
            self.assertEqual(imgCoord[0], ceil(shouldPos[1]))

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
        ScatterAngle = [params.GetNumber("particles::scatterer::direction::Fixed::angleY"), params.GetNumber("particles::scatterer::direction::Fixed::angleZ")]
        SimCellSize = [params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")]

        SimSize = list(map(int, os.environ["TEST_GRID_SIZE"].split(" ")))
        SimSize2D = array(SimSize[1:])

        PulseLen = floor(params.GetNumber("laserConfig::PULSE_LENGTH") / params.GetNumber("SI::DELTA_T"))
        NumPartsPerTsPerCell = params.GetNumber("laserConfig::distribution::Const::numParts")

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            DetSize = im.size

            ## Calculation

            ScatterOffsets = tan(ScatterAngle) * DetDist
            ScatterOffsets += (DensityPos2D - SimSize2D / 2) * SimCellSize[1:]
            # SimZ = DetX, SimY = DetY
            ScatterOffsets = flipud(ScatterOffsets)
            PosOnDet = ScatterOffsets / DetCellSize + array(DetSize) / 2
            if DetType.endswith("CountParticles"):
                ExpectedDetCellValue = NumPartsPerTsPerCell * PulseLen
            else:
                ExpectedDetCellValue = (NumPartsPerTsPerCell * PulseLen)**2

            ## Checks
            imgData = array(im)
            whitePts = transpose(where(imgData > 1.e-3))
            print(len(whitePts), "white pixels at:", whitePts)
            self.assertEqual(len(whitePts), 1)
            self.checkCoordinate(whitePts[0], PosOnDet)
            self.assertAlmostEqual(imgData[tuple(whitePts[0])], ExpectedDetCellValue, delta = 0.01)

if __name__ == '__main__':
    unittest.main()
