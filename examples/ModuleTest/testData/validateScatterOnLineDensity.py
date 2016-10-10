# Copyright 2015-2016 Alexander Grund
#
# This file is part of ParaTAXIS.
#
# ParaTAXIS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ParaTAXIS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with ParaTAXIS.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
from numpy import *
import unittest
from collections import Counter
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "buildSystem"))
from ParamParser import ParamParser

def getDetCell(DetDist, DetCellSizes, DetSize, scatterOffsets):
    # SimZ = DetX, SimY = DetY
    offsetsFromDetMiddle = flipud(scatterOffsets)
    # Angle binning of detector
    anglePerCell = [arctan(cellSize / DetDist) for cellSize in DetCellSizes]
    # Angle in which we hit the detector
    angleOffset = [arctan(offset / DetDist) for offset in offsetsFromDetMiddle]
    # Index for the offset
    idxOffset = array(angleOffset) / anglePerCell
    # And shift so 0 is middle
    return (idxOffset + array(DetSize) / 2).astype(float)

class TestLineDensity(unittest.TestCase):
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
        self.assertEqual(params.GetValue("Generator"), "AvailableGenerators::RaisingLine")
        DensityPos = params.GetNumber("AvailableGenerators::RaisingLine::offsetOther")

        params.SetCurNamespace("xrt")
        ScatterAngleFactors = [params.GetNumber("particles::scatterer::direction::LinearDensity::factorY"), params.GetNumber("particles::scatterer::direction::LinearDensity::factorZ")]
        SimCellSize = [params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")]

        SimSize = list(map(int, os.environ["TEST_GRID_SIZE"].split(" ")))
        SimSize2D = array(SimSize[1:])

        PulseLen = floor(params.GetNumber("laserConfig::PULSE_LENGTH") / params.GetNumber("SI::DELTA_T"))
        NumPartsPerTsPerCell = params.GetNumber("laserConfig::distribution::Const::numParts")

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            DetSize = array(im.size)

            ## Calculation
            # Only scattering along y axis. 1 Angle per cell (pt on line)
            ScatterAngles = array([array(ScatterAngleFactors) * (y + 1) for y in range(SimSize[1])])
            ScatterOffsets = tan(ScatterAngles) * DetDist
            ScatterOffsets += array([([y, DensityPos] - SimSize2D / 2) * SimCellSize[1:] for y in range(SimSize[1])])
            # SimZ = DetX, SimY = DetY
            PosOnDet = [getDetCell(DetDist, DetCellSize, DetSize, offset) for offset in ScatterOffsets]
            
            NumBeamCells = flipud(SimCellSize[1:] * SimSize2D) / DetCellSize
            BeamStart = ((DetSize - NumBeamCells) / 2).astype(int)
            BeamEnd   = ((DetSize + NumBeamCells) / 2).astype(int)
            PosOnDet = [pos for pos in PosOnDet if (floor(pos) < BeamStart).any() or (floor(pos) > BeamEnd).any()]
            
            HitCounts = Counter([(int(el[0]), int(el[1])) for el in PosOnDet])
            ExpectedDetCellValue = (NumPartsPerTsPerCell * PulseLen)**2

            ## Checks
            imgData = array(im)
            whitePts = transpose(where(imgData > 1.e-3))
            print(len(whitePts), "white pixels found")
            self.assertGreater(len(whitePts), 1)
            self.assertEqual(len(whitePts), len(HitCounts))
            for pt in whitePts:
                # Points in image are (y,x) but hits are (x,y)
                self.assertIn((pt[1], pt[0]), HitCounts)

if __name__ == '__main__':
    unittest.main()
