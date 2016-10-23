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
import numpy as np
import unittest
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "buildSystem"))
from ParamParser import ParamParser
import scatterHelpers as scatter

class TestSingleCell(unittest.TestCase):
    def checkCoordinate(self, imgCoord, shouldPos):
        """Check if shouldPos (float value) is the same as imgCoord (int value) using floor and ceil rounding"""
        # Note: Image coordinates are y,x
        if(imgCoord[1] != np.floor(shouldPos[0])):
            self.assertEqual(imgCoord[1], np.ceil(shouldPos[0]))
        if(imgCoord[0] != np.floor(shouldPos[1])):
            self.assertEqual(imgCoord[0], np.ceil(shouldPos[1]))

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

        params.SetCurNamespace("parataxis::detector::PhotonDetector")
        DetDist = params.GetNumber("distance")
        DetCellSize = [params.GetNumber("cellWidth"), params.GetNumber("cellHeight")]
        DetType = params.GetValue("IncomingParticleHandler")
 
        params.SetCurNamespace("parataxis::initialDensity")
        self.assertEqual(params.GetValue("Generator"), "AvailableGenerators::Cuboid")
        DensityPos = params.GetVector("AvailableGenerators::Cuboid::Offset")
        DensityPos2D = np.array(DensityPos[1:])

        params.SetCurNamespace("parataxis")
        ScatterAngle = [params.GetNumber("particles::scatterer::direction::Fixed::angleY"), params.GetNumber("particles::scatterer::direction::Fixed::angleZ")]
        scatterParticle = scatter.ParticleData(DensityPos, ScatterAngle)

        simulation = scatter.SimulationData(list(map(int, os.environ["TEST_GRID_SIZE"].split(" "))),
                                            [params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")])
        SimSize2D = np.array(simulation.size[1:])
        
 
        PulseLen = np.floor(params.GetNumber("laserConfig::PULSE_LENGTH") / params.GetNumber("SI::DELTA_T"))
        NumPartsPerTsPerCell = params.GetNumber("laserConfig::distribution::Const::numParts")

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            DetSize = im.size
            detector = scatter.DetectorData(DetSize, DetCellSize, DetDist)

            ## Calculation

            ScatterOffsets = np.tan(ScatterAngle) * (DetDist + (simulation.size[0] - DensityPos[0]) * simulation.cellSize[0])
            ScatterOffsets += (DensityPos2D - SimSize2D / 2) * simulation.cellSize[1:].astype(float)
            # SimZ = DetX, SimY = DetY
            ScatterOffsets = np.flipud(ScatterOffsets)
            PosOnDet = ScatterOffsets / DetCellSize + np.array(DetSize) / 2
            posOnDetNew = scatter.getDetCellIdx(scatterParticle, detector, simulation, False).astype(float)
            self.assertAlmostEqual(PosOnDet[0], posOnDetNew[0], places = 6)
            self.assertAlmostEqual(PosOnDet[1], posOnDetNew[1], places = 6)
            
            if DetType.endswith("CountParticles"):
                ExpectedDetCellValue = NumPartsPerTsPerCell * PulseLen
            else:
                ExpectedDetCellValue = (NumPartsPerTsPerCell * PulseLen)**2

            ## Checks
            imgData = np.array(im)
            whitePts = np.transpose(np.where(imgData > 1.e-3))
            print(len(whitePts), "white pixels at:", whitePts)
            self.assertEqual(len(whitePts), 1)
            self.checkCoordinate(whitePts[0], PosOnDet)
            self.assertAlmostEqual(imgData[tuple(whitePts[0])], ExpectedDetCellValue, delta = 0.01)

if __name__ == '__main__':
    unittest.main()
