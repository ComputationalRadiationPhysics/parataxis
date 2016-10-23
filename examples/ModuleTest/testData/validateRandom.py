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

        params.SetCurNamespace("parataxis::detector::PhotonDetector")
        DetDist = params.GetNumber("distance")
        DetCellSize = [params.GetNumber("cellWidth"), params.GetNumber("cellHeight")]
        DetType = params.GetValue("IncomingParticleHandler")

        params.SetCurNamespace("parataxis::initialDensity")
        assert(params.GetValue("Generator") == "AvailableGenerators::Cuboid"), "Must use cuboid generator"
        DensityPos = params.GetVector("AvailableGenerators::Cuboid::Offset")
        DensityPos2D = array(DensityPos[1:])

        params.SetCurNamespace("parataxis")
        ScatterAngle = params.GetNumber("particles::scatterer::direction::RandomDirection::maxPolar")
        SimCellSize = array([params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")])

        SimSize = list(map(int, os.environ["TEST_GRID_SIZE"].split(" ")))
        SimSize2D = array(SimSize[1:])

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            DetSize = im.size

            ## Calculation

            ScatterRadius = tan(ScatterAngle) * DetDist
            ScatterRadii = array([ScatterRadius, ScatterRadius])
            ScatterMiddlePt = (DensityPos2D - SimSize2D / 2) * SimCellSize[1:]
            # SimZ = DetX, SimY = DetY
            ScatterMiddlePt = flipud(ScatterMiddlePt)
            MiddlePtOnDet = (ScatterMiddlePt / DetCellSize + array(DetSize) / 2).astype(int)
            ScatterRadiiOnDet = (ScatterRadii / DetCellSize).astype(int)
            # Create mask centered around MiddlePt in image coordinates (y,x)
            y,x = ogrid[0:DetSize[0], 0:DetSize[1]]
            y -= MiddlePtOnDet[0]
            x -= MiddlePtOnDet[1]
            # Ignore outermost region (half cell hits which might change the count)
            ScatterRadiiOnDet -= [1,1]
            mask = x**2/ScatterRadiiOnDet[0]**2 + y**2/ScatterRadiiOnDet[1]**2 <= 1
            imgData = array(im)[where(mask)]
            for numHits in imgData:
                self.assertGreater(numHits, 0)
           
            avgHitcount = average(imgData)
            minHitcount = amin(imgData)
            maxHitcount = amax(imgData)
            stdDev = std(imgData)
            relDev = stdDev/avgHitcount * 100
            print("Hitcount:", minHitcount, "-", maxHitcount, "average =", avgHitcount, "stdDev=", stdDev, "rel. Dev=" + str(relDev) + "%")
            self.assertAlmostEqual(sqrt(avgHitcount), stdDev, delta=0.09)
            self.assertLess(relDev, 6.6) # 6.6%
 
if __name__ == '__main__':
    unittest.main()
