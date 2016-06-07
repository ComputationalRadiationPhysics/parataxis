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
            self.assertAlmostEqual(sqrt(avgHitcount), stdDev, delta=0.05)
            self.assertLess(relDev, 0.066) # 6.6%
 
if __name__ == '__main__':
    unittest.main()
