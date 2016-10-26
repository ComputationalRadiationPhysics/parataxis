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
import bigfloat as bf

class TestInterference(unittest.TestCase):
    def checkCoordinate(self, imgCoord, shouldPos):
        """Check if shouldPos (float value) is the same as imgCoord (int value) using floor and ceil rounding"""
        # Note: Image coordinates are y,x
        if(imgCoord[1] != np.floor(shouldPos[0])):
            self.assertEqual(imgCoord[1], np.ceil(shouldPos[0]))
        if(imgCoord[0] != np.floor(shouldPos[1])):
            self.assertEqual(imgCoord[0], np.ceil(shouldPos[1]))

    def testInterference(self):
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
        detector = scatter.DetectorData([], [params.GetNumber("cellWidth"), params.GetNumber("cellHeight")], params.GetNumber("distance"))
        self.assertEqual(params.GetValue("IncomingParticleHandler"), "particleHandlers::AddWaveParticles")

        params.SetCurNamespace("parataxis::particles::scatterer::direction::Fixed")
        scatterAngles = [params.GetNumber("angleY"), params.GetNumber("angleZ")]

        params.SetCurNamespace("parataxis::initialDensity")
        self.assertEqual(params.GetValue("Generator"), "AvailableGenerators::DoublePoint")
        params.SetCurNamespace("parataxis::initialDensity::AvailableGenerators::DoublePoint")
        scatterParticle1 = scatter.ParticleData([params.GetNumber("offsetX"), params.GetNumber("offsetY"), params.GetNumber("offsetZ1")], scatterAngles)
        scatterParticle2 = scatter.ParticleData([params.GetNumber("offsetX"), params.GetNumber("offsetY"), params.GetNumber("offsetZ2")], scatterAngles)

        params.SetCurNamespace("parataxis")
        simulation = scatter.SimulationData(list(map(int, os.environ["TEST_GRID_SIZE"].split(" "))),
                                            [params.GetNumber("SI::CELL_WIDTH"), params.GetNumber("SI::CELL_HEIGHT"), params.GetNumber("SI::CELL_DEPTH")])

        pulseLen = np.floor(params.GetNumber("laserConfig::PULSE_LENGTH") / params.GetNumber("SI::DELTA_T"))
        self.assertEqual(params.GetValue("laserConfig::distribution::UsedValue"), "EqualToPhotons")
        numPartsPerTsPerCell = params.GetNumber("laserConfig::photonCount::Const::numPhotons")
        waveLen = params.GetNumber("wavelengthPhotons", getFromValueIdentifier = True)
        #print("Wavelen=", waveLen)

        with open(os.environ["TEST_BASE_BUILD_PATH"] + "/" + os.environ["TEST_NAME"] + "_detector.tif", 'rb') as imFile:
            im = Image.open(imFile)
            detector.resize(im.size)
            
            self.assertTrue(scatter.checkFarFieldConstraint(simulation, detector, waveLen))
            
            ## Calculation
            posOnDet1 = scatter.getBinnedDetCellIdx(scatterParticle1, detector, simulation)
            posOnDet2 = scatter.getBinnedDetCellIdx(scatterParticle2, detector, simulation)
            np.testing.assert_allclose(posOnDet1, posOnDet2)

            ## Checks
            imgData = np.array(im)
            whitePts = np.transpose(np.where(imgData > 1.e-3))
            print(len(whitePts), "white pixels at:", whitePts)
            self.assertEqual(len(whitePts), 1)
            self.checkCoordinate(whitePts[0], posOnDet1)
            
            phi1 = scatter.calcPhase(scatterParticle1, detector, simulation, waveLen)
            phi2 = scatter.calcPhase(scatterParticle2, detector, simulation, waveLen)
            real = bf.cos(phi1) + bf.cos(phi2)
            imag = bf.sin(phi1) + bf.sin(phi2)
            sqrtIntensity = bf.sqrt(real*real + imag*imag)
            sqrtIntensityPerTs = sqrtIntensity * numPartsPerTsPerCell
            #print("Phis:", phi1, phi2, "Diff", phi1-phi2, "SqrtIntensityPerTs", sqrtIntensityPerTs)
            expectedDetCellValue = float(bf.sqr(sqrtIntensityPerTs * pulseLen))
            np.testing.assert_allclose(imgData[tuple(whitePts[0])], expectedDetCellValue, rtol = 1e-3)

if __name__ == '__main__':
    unittest.main()
