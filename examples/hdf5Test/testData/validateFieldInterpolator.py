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

import os
import numpy as np
import numpy.testing as npt
import unittest
import h5py
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "buildSystem"))
from ParamParser import ParamParser

def initParamParser():
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
    return params


class TestFieldInterpolator(unittest.TestCase):
    def getHDF5Timestep(self, inputFieldsPath, time):
        """Get biggest HDF5 timestep for which hdf5Time <= time"""
        with h5py.File(inputFieldsPath + "_0.h5", 'r') as f:
             data = list(f["data"].values())[0]
             h5Time = data.attrs["time"] * data.attrs["timeUnitSI"]
             h5DT = data.attrs["dt"] * data.attrs["timeUnitSI"]
             return int((time - h5Time) / h5DT)

    def testFieldInterpolator(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_50.h5"
        inputPath = os.environ["TEST_INSTALL_PATH"] + "/testData/" + os.environ["TEST_NAME"]
        inputFieldsPath = inputPath + "/field"
        
        with h5py.File(hdf5Path) as f:
            data = list(f["data"].values())[0]
            curTime = data.attrs["time"] * data.attrs["timeUnitSI"]
            # TODO: Currently the field in the checkpoint is from the past timestep. Once this is fixed remove the following line
            curTime -= data.attrs["dt"] * data.attrs["timeUnitSI"]
            hdf5Timestep = self.getHDF5Timestep(inputFieldsPath, curTime)
            with h5py.File(inputFieldsPath + "_" + str(hdf5Timestep) + ".h5", 'r') as fieldFileLast, h5py.File(inputFieldsPath + "_" + str(hdf5Timestep + 1) + ".h5", 'r') as fieldFileNext:
                fieldDataLast = list(fieldFileLast["data"].values())[0]
                fieldDataNext = list(fieldFileNext["data"].values())[0]
                lastTime = fieldDataLast.attrs["time"] * fieldDataLast.attrs["timeUnitSI"]
                nextTime = fieldDataNext.attrs["time"] * fieldDataNext.attrs["timeUnitSI"]
                self.assertLessEqual(lastTime, curTime)
                self.assertGreater(nextTime, curTime)
                lastField = np.array((fieldDataLast.get("fields") or fieldDataLast["meshes"])["electron_density"])
                nextField = np.array((fieldDataNext.get("fields") or fieldDataNext["meshes"])["electron_density"])
                expectedCurField = lastField + (nextField - lastField) * (curTime - lastTime) / (nextTime - lastTime)
                curField = (data.get("fields") or data["meshes"])["electron_density"]
                npt.assert_array_almost_equal(curField, expectedCurField)
 
def getPhotonCount(idxX, idxY, timestep):
    """Return the number of photons expected in the given cell and timestep"""
    result = (idxX - 4) + (idxY - 2) * 2 + (timestep - 4)
    return max(0, result)

class TestPhotonInterpolator(unittest.TestCase):
    def testPhotonInterpolator(self):
        params = initParamParser()
        params.SetCurNamespace("parataxis::SI")
        dt = params.GetNumber("DELTA_T")
        cellSize = np.array([params.GetNumber("CELL_WIDTH"), params.GetNumber("CELL_HEIGHT"), params.GetNumber("CELL_DEPTH")])
        simSize = os.environ["TEST_GRID_SIZE"].split(" ")
        
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        for timestep in [50, 100]:
            hdf5Path = checkpointPath + "/hdf5_checkpoint_" + str(timestep) + ".h5"
            
            with h5py.File(hdf5Path) as f:
                data = list(f["data"].values())[0]
                curTime = data.attrs["time"] * data.attrs["timeUnitSI"]
                # Time is from AFTER the timestep
                npt.assert_almost_equal(curTime, dt * (timestep + 1))
                
                hdf5Photons = data["particles/p"]
    
                photonWeighting = hdf5Photons["weighting"]
                constWeighting = photonWeighting.attrs["value"]
                
                # Probably cell index
                photonPosOffset = hdf5Photons["positionOffset"]
                offsetUnit = np.array([
                                photonPosOffset["x"].attrs["unitSI"],
                                photonPosOffset["y"].attrs["unitSI"],
                                photonPosOffset["z"].attrs["unitSI"]
                            ])
                # Make it an array of positions (each row has x,yz)
                photonPosOffset = np.transpose([
                                    np.array(photonPosOffset["x"]),
                                    np.array(photonPosOffset["y"]),
                                    np.array(photonPosOffset["z"])
                                  ])
                
                # Probably incell position
                photonPos = hdf5Photons["position"]
                posUnit = np.array([
                                photonPos["x"].attrs["unitSI"],
                                photonPos["y"].attrs["unitSI"],
                                photonPos["z"].attrs["unitSI"]
                          ])
                # Make it an array of positions (each row has x,yz)
                photonPos = np.transpose([np.array(photonPos["x"]), np.array(photonPos["y"]), np.array(photonPos["z"])])
                
                # Combine to full positions in cells
                photonPos = photonPosOffset * (offsetUnit / cellSize) + photonPos * (posUnit / cellSize)

                # Use only photons just spawned and moved once
                photonMask = photonPos[:, 0] <= 1
                photonPosInFirstSlice = photonPos[photonMask][:,1:3].astype(int)
                if constWeighting:
                    weightings = np.full(len(photonPosInFirstSlice), constWeighting)
                else:
                    weightings = np.array(photonWeighting)[photonMask]

                # Accumulate number of photons/cell
                numPhotonsPerCell, edgesX, edgesY = np.histogram2d(photonPosInFirstSlice[:,0], photonPosInFirstSlice[:,1],
                                                                   weights=weightings, bins=simSize[1:3])
                numPhotonsPerCell = np.round(numPhotonsPerCell).astype(int)
                for idxX in range(simSize[1]):
                    for idxY in range(simSize[2]):
                        self.assertEqual(numPhotonsPerCell[idxX, idxY], getPhotonCount(idxY, idxX, timestep))
 
if __name__ == '__main__':
    unittest.main()
