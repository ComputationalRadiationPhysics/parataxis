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
 
if __name__ == '__main__':
    unittest.main()
