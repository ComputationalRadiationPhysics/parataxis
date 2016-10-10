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

class TestRestart(unittest.TestCase):
    def testCheckpoint(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_100.h5"
        
        with h5py.File(hdf5Path) as f:
            data = list(f["data"].values())[0]
            fields = data.get("fields") or data["meshes"]
            # Number of fields
            self.assertGreater(len(fields), 0)
            for field in fields.values():
                self.assertGreater(len(field), 0)
                
            particles = data["particles"]
            # Number of species
            self.assertGreater(len(particles), 0)
            for species in particles.values():
                momentum = species.get("momentum")
                self.assertIsNotNone(momentum)
                # 3D momentum
                self.assertEqual(len(momentum), 3)
                # All momentums must be ~1
                momentums = np.vstack((momentum["x"], momentum["y"], momentum["z"]))
                momNorms = np.linalg.norm(momentums, axis=0)
                npt.assert_array_almost_equal(momNorms, 1.)
 
if __name__ == '__main__':
    unittest.main()
