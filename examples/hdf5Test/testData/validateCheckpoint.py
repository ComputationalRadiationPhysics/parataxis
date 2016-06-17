import os
import numpy as np
import numpy.testing as npt
import unittest
import h5py

class TestRestart(unittest.TestCase):
    def testCheckpoint(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_100.h5"
        
        f = h5py.File(hdf5Path)
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
