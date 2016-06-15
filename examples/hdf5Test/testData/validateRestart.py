import os
from numpy import *
import numpy.testing as npt
import unittest
import h5py

class TestRestart(unittest.TestCase):
    def testCheckpoint(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        f1 = h5py.File(hdf5PathOrig)
        f2 = h5py.File(hdf5Path)
        data1 = list(f1["data"].values())[0]
        data2 = list(f2["data"].values())[0]
        fields1 = data1.get("fields") or data1["meshes"]
        fields2 = data2.get("fields") or data2["meshes"]
        # Number of fields
        self.assertEqual(len(fields1), len(fields2))
        for field1, field2 in zip(fields1.values(), fields2.values()):
            self.assertEqual(field1.shape, field2.shape)
            npt.assert_almost_equal(array(field1), array(field2))
            
        particles1 = data1["particles"]
        particles2 = data2["particles"]
        # Number of species
        self.assertEqual(len(particles1), len(particles2))
        for species1, species2 in zip(particles1.values(), particles2.values()):
            # Number of species attributes
            self.assertEqual(len(species1), len(species2))
            for prop1, prop2 in zip(species1.values(), species2.values()):
                self.assertEqual(len(prop1.attrs), len(prop2.attrs))
                for attr1, attr2 in zip(prop1.attrs.values(), prop2.attrs.values()):
                    self.assertEqual(iAttr[1], prop2.attrs[iAttr[0]])
 
    def testDetector(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/PhotonDetector_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        f1 = h5py.File(hdf5PathOrig)
        f2 = h5py.File(hdf5Path)
        data1 = list(f1["data"].values())[0]
        data2 = list(f2["data"].values())[0]
        fields1 = data1.get("fields") or data1["meshes"]
        fields2 = data2.get("fields") or data2["meshes"]
        # Number of fields
        self.assertEqual(len(fields1), len(fields2))
        for field1, field2 in zip(fields1.values(), fields2.values()):
            if isinstance(field1, h5py.Dataset):
                self.assertEqual(field1.shape, field2.shape)
                npt.assert_almost_equal(array(field1), array(field2))
            else:
                for ds1, ds2 in zip(field1.values(), field2.values()):
                    self.assertEqual(ds1.shape, ds2.shape)
                    npt.assert_almost_equal(array(ds2), array(ds2))
 
if __name__ == '__main__':
    unittest.main()
