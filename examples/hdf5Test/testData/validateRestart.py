import os
import numpy as np
import numpy.testing as npt
import unittest
import h5py

class TestRestart(unittest.TestCase):
    """Recursivly iterate over val*.value() till h5py.Datasets are found and validate those"""
    def checkRecords(self, valIs, valExpected, sortValues):
        if isinstance(valExpected, h5py.Dataset):
            print("Checking record " + valExpected.name)
            self.assertEqual(valIs.shape, valExpected.shape)
            if sortValues:
                valIs = np.sort(valIs)
                valExpected = np.sort(valExpected)
            npt.assert_allclose(valIs, valExpected)
        else:
            for ds1, ds2 in zip(valIs.values(), valExpected.values()):
                self.checkRecords(ds1, ds2, sortValues)

    def testCheckpoint(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        f1 = h5py.File(hdf5Path)
        f2 = h5py.File(hdf5PathOrig)
        data1 = list(f1["data"].values())[0]
        data2 = list(f2["data"].values())[0]
        fields1 = data1.get("fields") or data1["meshes"]
        fields2 = data2.get("fields") or data2["meshes"]
        # Number of fields
        self.assertEqual(len(fields1), len(fields2))
        self.checkRecords(fields1, fields2, False)
            
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
                    npt.assert_equal(attr1, attr2)
                self.checkRecords(prop1, prop2, True)
 
    def atestDetector(self):
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/PhotonDetector_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        f1 = h5py.File(hdf5Path)
        f2 = h5py.File(hdf5PathOrig)
        data1 = list(f1["data"].values())[0]
        data2 = list(f2["data"].values())[0]
        fields1 = data1.get("fields") or data1["meshes"]
        fields2 = data2.get("fields") or data2["meshes"]
        # Number of fields
        self.assertEqual(len(fields1), len(fields2))
        self.checkRecords(fields1, fields2, False)
 
if __name__ == '__main__':
    unittest.main()

