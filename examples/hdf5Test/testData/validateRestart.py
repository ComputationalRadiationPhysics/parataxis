import os
import numpy as np
import numpy.testing as npt
import unittest
import h5py

class TestRestart(unittest.TestCase):
    """Recursivly iterate over val*.value() till h5py.Datasets are found and validate those"""
    def checkRecords(self, valIs, valExpected, sortValues):
        if isinstance(valExpected, h5py.Dataset):
            self.assertEqual(valIs.shape, valExpected.shape, "Record: " + valExpected.name)
            if sortValues:
                valArrayIs = np.sort(valIs)
                valArrayExpected = np.sort(valExpected)
            else:
                valArrayIs = np.array(valIs)
                valArrayExpected = np.array(valExpected)
            npt.assert_allclose(valArrayIs, valArrayExpected, err_msg = "Record: " + valExpected.name)
        else:
            for ds1, ds2 in zip(valIs.values(), valExpected.values()):
                self.checkRecords(ds1, ds2, sortValues)

    def testRestart(self):
        self.longMessage = True
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/hdf5_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        with h5py.File(hdf5Path) as f1, h5py.File(hdf5PathOrig) as f2:
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
                self.assertEqual(len(species1), len(species2), "Species: " + species1.name)
                for prop1, prop2 in zip(species1.values(), species2.values()):
                    self.assertEqual(len(prop1.attrs), len(prop2.attrs), "Species attributes: " + species1.name)
                    for attr1, attr2 in zip(prop1.attrs.items(), prop2.attrs.items()):
                        npt.assert_equal(attr1[1], attr2[1], err_msg = "Attribute: " + prop1.name + "/" + attr1[0])
                    self.checkRecords(prop1, prop2, True)
 
    def testDetector(self):
        self.longMessage = True
        checkpointPath = os.environ["TEST_SIMOUTPUT_PATH"] + "/checkpoints"
        hdf5Path = checkpointPath + "/PhotonDetector_checkpoint_100.h5"
        hdf5PathOrig = hdf5Path.replace("out_Restart", "out_Checkpoint")
        
        with h5py.File(hdf5Path) as f1, h5py.File(hdf5PathOrig) as f2:
            data1 = list(f1["data"].values())[0]
            data2 = list(f2["data"].values())[0]
            fields1 = data1.get("fields") or data1["meshes"]
            fields2 = data2.get("fields") or data2["meshes"]
            # Number of fields
            self.assertEqual(len(fields1), len(fields2))
            self.checkRecords(fields1, fields2, False)
 
if __name__ == '__main__':
    unittest.main()

