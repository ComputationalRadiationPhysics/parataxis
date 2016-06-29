import h5py
import numpy as np
import os.path
import datetime
from dateutil.tz import tzlocal
import argparse
import math

def writeOpenPMDHeader(f, basePath):
    """Write the root attributes (OpenPMD header)"""
    attr = f.attrs
    # Required attributes
    attr["openPMD"] = np.string_("1.0.0")
    attr["openPMDextension"] = 0
    attr["basePath"] = np.string_("/data/%T/")
    attr["meshesPath"] = np.string_("meshes/")
    attr["particlesPath"] = np.string_("particles/")
    attr["iterationEncoding"] = np.string_("fileBased")
    attr["iterationFormat"] = np.string_(os.path.basename(basePath) + "_%T.h5")

    # Recommended attributes
    attr["author"] = np.string_("Tool")
    attr["software"] = np.string_("Tool generated test data")
    attr["softwareVersion"] = np.string_("1.0.0")
    attr["date"] = np.string_( datetime.datetime.now(tzlocal()).strftime('%Y-%m-%d %H:%M:%S %z'))
    
def createBaseGroup(f, timestep, dt):
    """Create and return the base group at the given timestep and a given delta t (timestep length) in s"""
    baseGroup = f.create_group("/data/" + str(timestep))

    baseGroup.attrs["time"] = timestep * dt
    baseGroup.attrs["dt"] = dt
    baseGroup.attrs["timeUnitSI"] = 1.0
    return baseGroup

def createDensityFileset(basePath, shape, numTimesteps, dt, func):
    """Create a set of HDF5 files containing an electron_density field of the given 3D shape for the 
       period of numTimesteps of length dt with values of func(x, y, z, t)"""
    for timestep in range(numTimesteps):
        data = np.fromfunction(np.vectorize(lambda z,y,x: func(x, y, z, timestep * dt)), shape)
        with h5py.File(basePath + "_" + str(timestep) + ".h5", "w") as f:
            writeOpenPMDHeader(f, basePath)
            baseGroup = createBaseGroup(f, timestep, dt)
            meshes = baseGroup.create_group(f.attrs["meshesPath"])
            ds = meshes.create_dataset("electron_density", data = data)
            ds.attrs["dataOrder"] = np.string_("C")
            ds.attrs["axisLabels"] = np.array([np.string_("z"), np.string_("y"), np.string_("x")])
            
def createPhotonFileset(basePath, shape, numTimesteps, dt, func):
    """Create a set of HDF5 files containing the photon count in the given 2D shape for the 
       period of numTimesteps of length dt with values of func(x, y, t)"""
    for timestep in range(numTimesteps):
        data = np.fromfunction(np.vectorize(lambda y,x: func(x, y, timestep * dt)), shape)
        with h5py.File(basePath + "_" + str(timestep) + ".h5", "w") as f:
            writeOpenPMDHeader(f, basePath)
            baseGroup = createBaseGroup(f, timestep, dt)
            numPhotons = baseGroup.create_group(f.attrs["meshesPath"]).create_group("Nph")
            ds = numPhotons.create_dataset("x", data = data)
            ds.attrs["dataOrder"] = np.string_("C")
            ds.attrs["axisLabels"] = np.array([np.string_("y"), np.string_("x")])
            ds = numPhotons.create_dataset("y", data = np.zeros(shape))
            ds.attrs["dataOrder"] = np.string_("C")
            ds.attrs["axisLabels"] = np.array([np.string_("y"), np.string_("x")])

# Parse the command line.
parser = argparse.ArgumentParser(description="Create an HDF5 file set for the simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-f', '--folder', help='Output path')
parser.add_argument('-s', '--size', type=int, nargs=3, help='Size of the simulation (x y z)')
parser.add_argument('--dt', type=float, help='Timestep used for HDF5 records')
parser.add_argument('-t', '--time', type=float, help='Total time for HDF5 records')
options = parser.parse_args()

numTimesteps = int(math.ceil(options.time / options.dt))
size = np.flipud(np.array(options.size)) # z y x (fastest varying index last)
size2D = np.array([size[1], size[0]])
# Create time varying circle
createDensityFileset(os.path.join(options.folder, "field"), size, numTimesteps, options.dt,
                     lambda x,y,z,t: t * 0.65/options.dt if np.linalg.norm((y,z) - size2D/2) < 10 else 0.)
# Create time varying circle
createPhotonFileset(os.path.join(options.folder, "laserProfile"), size2D, numTimesteps, options.dt,
                     lambda x,y,t: t/options.dt * max(15 - np.linalg.norm((x,y) - size2D/2), 0))

