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
    attr["openPMDextension"] = np.uint32(0)
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

def createDensityFileset(basePath, shape, cellSizes, numTimesteps, dt, func):
    """Create a set of HDF5 files containing an electron_density field of the given 3D shape for the 
       period of numTimesteps of length dt with values of func(x, y, z, t)"""
    for timestep in range(numTimesteps):
        data = np.fromfunction(np.vectorize(lambda z,y,x: func(x, y, z, timestep * dt)), shape)
        with h5py.File(basePath + "_" + str(timestep) + ".h5", "w") as f:
            writeOpenPMDHeader(f, basePath)
            baseGroup = createBaseGroup(f, timestep, dt)
            meshes = baseGroup.create_group(f.attrs["meshesPath"])
            
            ds = meshes.create_dataset("electron_density", data = data)
            ds.attrs["axisLabels"] = np.array([np.string_("z"), np.string_("y"), np.string_("x")])
            ds.attrs["dataOrder"] = np.string_("C")
            ds.attrs["geometry"] = np.string_("cartesian")
            ds.attrs["unitDimension"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ds.attrs["gridGlobalOffset"] = np.array([0.0, 0.0, 0.0])
            gridUnitSI = 1.
            ds.attrs["gridSpacing"] = cellSizes / gridUnitSI
            ds.attrs["gridUnitSI"] = gridUnitSI
            ds.attrs["timeOffset"] = 0.0
            ds.attrs["position"] = np.array([0.0, 0.0, 0.0])
            ds.attrs["unitSI"] = 1.0
            
            
def createPhotonFileset(basePath, shape, cellSizes, numTimesteps, dt, func):
    """Create a set of HDF5 files containing the photon count in the given 2D shape for the 
       period of numTimesteps of length dt with values of func(x, y, t)"""
    for timestep in range(numTimesteps):
        data = np.fromfunction(np.vectorize(lambda y,x: func(x, y, timestep * dt)), shape)
        with h5py.File(basePath + "_" + str(timestep) + ".h5", "w") as f:
            writeOpenPMDHeader(f, basePath)
            baseGroup = createBaseGroup(f, timestep, dt)
            
            numPhotons = baseGroup.create_group(f.attrs["meshesPath"]).create_group("Nph")
            numPhotons.attrs["axisLabels"] = np.array([np.string_("y"), np.string_("x")])
            numPhotons.attrs["dataOrder"] = np.string_("C")
            numPhotons.attrs["geometry"] = np.string_("cartesian")
            numPhotons.attrs["gridGlobalOffset"] = np.array([0.0, 0.0])
            # Choose an arbitrary unit as additional test
            gridUnitSI = 2.
            numPhotons.attrs["gridSpacing"] = cellSizes / gridUnitSI
            numPhotons.attrs["gridUnitSI"] = gridUnitSI
            numPhotons.attrs["timeOffset"] = 0.0
            numPhotons.attrs["unitDimension"] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            
            # x-polarized
            ds = numPhotons.create_dataset("x", data = data)
            ds.attrs["position"] = np.array([0.0, 0.0])
            ds.attrs["unitSI"] = 1.0
            
            # y-polarized (nothing yet)
            ds = numPhotons.create_dataset("y", data = np.zeros(shape))
            ds.attrs["position"] = np.array([0.0, 0.0])
            ds.attrs["unitSI"] = 1.0

# Parse the command line.
parser = argparse.ArgumentParser(description="Create an HDF5 file set for the simulation", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--createDensity', action='store_true', help='Create electron density')
parser.add_argument('--createLaser', action='store_true', help='Create XRay photon distribution')
parser.add_argument('-f', '--folder', help='Output path')
parser.add_argument('-s', '--size', type=int, nargs=3, help='Size of the simulation (x y z)')
parser.add_argument('-c', '--cellSize', type=float, nargs=3, help='Size of the simulation cells (x y z)')
parser.add_argument('--dt', type=float, help='Timestep used for HDF5 records')
parser.add_argument('-t', '--time', type=float, help='Total time for HDF5 records')
options = parser.parse_args()

numTimesteps = int(math.ceil(options.time / options.dt))
size = np.flipud(np.array(options.size)) # z y x (fastest varying index last)
size2D = np.array([size[1], size[0]])
cellSize = np.flipud(np.array(options.cellSize)) # z y x (fastest varying index last)
cellSize2D = np.array([cellSize[1], cellSize[0]])

def calcDensityField(x, y, z, t, dt):
    # Don't scatter in first few cells to allow laser test
    if x < 5:
        return 0.
    if np.linalg.norm((y,z) - size2D/2) < 10:
        return t * 0.65/options.dt
    else:
        return 0.

def getPhotonCount(idxX, idxY, timestep):
    """Return the number of photons expected in the given cell and timestep"""
    result = (idxX - 4) + (idxY - 2) * 2 + (timestep - 4)
    return max(0, result)

if options.createDensity:
    # Create time varying circle
    createDensityFileset(os.path.join(options.folder, "field"), size, cellSize, numTimesteps, options.dt,
                         lambda x,y,z,t: calcDensityField(x, y, z, t, options.dt) )

if options.createLaser:
    # We assume 4 times bigger cells and 8 times bigger timestep than in simulation
    # the function is defined in terms of simulation units, so multiply our cells and timestep and then scale with area and time ratio
    photonFunc = lambda x,y,t: getPhotonCount(x*4, y*4, int(t/options.dt)*8) * 4*4*8
    createPhotonFileset(os.path.join(options.folder, "laserProfile"), size2D, cellSize2D, numTimesteps, options.dt, photonFunc)

