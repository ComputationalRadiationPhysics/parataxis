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

import numpy as np
import bigfloat as bf
from bigfloat import BigFloat
bf.setcontext(bf.precision(128))

class SimulationData:
    """Holds data about the simulation
    
       size in pixels
       cellSize: Size per cell in meters
    """
    def __init__(self, size, cellSize):
        self.size = np.array(size)
        self.cellSize = np.array(cellSize).astype(BigFloat)
        
    def __str__(self):
        return "Size: " + str(self.size) + ", cellSize: " + str(self.cellSize)

class DetectorData:
    """Holds data about the detector:
    
       size in pixels
       cellSize: Size per cell in meters
       distance to end of simulation in meters
       anglePerCell: radians per cell (Simulation uses angle binning)
    """
    def __init__(self, size, cellSize, distance):
        self.size = np.array(size)
        self.cellSize = np.array(cellSize).astype(BigFloat)
        self.distance = BigFloat(distance)
        # Angle binning of detector
        self.anglePerCell = [bf.atan(cellSize / self.distance) for cellSize in self.cellSize]
    def resize(self, size):
        self.size = np.array(size)
    def __str__(self):
        return "Size: " + str(self.size) + ", cellSize: " + str(self.cellSize) + ", distance: " + str(self.distance)
        
class ParticleData:
    """Holds data about a particle:
    
       pos: Position index
       directionAngles: Angle in y and z direction the particle is going
       dir: Direction of the particle
    """
    def __init__(self, pos, directionAngles):
        self.pos = np.array(pos)
        self.directionAngles = np.array(directionAngles).astype(BigFloat)
        assert(len(directionAngles) == 2)
        self.dir = np.array([BigFloat(1), bf.atan(directionAngles[0]), bf.atan(directionAngles[1])])
    def __str__(self):
        return "Pos: " + str(self.pos) + ", dir: " + str(self.dir) + ", angles: " + str(self.directionAngles)

def calcPosAtSimEnd(particle, sim):
    """Calculate the position of a particle at the end of the simulation with its current dir"""
    # Distance to end in meters
    dx = (sim.size[0] - particle.pos[0]) * sim.cellSize[0]
    # Offset to move in meters
    moveOffset = dx * particle.dir / particle.dir[0]
    # Offset to move in cells
    moveOffset /= sim.cellSize
    return particle.pos + moveOffset.astype(int)

def getBinnedDetCellIdx(particle, detector, sim, convertToInt = True):
    """Get the cellIdx where the particle hits the detector using angle based binning"""
    particleEndPos = calcPosAtSimEnd(particle, sim)
    offsetsFromDetMiddle = (particleEndPos - sim.size / 2) * sim.cellSize
    # SimZ = DetX, SimY = DetY
    angles = np.flipud(particle.directionAngles)
    offsetsFromDetMiddle = np.flipud(offsetsFromDetMiddle[1:]) / detector.cellSize
    angleOffset = angles / detector.anglePerCell
    # Index for the offset
    idxOffset = angleOffset + offsetsFromDetMiddle
    # And shift so 0 is middle
    idx = idxOffset + detector.size / 2
    if convertToInt:
        return idx.astype(int)
    else:
        return idx

def getDetCellIdx(particle, detector, sim, convertToInt = True):
    """Get the cellIdx where the particle hits the detector using ray tracing"""
    offsetsFromDetMiddle = (particle.pos - sim.size / 2) * sim.cellSize
    # SimZ = DetX, SimY = DetY
    angles = np.array([particle.directionAngles[1], particle.directionAngles[0]])
    offsetsFromDetMiddle = np.array([offsetsFromDetMiddle[2], offsetsFromDetMiddle[1]]) / detector.cellSize
    # Distance in X to detector
    dx = (sim.size[0] - particle.pos[0]) * sim.cellSize[0] + detector.distance
    angleOffset = np.tan(angles) * dx / detector.cellSize
    # Index for the offset
    idxOffset = angleOffset + offsetsFromDetMiddle
    # And shift so 0 is middle
    idx = idxOffset + detector.size / 2
    if convertToInt:
        return idx.astype(int)
    else:
        return idx

def calcPhase(particle, detector, sim, waveLen):
    """Calculate the phase the particle has at the detector (current phase assumed to be 0)"""
    detCellIdx = getBinnedDetCellIdx(particle, detector, sim)
    #print("Target angles:", (detCellIdx - detector.size / 2) * detector.anglePerCell)
    dx = detector.distance + (sim.size[0] - particle.pos[0]) * sim.cellSize[0]
    dzdy = (detCellIdx - detector.size / 2) * detector.cellSize
    dydz = np.flipud(dzdy) + (sim.size[1:] / 2 - particle.pos[1:]) * sim.cellSize[1:]
    distance = bf.sqrt(dx**2 + dydz[0]**2 + dydz[1]**2)
    phase = 2 * bf.const_pi() / BigFloat(waveLen) * distance
    phase = bf.mod(phase, 2*bf.const_pi())
    return phase

def calcPhaseFarField(particle, sim, waveLen):
    """Calculate the phase the particle has at the detector in the far field"""
    # Only scatter in 1 direction
    assert(particle.directionAngles[0] == 0 or particle.directionAngles[1] == 0)
    # One of the sin is 0 -> Ok to use addition instead of a full projection
    pathDiff = particle.pos[1] * sim.cellSize[1] * bf.sin(particle.directionAngles[0]) + \
               particle.pos[2] * sim.cellSize[2] * bf.sin(particle.directionAngles[1])
    # Negate as increasing position decreases phase
    pathDiff = -pathDiff
    phase = 2 * bf.const_pi() / BigFloat(waveLen) * pathDiff
    phase = bf.mod(phase, 2*bf.const_pi())
    return phase

def calcRayLengthsAndFarFieldDiff(refRayAngles, cellOffsetToRefRay, sim, detector):
    """Calculates the distance of a ref ray with given angles (y,z) (towards smaller idxs) to the detector,
       the distance of a 2nd ray with a given offset in sim cells (y,z) going to the same point
       and the difference between those 2 using the far field approximation
       Return tuple (lRefLen, lRayLen, lApproxDiff)
    """
    assert(len(refRayAngles) == 2)
    assert(len(cellOffsetToRefRay) == 2)
    # Convert to meters
    rayOffset = cellOffsetToRefRay * sim.cellSize[1:]
    dydzRef = np.array([bf.tan(refRayAngles[0]), bf.tan(refRayAngles[1])]) * detector.distance
    lRef = bf.sqrt(detector.distance**2 + dydzRef[0]**2 + dydzRef[1]**2)
    dydzRay = dydzRef + rayOffset
    lRay = bf.sqrt(detector.distance**2 + dydzRay[0]**2 + dydzRay[1]**2)
    # Projection of rayOffset-Vector (with x=0) on lRay
    lDiff = np.dot(dydzRay, rayOffset)/lRay
    return (lRef, lRay, lDiff)

def checkFarFieldConstraint(sim, detector, waveLen):
    """Check if we can use the far field approximation"""
    # Half the range, as detector is in the middle of the volume
    maxAngles = detector.anglePerCell * detector.size / 2    
    # SimZ = DetX, SimY = DetY
    maxAngles = np.flipud(maxAngles)
    #print("Max Angles=", maxAngles)
    # Diff of rays going to same position calculated exactly vs "triangle" approximation
    # |lMin + lDiff - lMax| <<(!) waveLen
    # Ref ray is from middle of volume -> Max distance is half the volume
    (lMin, lMax, lDiff) = calcRayLengthsAndFarFieldDiff(maxAngles, sim.size[1:] / 2, sim, detector)
    maxError = bf.abs(lMin + lDiff - lMax)
    #print("lMin", lMin, "lMax", lMax, "lDiff", lDiff, "maxError=", maxError)
    if(maxError >= 0.05 * waveLen):
        #print("ERROR")
        return False
    return True

def checkExtendedFarFieldConstraint(sim, detector, waveLen):
    """Check the errors when the ray is not really going to the top of the detector cell"""
    # l2 is the actual ray going to the bottom of the detector cell (furthes away from top)
    # l1 is the ref ray going to the top of the cell
    # l2_ is the modified l2 to top of cell and l1_ the modified ref ray to the target pt of l2
    maxAnglesTop = detector.anglePerCell * detector.size / 2
    maxAnglesBottom = maxAnglesTop - detector.anglePerCell
    # SimZ = DetX, SimY = DetY
    maxAnglesTop = np.flipud(maxAnglesTop)
    maxAnglesBottom = np.flipud(maxAnglesBottom)
    # This is what we acutally want
    (l1, l2_, l1l2_Diff) = calcRayLengthsAndFarFieldDiff(maxAnglesTop, sim.size[1:] / 2, sim, detector)
    # This is what we easily get
    (l1_, l2, l1_l2Diff) = calcRayLengthsAndFarFieldDiff(maxAnglesBottom, sim.size[1:] / 2, sim, detector)
    # 1: |(l2_-l1) - (l2-l1_)| <<(!) waveLen
    cellError = bf.abs((l2_-l1) - (l2-l1_))
    print("cellError=", cellError, "OK" if cellError < 0.05 * waveLen else "ERROR")
    # 2: |(l2_-l1) - l1_l2Diff| <<(!) waveLen (Projection used)
    projError = bf.abs((l2_-l1) - l1_l2Diff)
    print("projError=", projError, "OK" if projError < 0.05 * waveLen else "ERROR")
    # 3: Try reverse projection: Project onto ref ray as its angles are known
    vecRefRay = np.array([bf.tan(maxAnglesTop[0]), bf.tan(maxAnglesTop[1])]) * detector.distance
    lRef = bf.sqrt(detector.distance**2 + vecRefRay[0]**2 + vecRefRay[1]**2)
    lReverseProjDiff = np.dot(vecRefRay, sim.size[1:] / 2 * sim.cellSize[1:])/lRef
    revProjError = bf.abs((l2_-l1) - lReverseProjDiff)
    print("revProjError=", revProjError, "OK" if revProjError < 0.05 * waveLen else "ERROR")

