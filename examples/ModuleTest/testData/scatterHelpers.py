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
    angles = np.array([particle.directionAngles[1], particle.directionAngles[0]])
    offsetsFromDetMiddle = np.array([offsetsFromDetMiddle[2], offsetsFromDetMiddle[1]]) / detector.cellSize
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
    dx = detector.distance + (sim.size[0] - particle.pos[0]) * sim.cellSize[0]
    dzdy = (detCellIdx - detector.size / 2) * detector.cellSize
    dydz = np.flipud(dzdy) + (particle.pos[1:] - sim.size[1:] / 2) * sim.cellSize[1:]
    distance = bf.sqrt(dx**2 + dydz[0]**2 + dydz[1]**2)
    phase = 2 * bf.const_pi() / BigFloat(waveLen) * distance
    phase = bf.mod(phase, 2*bf.const_pi())
    return phase
