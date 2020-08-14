import os,sys
import numpy as np
sys.path.append('.')  # make sure python knows to look in the current folder
from PhaseSpaceAnalyser import ElectronPhaseSpace

## please see https://github.com/bwheelz36/ElectronPhaseSpace for allowable data imports:

# Random numpy data
# -----------------
# for the purpose of the example lets just use some random data. Note that here and throughout the code the assumption
# is that the beam is primarily directed in the Z direction
Nparticles = 10000
x = np.random.randn(Nparticles)  # normal distributed random data
y = np.random.randn(Nparticles)  # normal distributed random data
z = np.ones(Nparticles) * 100  # let's say z = 100 mm for arguments sake
px = x * .01 + np.random.randn(Nparticles) * .01 # transverse momentum with some noise (MeV)
py = y * .01 + np.random.randn(Nparticles) * .01 # + np.random.rand(Nparticles) * .001 # transverse momentum with some noise (MeV)
pz = np.ones(Nparticles) * 11 + np.random.randn(Nparticles) * .1 # primary beam direction
Data = np.vstack((x, y, z, px, py, pz))
Data = np.transpose(Data)

# Read in SLAC/CST/Topas data:
# ---------------------------
# DataLoc = /path/to/data
# PS = ElectronPhaseSpace(DataLoc)

# read our data into an instance of ElectronPhaseSpace
# ----------------------------------------------------
PS = ElectronPhaseSpace(Data)

PS.GenerateTopasImportFile(Zoffset=-100)  # write our data to a topas phase space file, with Zoffset controlling the
# Z position. In the topas import, the Z position seems to behave relative to the local coordinate system of whatever
# component you attach the phase space import to
# If you want to check the z position of the output data call PrintData again:
PS.PrintData()

# One can also call:
# PS.GenerateCSTParticleImportFile(Zoffset=0)  # generates CST *.pid file

# You can also have a look at the distribution of the data

# position distribution
# ---------------------
# PS.PlotParticlePositions()

# energy distribution
# -------------------
# PS.PlotEnergyHistogram()

# phase space distribution
# -------------------------
PS.PlotPhaseSpaceX()