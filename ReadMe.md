# Electron Phase Space

> :warning: this code has been replaced by the more general purpose and well writen [ParticlePhaseSpace](https://github.com/bwheelz36/ParticlePhaseSpace); please use that instead! 

The purpose of this code is to provide a tool to read electron phase spaces in python, perform some basic analysis, and - if needed, convert them to an output format appropriate to read into either CST or topas.

Data can be one of four formats:

1. topas phase space
2. Data from SLAC
3. CST trk particle monitor
3. Numpy array

For a numpy array, the data should be

[x y z px py pz]
x y z in in mm
px py pz  in [MeV/c]

(feel free to add more input read methods to read your own data if necessary)

## Use

Basic example below:

```python
import numpy as np
from ElectronPhaseSpace import ElectronPhaseSpace
# ^ this import statement assumes you have cloned this repo into the same directory as this script

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
weight = np.ones(Nparticles) # all particles weighted equally
Data = np.vstack((x, y, z, px, py, pz, weight))
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
```

## Requirements

You require the following python libraries to run this code::

	numpy
	scipy
	matplotlib
	topas2numpy
	
These can be installed by doing

```
pip install -r requirements.txt
```
   
