import os,sys
import numpy as np
sys.path.append('.')  # make sure python knows to look in the current folder
from PhaseSpaceAnalyser import ElectronPhaseSpace


DataLocation = os.path.abspath("../data/LinacPhaseSpace/atExit_Jul29_2020.dat")
DataLocation = "Z:/2RESEARCH/2_ProjectData/Phaser/PhaserSims/LinacPhaseSpace/atExit_Jul29_2020.dat"

PS = ElectronPhaseSpace(DataLocation)  # read our data into an instance of ElectronPhaseSpace

PS.GenerateTopasImportFile(Zoffset=-652.3)  # write our data to a topas phase space file, with Zoffset controlling the
# Z position. In the topas import, the Z position seems to behave relative to the local coordinate system of whatever
# component you attach the phase space import to

# If you want to check the z position of the output data call PrintData again:
PS.PrintData()

# You can also have a look at the distribution of the data with the plotting functions:
PS.PlotParticlePositions()