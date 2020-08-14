# Documentation for Electron Phase Space

## Author
Brendan Whelan

## Purpose

The purpose of this code is to provide a tool to read electron phase spaces in python, perform some basic analysis, and - if needed, convert them to an output format appropriate to read into either CST or topas.

Input Data can at present be one of four kinds::

Data can be one of three formats:
1. CST trk particle monitor
2. Data from SLAC
3. Numpy array

For the third case, the data should be
[x y z px py pz]
x y z in in mm
px py pz  in [MeV/c]


(feel free to add more input read methods to read your own data if necessary)

## Use

A basic example of use is provided with 
ElectronPhaseSpaceExample.py

and below::
	
	import os,sys
	import numpy as np
	sys.path.append('.')  # make sure python knows to look in the current folder
	from PhaseSpaceAnalyser import ElectronPhaseSpace


	DataLocation = os.path.abspath("../data/LinacPhaseSpace/atExit_Jul29_2020.dat")

	PS = ElectronPhaseSpace(DataLocation)  # read our data into an instance of ElectronPhaseSpace

	PS.GenerateTopasImportFile(Zoffset=-652.3)  # write our data to a topas phase space file, with Zoffset controlling the
	# Z position. In the topas import, the Z position seems to behave relative to the local coordinate system of whatever
	# component you attach the phase space import to

	# If you want to check the z position of the output data call PrintData again:
	PS.PrintData()

	# You can also have a look at the distribution of the data with the plotting functions:
	PS.PlotParticlePositions()
	
## Requirements

You require the following python libraries to run this code::

	numpy
	scipy
	matplotlib
	topas2numpy (only relevant if you plan to read in topas data)
 


   
