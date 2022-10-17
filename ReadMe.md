# Documentation for Electron Phase Space

## Author
Brendan Whelan

## Purpose

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

A basic example of use is provided with 
ElectronPhaseSpaceExample.py

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
 
## Liability

This seems to work properly, but no promises! See license file for exact terms of use.
If you do find errors please fix and pull request or add an issue.

   
