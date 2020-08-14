# PhaseSpaceAnalyser.py
import numpy as np
from matplotlib import pyplot as plt
from scipy import constants
import math
import os, sys
try:
    import topas2numpy as tp
except ImportError:
    print('unable to import topas2numpy, which is necessary to read in topas data files')


class ElectronPhaseSpace:
    """
    A set of functions for analysing and converting phase spaces.
    At the moment there is an implict assumption that electron phase spaces are being used, which impacts on the conversion
    of momentum to kinetic energy. This could be updated with minimal effort if needed.
    """

    def __init__(self, Data):
        """
        :param Data: can be one of four things:
        1. CST trk particle monitor
        2. Data from SLAC
        3. Topas ASCII phase space (read in with topas2numpy)
        4. Numpy array

        For the third case, the data should be
        [x y z px py pz]
        with x y z in in mm
        and px py pz  in [MeV/c]
        """
        self.Data = Data
        self.MakePlots = False  # there are some phase space plots included but they are not very robust
        self.me_MeV = 0.511  # electron mass in MeV

        # we will trey to figure out what data is being put in.
        # this function will probably require some updates to work with different data formats
        self.__DetectDataType()
        self.__ReadInData()
        # calculations:
        self.__CalculateBetaAndGamma()
        self.__CalculateTwissParameters()  # at the moment just caclculates transverse phse space in X.
        self.__AnalyseEnergyDistribution()
        self.PrintData()
        if self.MakePlots:
            self.PlotPhaseSpaceX()
            self.PlotEnergyHistogram()
            self.PlotAngularSpread()
            self.PlotRadialDependency()

    def PrintData(self):
        """
        can be used to print info to the termainl
        """
        print(f'\u03C0\u03B5: {self.epsilon: 1.1f} mm mrad, \u03B1: {self.alpha: 1.1f}, \u0392: {self.beta: 1.1f}, '
              f'\u03B3: {self.gamma: 1.1f}')
        print(f'Mean energy: {self.meanEnergy: 1.1f} MeV \u00B1 {self.EnergySpread} (std) ')
        print(f'Mean Z position of input data is {np.mean(self.z): 3.1f} \u00B1 {np.std(self.z): 1.1f} (std)')
        if hasattr(self, 'zOut'):
            print(f'Mean Z position of output data is {np.mean(self.zOut): 3.1f} \u00B1 {np.std(self.zOut): 1.1f} (std)')

    def __DetectDataType(self):
        """
        This function is used to detect what type of data the user has put in. see __init__ for allowable data types
        """
        if isinstance(self.Data,np.ndarray):
            self.DataType = 'numpy'

        elif os.path.isfile(self.Data):
            # check for file extension:

            Filepath, Filename = os.path.split(self.Data)
            Filename, Filetype = os.path.splitext(Filename)
            if Filetype == '.phsp':
                self.DataType = 'topas'
            else:
                with open(self.Data) as f:
                    FirstLine = f.readline()
                CSTfirstLinePattern = '%  ASCII export :'
                SLACfirstLinePattern = 'Phase space'

                if FirstLine.find(SLACfirstLinePattern) >= 0:
                    self.DataType = "SLAC"
                elif FirstLine.find(CSTfirstLinePattern) >= 0:
                    self.DataType = "CST"
                else:
                    sys.exit('unable to determine data type')
        else:
            sys.exit('unable to determine data type')

    def __ReadInData(self):
        """
        Controller function for data read in
        """
        if self.DataType == 'SLAC':
            self.__ReadInSLACData()
            self.__CheckEnergyCalculation()
            self.OutputDataLoc, Filename = os.path.split(self.Data)
            self.OutputFile, Filetype = os.path.splitext(Filename)
            self.TotalCurrent = .3  # total current in A. This is hard to figure out directly from the file.
        elif self.DataType == "CST":
            self.__ReadInCSTdata()
            self.OutputDataLoc, Filename = os.path.split(self.Data)
            self.OutputFile, Filetype = os.path.splitext(Filename)
        elif self.DataType == 'numpy':
            self.__ReadInNumpyData()
            # in this case we just dump the out put in the working directory. Should implement a more elegant version...
            self.OutputDataLoc = os.getcwd()
            self.OutputFile = 'PSoutput'  # should change this to a user specification.
        elif self.DataType == 'topas':
            self.OutputDataLoc, Filename = os.path.split(self.Data)
            self.OutputFile, Filetype = os.path.splitext(Filename)
            self.__ReadInTopasASCIIdata()

    def __ReadInSLACData(self):
        """
        Read in data from data file supplied by SLAC.

        The following info supplied by SLAC:
        The file atExit.dat contains the distribution of the charge particles in a bunch that has just exited the last cavity.
        This data is in the following format, where only first four data rows are shown. The last row corresponds to the
        kinetic energy calculated from the momenta. The data in this file has been ordered in the descending order w.r.t. the energy.

        There are in total 19,736 data rows, each corresponding to one charge particle (a group of electrons) comprising of -1.33 fC of charge (8266 electrons).
        Thus, the total charge in the bunch is 26.25 pC. Since the bunches are exiting the linac at the rate of the rf frequency (11.424 GHz), the current is 300 mA.

        Data format:
        x [mm]      y [mm]      z[mm]       px [MeV/c]      py [MeV/c]      pz [MeV/c]      E [MeV]

        """
        Data = np.loadtxt(self.Data, skiprows=2)
        self.x = Data[:, 0]
        self.y = Data[:, 1]
        self.z = Data[:, 2]
        self.px = Data[:, 3]
        self.py = Data[:, 4]
        self.pz = Data[:, 5]
        self.E = Data[:, 6]

    def __ReadInCSTdata(self):
        """
        Read in CST data file of format:

        [posX   posY    posZ    particleID      sourceID    mass    macro-charge    time    Current     momX    momY    momZ    SEEGeneration]
        """

        Data = np.loadtxt(self.Data, skiprows=8)
        self.x = Data[:, 0]
        self.y = Data[:, 1]
        self.z = Data[:, 2]
        self.px = Data[:, 9]
        self.py = Data[:, 10]
        self.pz = Data[:, 11]

        # calculate energies
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2)) * self.me_MeV
        self.TOT_E = np.sqrt(Totm ** 2 + self.me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self.me_MeV)
        self.E = Kin_E

        print('Read in of CST data succesful')

    def __ReadInTopasASCIIdata(self):
        """
        Read in topas ASCII phase space data
        assumption is that this in in cm and MeV
        """

        PhaseSpace = tp.read_ntuple(self.Data)
        ParticleTypes = PhaseSpace['Particle Type (in PDG Format)']
        ParticleDir = PhaseSpace['Flag to tell if Third Direction Cosine is Negative (1 means true)']
        PaarticleDirInd = ParticleDir == 1  # only want forward moving particles
        ParticleTypeInd = ParticleTypes == 11  # only want electrons
        Ind = np.logical_and(PaarticleDirInd,ParticleTypeInd)

        self.x = PhaseSpace['Position X [cm]'][Ind] * 1e1
        self.y = PhaseSpace['Position Y [cm]'][Ind] * 1e1
        self.z = PhaseSpace['Position Y [cm]'][Ind] * 1e1
        DirCosineX = PhaseSpace['Direction Cosine X'][Ind]
        DirCosineY = PhaseSpace['Direction Cosine Y'][Ind]
        self.E = PhaseSpace['Energy [MeV]'][Ind]



        # figure out the momentums:
        self.__CosinesToMom(DirCosineX, DirCosineY, self.E)

        print('hello!')

    def __CosinesToMom(self,DirCosineX,DirCosineY,E):
        """
        Internal function to convert direction cosines and energy back into momentum
        """
        # first calculte total momentum from total energy:
        P = np.sqrt(E**2 + self.me_MeV**2)
        self.TOT_E = np.sqrt(P ** 2 + self.me_MeV ** 2)
        self.px = np.multiply(P,DirCosineX)
        self.py = np.multiply(P, DirCosineY)
        self.pz = np.sqrt(P**2 - self.px**2 - self.py**2)

    def __ReadInNumpyData(self):
        """
        Read mnumpy array of the form
        [x y z px py pz]
        """
        self.x = self.Data[:, 0]
        self.y = self.Data[:, 1]
        self.z = self.Data[:, 2]
        self.px = self.Data[:, 3]
        self.py = self.Data[:, 4]
        self.pz = self.Data[:, 5]

        # calculate energies
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self.me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self.me_MeV)
        self.E = Kin_E

    def __AnalyseEnergyDistribution(self):
        self.meanEnergy = np.mean(self.E)
        self.EnergySpread = np.std(self.E)

    def __CalculateTwissParameters(self):
        """
        Calculate the twiss parameters
        """
        # Calculate in X direction
        self.x2 = np.mean(np.square(self.x))
        self.xp = np.divide(self.px, self.pz) * 1e3
        self.xp2 = np.mean(np.square(self.xp))
        self.x_xp = np.mean(np.multiply(self.x, self.xp))

        self.epsilon = np.sqrt((self.x2 * self.xp2) - (self.x_xp ** 2)) * np.pi
        self.alpha = -self.x_xp / self.epsilon
        self.beta = self.x2 / self.epsilon
        self.gamma = self.xp2 / self.epsilon

    def __CheckEnergyCalculation(self):
        """
        For the SLAC data, if we understand the units correctly, we should be able to recover the energy from the momentum....
        """
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self.me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self.me_MeV)

        E_error = max(self.E - Kin_E)
        if E_error > .01:
            sys.exit('Energy check failed: read in of data is wrong.')

    def __CalculateBetaAndGamma(self):
        """
        Calculate the beta and gamma factors from the momentum data

        input momentum is assumed to be in units of MeV/c
        """

        self.BetaX = np.divide(self.px, self.TOT_E)
        self.BetaY = np.divide(self.py, self.TOT_E)
        self.BetaZ = np.divide(self.pz, self.TOT_E)

        self.GammaX = 1 / np.sqrt(1 - np.square(self.BetaX))
        self.GammaY = 1 / np.sqrt(1 - np.square(self.BetaY))
        self.GammaZ = 1 / np.sqrt(1 - np.square(self.BetaZ))

    def GenerateCSTParticleImportFile(self,Zoffset=None):
        """
        Generate a phase space which can be directly imported into CST
        For a constant emission model: generate a .pid ascii file
        Below is the example from CST:

        % Use always SI units.
        % The momentum (mom) is equivalent to beta* gamma.
        %
        % Columns: pos_x  pos_y  pos_z  mom_x  mom_y  mom_z  mass  charge  current

        1.0e-3   4.0e-3  -1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        2.0e-3   4.0e-3   1.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   1.0e-6
        3.0e-3   2.0e-3   1.0e-3   1.0   2.0   2.0   9.11e-31  -1.6e-19   1.0e-6
        4.0e-3   4.0e-3   5.0e-3   1.0   2.0   1.0   9.11e-31  -1.6e-19   2.0e-6
        """

        # Split the original file and extract the file name
        NparticlesToWrite = np.size(self.x)  # use this to create a smaller PID flie for easier trouble shooting
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '.pid'
        # generate other information required by pid file:

        Charge = np.ones(np.shape(self.x)) * constants.elementary_charge * -1
        Mass = np.ones(np.shape(self.x)) * constants.electron_mass
        Current = np.ones(np.shape(self.x)) * self.TotalCurrent / np.size(self.x)  # very crude approximation!!
        x = self.x * 1e-3  ## convert to m
        y = self.y * 1e-3
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z
        else:
            self.zOut = self.z + Zoffset
        px = np.multiply(self.BetaX, self.GammaX)
        py = np.multiply(self.BetaY, self.GammaY)
        pz = np.multiply(self.BetaZ, self.GammaZ)
        # generate PID file
        Data = [x[0:NparticlesToWrite], y[0:NparticlesToWrite], self.zOut[0:NparticlesToWrite],
                px[0:NparticlesToWrite], py[0:NparticlesToWrite], pz[0:NparticlesToWrite],
                Mass[0:NparticlesToWrite], Charge[0:NparticlesToWrite], Current[0:NparticlesToWrite]]

        Data = np.transpose(Data)
        np.savetxt(WritefilePath, Data, fmt='%01.3e', delimiter='      ')

    def GenerateTopasImportFile(self, Zoffset=None):
        """
        Convert Phase space into format appropriate for topas.
        You can read more about the required format
        `Here <https://topas.readthedocs.io/en/latest/parameters/scoring/phasespace.html>`_

        :param Zoffset: number to add to the Z position of each particle. To move it upstream, Zoffset should be negative.
         No check is made for units, the user has to figure this out themselves! If Zoffset=None, the read in X value
         will be used.
        :type Zoffset: None or double
        """
        print('generating topas data file')
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.phsp'

        # write the header file:
        self.__GenerateTopasHeaderFile()

        # generare the required data and put it all in an ndrray
        DirCosX, DirCosY = self.__CalculateDirectionCosines()
        Weight = np.ones(len(self.x))  # i think weight is scaled relative to particle type
        ParticleType = 11 * np.ones(len(self.x))   # 11 is the particle ID for electrons
        ThirdDirectionFlag = np.zeros(len(self.x))  # not really sure what this means.
        FirstParticleFlag = np.ones(len(self.x))  # don't actually know what this does but as we import a pure phase space
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z
        else:
            self.zOut = self.z + Zoffset

        # Nb: topas seems to require units of cm
        Data = [self.x * 0.1, self.y * 0.1, self.zOut * 0.1, DirCosX, DirCosY, self.E, Weight,
                ParticleType, ThirdDirectionFlag, FirstParticleFlag]

        # write the data to a text file
        Data = np.transpose(Data)
        FormatSpec = ['%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%2d', '%2d', '%2d', '%2d']
        np.savetxt(WritefilePath, Data, fmt=FormatSpec, delimiter='      ')

    def __CalculateDirectionCosines(self):
        """
        Calculate direction cosines, which are required for topas import:

        U (direction cosine of momentum with respect to X)
        V (direction cosine of momentum with respect to Y)

        This is only intended to be used from within the class (private method)
        """
        P = np.sqrt(self.px**2 + self.py**2 + self.pz**2)
        U = self.px/P
        V = self.py/P
        return U,V

        # Does it want these in terms of momentum or velocity? I don't think they are the same :-/

    def __GenerateTopasHeaderFile(self):
        """
        Generate the header file required for a topas phase space source.
        This is only intended to be used from within the class (private method)
        """

        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.header'


        ParticlesInPhaseSpace = str(len(self.x))
        TopasHeader = []

        TopasHeader.append('TOPAS ASCII Phase Space\n')
        TopasHeader.append('Number of Original Histories: ' + ParticlesInPhaseSpace )
        TopasHeader.append('Number of Original Histories that Reached Phase Space: ' + ParticlesInPhaseSpace)
        TopasHeader.append('Number of Scored Particles: ' + ParticlesInPhaseSpace + '\n')
        TopasHeader.append('Columns of data are as follows:')
        TopasHeader.append(' 1: Position X [cm]')
        TopasHeader.append(' 2: Position Y [cm]')
        TopasHeader.append(' 3: Position Z [cm]')
        TopasHeader.append(' 4: Direction Cosine X')
        TopasHeader.append(' 5: Direction Cosine Y')
        TopasHeader.append(' 6: Energy [MeV]')
        TopasHeader.append(' 7: Weight')
        TopasHeader.append(' 8: Particle Type (in PDG Format)')
        TopasHeader.append(' 9: Flag to tell if Third Direction Cosine is Negative (1 means true)')
        TopasHeader.append(' 10: Flag to tell if this is the First Scored Particle from this History (1 means true)\n')
        TopasHeader.append('Number of e-: ' + ParticlesInPhaseSpace + '\n')
        TopasHeader.append('Minimum Kinetic Energy of e-: ' + str(min(self.E)) + ' MeV\n')
        TopasHeader.append('Maximum Kinetic Energy of e-: ' + str(max(self.E)) + ' MeV')

        # open file:
        try:
            f = open(WritefilePath, 'w')
        except FileNotFoundError:
            sys.exit('couldnt open file for writing')

        # Write file line by line:
        for Line in TopasHeader:
            f.write(Line)
            f.write('\n')

        f.close

    def PlotPhaseSpaceX(self):
        plt.figure()
        plt.scatter(self.x, self.xp, s=1, marker='.')
        plt.xlabel('X [mm]')
        plt.ylabel("X' [mrad]")

        # # add in the phase elipse
        xq = np.linspace(min(self.x), max(self.x), 1000)
        xpq = np.linspace(min(self.xp), max(self.xp), 1000)
        [ElipseGridx, ElipseGridy] = np.meshgrid(xq, xpq)
        EmittanceGrid = (self.gamma * np.square(ElipseGridx)) + \
                        (2 * self.alpha * np.multiply(ElipseGridx, ElipseGridy)) + \
                        (self.beta * np.square(ElipseGridy))
        tol = .01 * self.epsilon
        Elipse = (EmittanceGrid >= self.epsilon - tol) & (EmittanceGrid <= self.epsilon + tol)
        ElipseIndex = np.where(Elipse == True)
        elipseX = ElipseGridx[ElipseIndex]
        elipseY = ElipseGridy[ElipseIndex]

        plt.scatter(elipseX, elipseY, s=1, c='r')
        # plt.ylim([min(elipseY)*2,max(elipseY)*2])

        TitleString = "\u03C0\u03B5: %1.1f mm mrad, \u03B1: %1.1f, \u03B2: %1.1f, \u03B3: %1.1f" % \
                      (self.epsilon, self.alpha, self.beta, self.gamma)
        plt.title(TitleString)
        plt.grid(True)

    def PlotEnergyHistogram(self):
        plt.figure()
        plt.hist(self.E, bins=1000)
        plt.xlabel('Energy [Mev]')
        plt.ylabel('N counts')
        plt.title('Energy histogram')

    def PlotParticlePositions(self):


        fig, axs = plt.subplots(1, 3)
        fig.set_size_inches(15,5)

        axs[0].scatter(self.x, self.y, s=1)
        axs[0].set_xlabel('X position [mm]')
        axs[0].set_ylabel('Y position [mm]')
        # axs[0].set_xlim([-1.5,1.5])
        # axs[0].set_ylim([-1.5,1.5])

        axs[1].hist(self.x,bins=100)
        axs[1].set_xlabel('Y position [mm]')
        axs[1].set_ylabel('counts')

        axs[2].hist(self.y, bins=100)
        axs[2].set_xlabel('X position [mm]')
        axs[2].set_ylabel('counts')

        plt.tight_layout()
        plt.show()