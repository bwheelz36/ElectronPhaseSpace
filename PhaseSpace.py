# PhaseSpace.py
import numpy as np
import scipy.constants
from matplotlib import pyplot as plt
from scipy import constants
from scipy.stats import norm
import os, sys
import glob
import logging

logging.basicConfig(level=logging.WARNING)
import warnings
import matplotlib.patches as patches
from scipy.stats import gaussian_kde
from scipy import constants
from time import perf_counter
import re

sys.path.append(os.path.abspath("../PhaserGeometry"))
import topas2numpy as tp


class FigureSpecs:
    """
    Thought this might be the easiest way to ensure universal parameters accross all figures
    """

    LabelFontSize = 14
    TitleFontSize = 16
    Font = 'serif'
    AxisFontSize = 14


class ElectronPhaseSpace:
    """
    A set of functions for analysing and converting phase spaces.
    At the moment there is an implict assumption that electron phase spaces are being used, which impacts on the conversion
    of momentum to kinetic energy. This could be updated with minimal effort if needed.

    example of using ElectronPhaseSpace class::

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
    """

    def __init__(self, Data, verbose=False, weight_position_plot=False):
        """
        Data can be one of four things:

        1. CST trk particle monitor
        2. Data from SLAC
        3. Topas ASCII phase space
        4. Data from tibaray
        5. Numpy array

        For the third case, the data should be
        [x y z px py pz weight]
        with x y z in mm
        and px py pz  in [MeV/c]

        """
        self.Data = Data
        self.ParticleType = 'electrons'
        self._me_MeV = 0.511  # electron mass in MeV
        self._c = 2.998e8  # speed of light in m/s
        self.weight = None  # relative weight of each particle; overwritten with ones if not set during read in
        self.verbose = verbose
        self._weight_position_plot = weight_position_plot  # weights plots by density. Looks better but is slow.
        self.FigureSpecs = FigureSpecs()
        self.ROI = None  # [700, 5]
        '''
        '^^ this is used when asessing number of particles in a certain radius.
        If the particle is projected to fall within the ROI at ROI[0]=z and ROI[1] =r, it is counted.
        set it to None to count all particles
        '''
        plt.rc('font', family=self.FigureSpecs.Font)
        plt.rc('xtick', labelsize=self.FigureSpecs.AxisFontSize)
        plt.rc('ytick', labelsize=self.FigureSpecs.AxisFontSize)

        # we will try to figure out what data is being put in.
        # this function will probably require some updates to work with different data formats
        self.__DetectDataType()
        self.__ReadInData()

        # calculations:
        self.__CalculateBetaAndGamma()
        self.__ConvertMomentumToVelocity()
        self.__CalculateTwissParameters()  # at the moment just caclculates transverse phse space in X.
        self.__AnalyseEnergyDistribution()

        if self.verbose == True:
            self.PrintData()

    def __DetectDataType(self):
        """
        This function is used to detect what type of data the user has put in. see __init__ for allowable data types
        """
        if isinstance(self.Data, np.ndarray):
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
                tibarayFirstLinePattern = 'x y z rxy Bx By Bz G t m q nmacro rmacro ID'
                if FirstLine.find(SLACfirstLinePattern) >= 0:
                    self.DataType = "SLAC"
                elif FirstLine.find(CSTfirstLinePattern) >= 0:
                    self.DataType = "CST"
                elif FirstLine.find(tibarayFirstLinePattern) >= 0:
                    self.DataType = 'tibaray'
                else:
                    sys.exit('unable to determine data type')
        else:
            logging.error('unable to determine file type; making a guess: topas')
            self.DataType = 'topas'

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
            self.__ReadInTopasData()
        elif self.DataType == 'tibaray':
            self.OutputDataLoc, Filename = os.path.split(self.Data)
            self.OutputFile, Filetype = os.path.splitext(Filename)
            self.__ReadInTibarayData()

        if self.weight is None:
            self.weight = np.ones(len(self.x))

    def __ReadInTibarayData(self):
        """
        Read in data from tibaray, which had header:

        `x y z rxy Bx By Bz G t m q nmacro rmacro ID`

        x[m] y[m] z[m] rxy[m] (positions in space)
        Bx[v/c] By[v/c] Bz[v/c] (normalized velocity)
        G [Lorentz factor]
        t[time, s]
        m[kg, mass of elementary particle of the macro particle]
        q[Coulomb, charge of elementary particle of the macro particle]
        nmacro[number of electrons in this macro particle]
        rmacro[NA]
        ID[Macro Particle ID Number]
        """
        warnings.warn('Read in of this file format is still under development')
        Data = np.loadtxt(self.Data, skiprows=1)
        self.x = Data[:, 0] * 1e3  # mm to m
        self.y = Data[:, 1] * 1e3
        self.z = Data[:, 2] * 1e3
        Bx = Data[:, 4]
        By = Data[:, 5]
        Bz = Data[:, 6]
        Gamma = Data[:, 7]
        self.t = Data[:, 8]
        m = Data[:, 9]
        q = Data[:, 10]
        nmacro = Data[:, 11]
        rmacro = Data[:, 12]
        ID = Data[:, 13]

        self.px = np.multiply(Bx, Gamma) * self._me_MeV
        self.py = np.multiply(By, Gamma) * self._me_MeV
        self.pz = np.multiply(Bz, Gamma) * self._me_MeV

        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self._me_MeV)
        self.weight = nmacro
        self.E = Kin_E

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
        self.px = Data[:, 9] * self._me_MeV
        self.py = Data[:, 10] * self._me_MeV
        self.pz = Data[:, 11] * self._me_MeV
        _macro_charge = Data[:, 6]
        self.weight = _macro_charge / scipy.constants.elementary_charge

        # calculate energies
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self._me_MeV)
        self.E = Kin_E

        print('Read in of CST data succesful')

    def __ReadInTopasData(self):
        """
        Read in topas  data
        assumption is that this is in cm and MeV
        """

        PhaseSpace = tp.read_ntuple(self.Data)
        ParticleTypes = PhaseSpace['Particle Type (in PDG Format)']
        ParticleTypes = ParticleTypes.astype(int)
        ParticleDir = PhaseSpace['Flag to tell if Third Direction Cosine is Negative (1 means true)']
        # ParticleDir = ParticleDir.astype(int)
        # ParticleDirInd = ParticleDir == 1  # only want forward moving particles
        if self.ParticleType == 'electrons':
            ParticleTypeInd = ParticleTypes == 11  # only want electrons
            Ind = ParticleTypeInd
        elif self.ParticleType == 'gamma':
            ParticleTypeInd = ParticleTypes == 22  # only want photons
            Ind = ParticleTypeInd
        else:
            raise TypeError(f'no read in for {self.ParticleType}')

        self.x = PhaseSpace['Position X [cm]'][Ind] * 1e1
        self.y = PhaseSpace['Position Y [cm]'][Ind] * 1e1
        self.z = PhaseSpace['Position Z [cm]'][Ind] * 1e1
        self.DirCosineX = PhaseSpace['Direction Cosine X'][Ind]
        self.DirCosineY = PhaseSpace['Direction Cosine Y'][Ind]
        self.E = PhaseSpace['Energy [MeV]'][Ind]
        self.weight = PhaseSpace['Weight'][Ind]

        # figure out the momentums:
        self.__CosinesToMom()

        # if any values of pz == 0 exist, remove them with warning:
        if np.any(self.pz == 0):
            ind = self.pz == 0
            logging.warning(
                f'\nIn read in of topas data, removing {np.count_nonzero(ind)} of {ind.shape[0]} values where pz ==0. '
                f'\nWhile this is not necesarily an error,it means electrons are going completely sideways.'
                f'\nIt also makes transverse emittance calcs difficult, so im just going to delete those entries.'
                f'\nIf this is happening a lot I need to find a better solution\n')

            self.x = np.delete(self.x, ind)
            self.y = np.delete(self.y, ind)
            self.z = np.delete(self.z, ind)
            self.px = np.delete(self.px, ind)
            self.py = np.delete(self.py, ind)
            self.pz = np.delete(self.pz, ind)
            self.E = np.delete(self.E, ind)
            self.TOT_E = np.delete(self.TOT_E, ind)

    def __ConvertMomentumToVelocity(self):
        """
        I think that I may need to define the cosines in terms of velocity, and not in terms of momentum
        as I have been doing.
        I'm also not totally sure that i'm calculating these correctly....
        """
        self.vx = np.divide(self.px, (self.Gamma * self._me_MeV))
        self.vy = np.divide(self.py, (self.Gamma * self._me_MeV))
        self.vz = np.divide(self.pz, (self.Gamma * self._me_MeV))

    def __CosinesToMom(self):
        """
        Internal function to convert direction cosines and energy back into momentum
        """
        # first calculte total momentum from total energy:
        if self.ParticleType == 'electrons':
            P = np.sqrt((self.E + self._me_MeV) ** 2 - self._me_MeV ** 2)
            self.TOT_E = np.sqrt(P ** 2 + self._me_MeV ** 2)
        elif self.ParticleType == 'gamma':
            # zero rest mass
            P = np.sqrt(self.E ** 2)
            self.TOT_E = np.sqrt(P ** 2)

        self.px = np.multiply(P, self.DirCosineX)
        self.py = np.multiply(P, self.DirCosineY)
        temp = P ** 2 - self.px ** 2 - self.py ** 2
        if any(temp < 0):
            logging.warning(
                f'{np.count_nonzero(temp < 0): 1.0f} negative values found in cosine to momentum conversion. setting these to zero. '
                'Sometimes this is a simple rounding error, but '
                'sometimes it may be indicative of a serious error...')
            temp[temp < 0] = 0
        self.pz = np.sqrt(temp)

    def __ReadInNumpyData(self):
        """
        Read mnumpy array of the form
        [x y z px py pz weight]
        """
        self.x = self.Data[:, 0]
        self.y = self.Data[:, 1]
        self.z = self.Data[:, 2]
        self.px = self.Data[:, 3]
        self.py = self.Data[:, 4]
        self.pz = self.Data[:, 5]
        self.weight = self.Data[:, 6]

        # calculate energies
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self._me_MeV)
        self.E = Kin_E

    def __weighted_median(self, data, weights):
        """
        calculate a weighted median
        @author Jack Peterson (jack@tinybike.net)
        credit: https://gist.github.com/tinybike/d9ff1dad515b66cc0d87

        Args:
          data (list or numpy.array): data
          weights (list or numpy.array): weights
        """
        data, weights = np.array(data).squeeze(), np.array(weights).squeeze()
        s_data, s_weights = map(np.array, zip(*sorted(zip(data, weights))))
        midpoint = 0.5 * sum(s_weights)
        if any(weights > midpoint):
            w_median = (data[weights == np.max(weights)])[0]
        else:
            cs_weights = np.cumsum(s_weights)
            idx = np.where(cs_weights <= midpoint)[0][-1]
            if cs_weights[idx] == midpoint:
                w_median = np.mean(s_data[idx:idx + 2])
            else:
                w_median = s_data[idx + 1]
        return w_median

    def __weighted_avg_and_std(self, values, weights):
        """
        credit: https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        average = np.average(values, weights=weights)
        # Fast and numerically precise:
        variance = np.average((values - average) ** 2, weights=weights)
        return (average, np.sqrt(variance))

    def __weighted_quantile(self, values, quantiles, sample_weight=None):
        """
        credit: https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy

        Very close to numpy.percentile, but supports weights.
        NOTE: quantiles should be in [0, 1]!
        :param values: numpy.array with data
        :param quantiles: array-like with many quantiles needed
        :param sample_weight: array-like of the same length as `array`
        :param values_sorted: bool, if True, then will avoid sorting of
            initial array
        :param old_style: if True, will correct output to be consistent
            with numpy.percentile.
        :return: numpy.array with computed quantiles.
        """
        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(quantiles <= 1), 'quantiles should be in [0, 1]'
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)

    def __AnalyseEnergyDistribution(self):
        self.meanEnergy, self.stdEnergy = self.__weighted_avg_and_std(self.E, self.weight)
        self.medianEnergy = self.__weighted_median(self.E, self.weight)
        self.EnergySpreadSTD = np.std(np.multiply(self.E, self.weight))
        q75, q25 = self.__weighted_quantile(self.E, [0.25, 0.75], sample_weight=self.weight)
        self.EnergySpreadIQR = q75 - q25

    def __CalculateTwissParameters(self):
        """
        Calculate the twiss parameters
        """
        # Calculate in X direction
        self.x2 = np.average(np.square(self.x), weights=self.weight)
        self.xp = np.divide(self.px, self.pz) * 1e3
        self.xp2 = np.average(np.square(self.xp), weights=self.weight)
        self.x_xp = np.average(np.multiply(self.x, self.xp), weights=self.weight)

        self.twiss_epsilon = np.sqrt((self.x2 * self.xp2) - (self.x_xp ** 2)) * np.pi
        self.twiss_alpha = -self.x_xp / self.twiss_epsilon
        self.twiss_beta = self.x2 / self.twiss_epsilon
        self.twiss_gamma = self.xp2 / self.twiss_epsilon

    def __CheckEnergyCalculation(self):
        """
        For the SLAC data, if we understand the units correctly, we should be able to recover the energy from the momentum....
        """
        Totm = np.sqrt((self.px ** 2 + self.py ** 2 + self.pz ** 2))
        self.TOT_E = np.sqrt(Totm ** 2 + self._me_MeV ** 2)
        Kin_E = np.subtract(self.TOT_E, self._me_MeV)

        E_error = max(self.E - Kin_E)
        if E_error > .01:
            sys.exit('Energy check failed: read in of data is wrong.')

    def __CalculateBetaAndGamma(self):
        """
        Calculate the beta and gamma factors from the momentum data

        input momentum is assumed to be in units of MeV/c
        I need to figure out if BetaX and BetaY make sense, or it's just Beta
        """

        if self.ParticleType == 'gamma':
            # then this stuff makes no sense
            return

        self.TOT_P = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        self.Beta = np.divide(self.TOT_P, self.TOT_E)
        self.Gamma = 1 / np.sqrt(1 - np.square(self.Beta))

    def __CalculateDirectionCosines(self):
        """
        Calculate direction cosines, which are required for topas import:

        U (direction cosine of momentum with respect to X)
        V (direction cosine of momentum with respect to Y)

        nb: using velocity or momentum seem to give the same results

        """
        V = np.sqrt(self.px ** 2 + self.py ** 2 + self.pz ** 2)
        U = self.px / V
        V = self.py / V
        return U, V

    def __GenerateTopasHeaderFile(self):
        """
        Generate the header file required for a topas phase space source.
        This is only intended to be used from within the class (private method)
        """

        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.header'

        ParticlesInPhaseSpace = str(len(self.x))
        TopasHeader = []

        TopasHeader.append('TOPAS ASCII Phase Space\n')
        TopasHeader.append('Number of Original Histories: ' + ParticlesInPhaseSpace)
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

    # public methods

    def GenerateCSTParticleImportFile(self, Zoffset=None):
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
        warnings.warn('I havent tested this function for a very long time, so please verify that it works..')
        # Split the original file and extract the file name
        NparticlesToWrite = np.size(self.x)  # use this to create a smaller PID flie for easier trouble shooting
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '.pid'
        # generate other information required by pid file:

        Charge = self.weight * constants.elementary_charge * -1
        Mass = self.weight * constants.electron_mass
        Current = np.ones(np.shape(self.x)) * self.TotalCurrent / np.size(self.x)  # very crude approximation!!
        x = self.x * 1e-3  ## convert to m
        y = self.y * 1e-3
        if Zoffset == None:
            # Zoffset is an optional parameter to change the starting location of the particle beam (which
            # assume propogates in the Z direction)
            self.zOut = self.z * 1e-3
        else:
            self.zOut = (self.z + Zoffset) * 1e-3
        px = self.px / self._me_MeV
        py = self.py / self._me_MeV
        pz = self.pz / self._me_MeV
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
        import platform
        if 'windows' in platform.system().lower():
            warnings.warn('to generate a file that topas will accept, you need to do this from linux. I think'
                          'its the line endings.')
        WritefilePath = self.OutputDataLoc + '/' + self.OutputFile + '_tpsImport.phsp'

        # write the header file:
        self.__GenerateTopasHeaderFile()

        # generare the required data and put it all in an ndrray
        self.__ConvertMomentumToVelocity()
        DirCosX, DirCosY = self.__CalculateDirectionCosines()
        Weight = self.weight  # i think weight is scaled relative to particle type
        # Weight = np.ones(len(self.x))  # i think weight is scaled relative to particle type
        ParticleType = 11 * np.ones(len(self.x))  # 11 is the particle ID for electrons
        ThirdDirectionFlag = np.zeros(len(self.x))  # not really sure what this means.
        FirstParticleFlag = np.ones(
            len(self.x))  # don't actually know what this does but as we import a pure phase space
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
        FormatSpec = ['%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%11.5f', '%2d', '%2d', '%2d']
        np.savetxt(WritefilePath, Data, fmt=FormatSpec, delimiter='      ')
        print('success')

    def PlotPhaseSpaceX(self):

        if self.weight.max() > 1:
            warnings.warn('this plot does not take into account particle weights')
        plt.figure()

        #plot phase elipse
        xq = np.linspace(min(self.x), max(self.x), 1000)
        xpq = np.linspace(min(self.xp), max(self.xp), 1000)
        [ElipseGridx, ElipseGridy] = np.meshgrid(xq, xpq)
        EmittanceGrid = (self.twiss_gamma * np.square(ElipseGridx)) + \
                        (2 * self.twiss_alpha * np.multiply(ElipseGridx, ElipseGridy)) + \
                        (self.twiss_beta * np.square(ElipseGridy))
        tol = .01 * self.twiss_epsilon
        Elipse = (EmittanceGrid >= self.twiss_epsilon - tol) & (EmittanceGrid <= self.twiss_epsilon + tol)
        ElipseIndex = np.where(Elipse == True)
        elipseX = ElipseGridx[ElipseIndex]
        elipseY = ElipseGridy[ElipseIndex]
        plt.scatter(elipseX, elipseY, s=1, c='r')
        xmin, xmax, ymin, ymax = plt.axis()

        plt.scatter(self.x, self.xp, s=1, marker='.')
        plt.xlabel('X [mm]')
        plt.ylabel("X' [mrad]")
        plt.ylim([ymin, ymax])
        plt.xlim([xmin, xmax])
        TitleString = "\u03C0\u03B5: %1.1f mm mrad, \u03B1: %1.1f, \u03B2: %1.1f, \u03B3: %1.1f" % \
                      (self.twiss_epsilon, self.twiss_alpha, self.twiss_beta, self.twiss_gamma)
        plt.title(TitleString)

        plt.grid(True)
        plt.show()

    def PlotEnergyHistogram(self):

        try:
            test = self.Eaxs
        except AttributeError:
            self.Efig, self.Eaxs = plt.subplots()
        n, bins, patches = self.Eaxs.hist(self.E, bins=1000, weights=self.weight)
        # self.fig.set_size_inches(10, 5)
        self.Eaxs.set_xlabel('Energy [Mev]', fontsize=self.FigureSpecs.LabelFontSize)
        self.Eaxs.set_ylabel('N counts', fontsize=self.FigureSpecs.LabelFontSize)
        plot_title = self.OutputFile
        self.Eaxs.set_title(plot_title, fontsize=self.FigureSpecs.TitleFontSize)
        # self.Eaxs.tick_params(axis="y", labelsize=14)
        # self.Eaxs.tick_params(axis="x", labelsize=14)

        # self.Eaxs.set_xlim([0, 10.5])
        plt.tight_layout()
        plt.show()
        self.Eaxs.set_title(plot_title, fontsize=self.FigureSpecs.TitleFontSize)

    def PlotParticlePositions(self):

        try:
            test = self.axs[0]
        except AttributeError:
            self.fig, self.axs = plt.subplots(1, 2)
            self.fig.set_size_inches(10, 5)

        if self._weight_position_plot:
            _kde_data_grid = 150 ** 2
            print('generating weighted scatter plot...')

            xy = np.vstack([self.x, self.y])
            k = gaussian_kde(xy, weights=self.weight)
            _end_time = perf_counter()

            if self.x.shape[0] > _kde_data_grid:
                down_sample_factor = np.round(self.x.shape[0] / _kde_data_grid)
                print(f'down sampling scatter plot data by factor of {down_sample_factor}')
                # in this case we can downsample the grid...
                rng = np.random.default_rng()
                rng.shuffle(xy)  # operates in place for some confusing reason
                xy = rng.choice(xy, np.int(self.x.shape[0] / down_sample_factor), replace=False, axis=1, shuffle=False)
            z = k(xy)


            z = z / max(z)
            SP = self.axs[0].scatter(xy[0], xy[1], c=z, s=1)
            self.axs[0].set_aspect(1)
            # self.fig.colorbar(SP,ax=self.axs[0])
            self.axs[0].set_title('Particle Positions', fontsize=self.FigureSpecs.TitleFontSize)
            self.axs[0].set_xlim([-2, 2])
            self.axs[0].set_ylim([-2, 2])
            self.axs[0].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            self.axs[0].set_ylabel('Y position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            plt.show()
            plt.tight_layout()
        else:
            self.axs[0].set_title('Particle Positions', fontsize=self.FigureSpecs.TitleFontSize)
            self.axs[0].scatter(self.x, self.y, s=1, c=self.weight)
            self.axs[0].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
            self.axs[0].set_ylabel('Y position [mm]', fontsize=self.FigureSpecs.LabelFontSize)

        self.axs[1].hist(self.x, bins=100, weights=self.weight)
        self.axs[1].set_xlabel('X position [mm]', fontsize=self.FigureSpecs.LabelFontSize)
        self.axs[1].set_ylabel('counts', fontsize=self.FigureSpecs.LabelFontSize)
        plt.tight_layout()
        plt.show()

    def PlotPositionHistogram(self):

        try:
            test = self.self.PosHistAxs[0]
        except AttributeError:
            self.fig, self.PosHistAxs = plt.subplots(1, 2)
            self.fig.set_size_inches(10, 5)

            n, bins, patches = self.PosHistAxs[0].hist(self.x, bins=100, density=True)
            self.PosHistAxs[0].set_xlabel('X position [mm]')
            self.PosHistAxs[0].set_ylabel('counts')
            # self.PosHistAxs[0].set_xlim([-2, 2])
            # Plot the PDF.
            mu, std = norm.fit(self.x)
            xmin, xmax = self.PosHistAxs[0].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            self.PosHistAxs[0].plot(x, p, 'k', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            self.PosHistAxs[0].set_title(title)

            n, bins, patches = self.PosHistAxs[1].hist(self.y, bins=100)
            self.PosHistAxs[1].set_xlabel('X position [mm]')
            self.PosHistAxs[1].set_ylabel('counts')
            # self.PosHistAxs[1].set_xlim([-2, 2])
            # Plot the PDF.
            mu, std = norm.fit(self.y)
            xmin, xmax = self.PosHistAxs[1].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, mu, std)
            self.PosHistAxs[1].plot(x, p * n.max(), 'k', linewidth=2)
            title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
            self.PosHistAxs[1].set_title(title)

            plt.tight_layout()
            plt.show()

    def ProjectParticles(self, zdiff):
        """
        Update the X,Y,Z position of each particle by projecting it forward/back by zdiff.

        This serves as a crude approximation to more advanced particle transport codes, but can be used to quickly
        check results
        """
        self.x = self.x + np.divide(self.vx, self.vz) * zdiff
        self.y = self.y + np.divide(self.vy, self.vz) * zdiff
        self.z = self.z + zdiff

        # update the info on particle distribution
        self.__CalculateBetaAndGamma()
        self.__CalculateTwissParameters()  # at the moment just caclculates transverse phse space in X.
        self.__AnalyseEnergyDistribution()

    def Plot_nParticles_v_time(self):
        """
        basic plot of number of particles versus time; useful for quickly seperating out different bunches
        of electrons such that you can apply the 'filter_by_time' method
        """
        if not hasattr(self, 't'):
            warnings.warn('cant plot time as it hasnt been read in')
            return

        plt.figure()
        plt.hist(self.t, 100)
        plt.xlabel('time [AU]')
        plt.ylabel('N particles')
        plt.tight_layout()

    def filter_by_time(self, t_start, t_finish):
        """
        remove all particles outside of t_start and t_finish. times EQUAL to t_start and t_finish are kept.
        t will be in whatever units it was read in as; use Plot_nParticles_v_time
        to visualise your data
        """
        if not hasattr(self, 't'):
            warnings.warn('cant plot time as it hasnt been read in')
            return
        ind = np.logical_and(self.t >= t_start, self.t <= t_finish)
        print(f'removing {np.count_nonzero(np.logical_not(ind)) * 100 /ind.shape[0]: 1.1f}% of particles')
        print(f'new phase space will contain {np.count_nonzero(ind)} particles')
        self.x = self.x[ind]
        self.y = self.y[ind]
        self.z = self.z[ind]
        self.px = self.px[ind]
        self.py = self.py[ind]
        self.pz = self.pz[ind]
        self.E = self.E[ind]
        self.t = self.t[ind]
        self.TOT_E = self.TOT_E[ind]
        self.TOT_P = self.TOT_P[ind]
        self.weight = self.weight[ind]

        print('updating all metrics based on new phase space')
        self.__CalculateBetaAndGamma()
        self.__ConvertMomentumToVelocity()
        self.__CalculateTwissParameters()  # at the moment just caclculates transverse phse space in X.
        self.__AnalyseEnergyDistribution()

    def AssessDensityVersusR(self, Rvals=None):
        """
        Crude code to assess how many particles are in a certain radius

        If ROI = None,  then all particles are assessed.
        Otherwise, use ROI = [zval, radius] to only include particles that would be within radius r at distance z from
        the read in location
        """

        if self.weight.max() > 1:
            warnings.warn('The AssessDensityVersusR function has not been updated for weighted particles')
        if Rvals is None:
            # pick a default
            Rvals = np.linspace(0, 2, 21)

        r = np.sqrt(self.x ** 2 + self.y ** 2)
        numparticles = self.x.shape[0]
        rad_prop = []

        if self.verbose:
            if self.ROI == None:
                print(f'Assessing particle density versus R for all particles')
            else:
                print(f'Assessing particle density versus R for particles projected to be within a radius of'
                      f' {self.ROI[1]} at a distance of {self.ROI[0]}')

        for rcheck in Rvals:
            if self.ROI == None:
                Rind = r <= rcheck
                rad_prop.append(np.count_nonzero(Rind) * 100 / numparticles)
            else:
                # apply the additional ROI filter by projecting x,y to the relevant z position
                Xproj = np.multiply(self.ROI[0], np.divide(self.px, self.pz)) + self.x
                Yproj = np.multiply(self.ROI[0], np.divide(self.py, self.pz)) + self.y
                Rproj = np.sqrt(Xproj ** 2 + Yproj ** 2)
                ROIind = Rproj <= self.ROI[1]

                Rind = r <= rcheck
                ind = np.multiply(ROIind, Rind)
                rad_prop.append(np.count_nonzero(ind) * 100 / numparticles)

        self.rad_prop = rad_prop

    def PrintData(self):
        """
        can be used to print info to the termainl
        """
        if (self.DataType == 'topas') or (self.DataType == 'SLAC') or (self.DataType == 'CST'):
            Filepath, filename = os.path.split(self.Data)
            print(f'\nFor file: {filename}')
        print(
            f'\u03C0\u03B5: {self.twiss_epsilon: 1.1f} mm mrad, \u03B1: {self.twiss_alpha: 1.1f}, \u0392: {self.twiss_beta: 1.1f}, '
            f'\u03B3: {self.twiss_gamma: 1.1f}')
        print(f'Median energy: {self.medianEnergy: 1.2f} MeV \u00B1 {self.EnergySpreadIQR: 1.2f} (IQR) ')
        print(f'Mean energy: {self.meanEnergy: 1.2f} MeV \u00B1 {self.stdEnergy: 1.2f} (std) ')
        _mean_z, _std_z = self.__weighted_avg_and_std(self.z, self.weight)
        print(f'Mean Z position of input data is {_mean_z: 3.1f} mm \u00B1 {_std_z: 1.1f} (std)')
        medianEnergy = self.medianEnergy
        CutOff = .05
        ind = np.logical_and(self.E > (medianEnergy - (.05 * medianEnergy)),
                             self.E < (medianEnergy + (.05 * medianEnergy)))
        weighted_ind = np.multiply(ind, self.weight)
        weighted_percent = np.sum(weighted_ind) * 100 / np.sum(self.weight)

        print(
            f'{weighted_percent:1.1f} % of particles are within \u00B1 5% of the median dose ({medianEnergy: 1.1f} MeV) ')

        if hasattr(self, 'zOut'):
            _mean_zOut, _std_zOut = self.__weighted_avg_and_std(self.zOut, self.weight)
            print(
                f'Mean Z position of output data is {_mean_zOut: 3.1f} \u00B1 {_std_zOut: 1.1f} (std)')
