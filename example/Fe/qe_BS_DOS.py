import numpy as np
import pandas as pd
import numpy.linalg as LA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import qeschema

import pickle 
from  tqdm import tqdm
import os
import re

import wannier_loader

Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr

import contextlib

@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally: 
        np.set_printoptions(**original)



class VASP_analyse_spinpolarized_3D():
    def __init__(self, dir, name):
        self.directory = dir # './'
        self.name = name # 'CrTe2'

        self.get_full_DOS()
        self.get_crystell_str()
        self.get_sym_points()
        self.get_hr()

    def get_full_DOS(self):
        
        self.eDOS = []
        self.dosup = []
        self.dosdn = []
        self.efermi = 0
        try:
            with open(self.directory + "qe/dos.dat") as f:
                line = f.readline()
                self.efermi = float(re.search(r"EFermi =\s*(-?\d+\.\d*)\s*eV", line).group(1))
                for line in f:
                    if not line.strip():
                        continue
                    energy, edosup, edosdn, *_ = line.split()
                    self.eDOS.append(float(energy))
                    self.dosup.append(float(edosup))
                    self.dosdn.append(float(edosdn))
        except IOError:
            print("Error: DOS file does not appear to exist.")
        print(f'efermi {self.efermi:.2f}')
        self.eDOS = np.array(self.eDOS)
        self.dosup = np.array(self.dosup)
        self.dosdn = np.array(self.dosdn)


    def get_crystell_str(self):
        pw_document = qeschema.PwDocument()
        try:
            with open(self.directory+ "qe/data-file-schema.xml") as fin:
                pass
        except IOError:
            print("Error: data-file-schema.xml file does not appear to exist.")

        pw_document.read(self.directory+ "qe/data-file-schema.xml")
        acell = np.array(pw_document.get_cell_parameters())*Bohr2Ang
        V = LA.det(acell)
        print(f'Unit Cell Volume:   {V:.4f}  (Ang^3)')
        b1 = 2*np.pi*np.cross(acell[1], acell[2])/V
        b2 = 2*np.pi*np.cross(acell[2], acell[0])/V
        b3 = 2*np.pi*np.cross(acell[0], acell[1])/V
        self.bcell = np.array([b1, b2, b3])
        self.acell = acell
        # print('Reciprocal-Space Vectors (Ang^-1)')
        # with printoptions(precision=10, suppress=True):
        #     print(b)
        print('Reciprocal-Space Vectors (Ang^-1)')
        with printoptions(precision=10, suppress=True):
            print(self.bcell)

        print('Real-Space Vectors (Ang)')
        with printoptions(precision=10, suppress=True):
            print(acell)


    def get_sym_points(self):
        self.HighSymPointsNames = []
        self.HighSymPointsDists = []
        self.HighSymPointsCoords = []
        try:
            with open(self.directory + "qe/band.in") as fin:
                file_row = fin.readline()
                while file_row.split()[0] != 'K_POINTS':
                    file_row = fin.readline()
                    # print(file_row.split()[0])
                
                n_strings = int(fin.readline())
                k_string = fin.readline().split()
                Letter_prev = k_string[5]
                dist = 0.0
                k_prev = np.array(list(map(float, k_string[:3])))
                
                self.HighSymPointsNames.append(Letter_prev)
                self.HighSymPointsDists.append(dist)
                self.HighSymPointsCoords.append(k_prev)

                for _ in range(n_strings-1):
                    line = fin.readline()
                    k_string = line.split()
                    Letter_new = k_string[5]
                    k_new = np.array(list(map(float, k_string[:3])))
                    delta_k = k_new - k_prev
                    dist += LA.norm(self.bcell.T@delta_k)
                    k_prev = k_new
                    self.HighSymPointsNames.append(Letter_new)
                    self.HighSymPointsDists.append(dist)
                    self.HighSymPointsCoords.append(k_prev)
        
        except IOError:
            print("Error: band.in file does not appear to exist.")

        print(self.HighSymPointsNames)
        # print(self.HighSymPointsDists)
    
                   
    @staticmethod
    def get_spin_BS(path):
        hr_fact_data = []
        with open(path) as f:
                band = 0
                hr_fact_data.append([])

                for line in f:
                    
                    if line == ' \n':
                        hr_fact_data[-1] = np.array(hr_fact_data[-1])
                        hr_fact_data.append([])
                        band+=1
                    else:
                        hr_string = line.split()
                        hr_fact_data[-1].append(np.array([
                            float(hr_string[0]), float(hr_string[1]), 
                        ]))
                        
        hr_fact_data = np.array(hr_fact_data[:-1])  
        return hr_fact_data
    

    def plot_FullDOS(self, saveQ=False, picname='DOS'):
        fig, dd = plt.subplots() 
        
        dd.plot(self.eDOS - self.efermi, self.dosup, 
                    label="DOS up", color='red', linewidth=0.5)

        dd.plot(self.eDOS - self.efermi, -self.dosdn, 
                    label="DOS dn", color='blue', linewidth=0.5)

        plt.fill_between(
                x= self.eDOS-self.efermi, 
                y1=self.dosup,
                y2=-self.dosdn,
                color= "grey",
                alpha= 0.1)

        # locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        dd.yaxis.set_major_locator(MultipleLocator(2))
        dd.xaxis.set_minor_locator(MultipleLocator(1))
        dd.xaxis.set_major_locator(MultipleLocator(2))

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title("Spinpolarized DOS")
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        
        dd.vlines(0, ymin=-30, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        dd.hlines(0, xmin=-30, xmax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((-10, 10))
        dd.set_xlim((-5, 5))
        
        if saveQ:
            plt.savefig('./'+ picname, dpi=200, bbox_inches='tight')
        plt.show()


    def get_pDOS(self):
        
        def read_pdos(file, i):
            df = pd.read_csv(self.directory +'qe/'+ str(file), sep='\s+', skiprows=[0], header=None)
            e, pdos = df.iloc[:, 0], df.iloc[:, [i,i+2]].sum(axis=1)
            return e, pdos

        def list_pdos_files(path):
            for f in os.listdir(path):
                
                if f.startswith( self.name + '.pdos_atm'):
                    match = re.search(
                        r"pdos_atm#(\d+)\((\w+)\)\_wfc#(\d+)\((\w+)\)", f)
                    if not match:
                        raise FileNotFoundError
                    yield f, match.groups()

        self.pdos_up = {"s": dict(), "p": dict(), "d": dict()}
        self.pdos_dn = {"s": dict(), "p": dict(), "d": dict()}
        for file, info in list_pdos_files(self.directory + 'qe/'):
            atom_number,  _, _, orbital_type = info
            
            self.ePDOS, pdos_up = read_pdos(file, 1)#spinup
            self.pdos_up[orbital_type].update({atom_number: pdos_up})

            _, pdos_dn = read_pdos(file, 2)#spindown
            self.pdos_dn[orbital_type].update({atom_number: pdos_dn})


    def plot_pDOS(self, element="1"):
        # plt.figure(figsize= (40, 20))
        fig, dd = plt.subplots()  # Create a figure containing a single axes.

        ########################### UP spin
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_up['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_up[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_up[orbital_type][str(element)]
                atom_tdos += self.pdos_up[orbital_type][str(element)]

        atom_pdos = pd.DataFrame(atom_pdos)
        atom_pdos.index = self.ePDOS -self.efermi

        dd.plot(self.ePDOS-self.efermi, atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 

        # for orbital_type in atom_pdos.keys():
        if atom_pdos['s'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'][0] is not None:
            dd.plot(atom_pdos.index, atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
        plt.fill_between(
                x= self.ePDOS-self.efermi, 
                y1=atom_tdos, 
                # where= (-1 < t)&(t < 1),
                color= "grey",
                alpha= 0.1)


        ########################### DOWN spin
        atom_pdos = {"s": None, "p": None, "d": None}
        atom_tdos = np.zeros((len(self.pdos_dn['s']['1'])))
        
        for orbital_type in atom_pdos.keys():
            if str(element) in self.pdos_dn[orbital_type].keys():
                atom_pdos[orbital_type] = self.pdos_dn[orbital_type][str(element)]
                atom_tdos += self.pdos_dn[orbital_type][str(element)]

        atom_pdos = pd.DataFrame(atom_pdos)
        atom_pdos.index = self.ePDOS -self.efermi

        dd.plot(self.ePDOS-self.efermi, -atom_tdos, color='green', label='TDOS '+element, linewidth=0.8, linestyle='dashed') 
        
        # for orbital_type in atom_pdos.keys():
        if atom_pdos['s'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['s'], 
                    label="s DOS", color='c', linewidth=0.5)

        if atom_pdos['p'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['p'], 
                    label="p DOS", color='red', linewidth=0.5)

        if atom_pdos['d'][0] is not None:
            dd.plot(atom_pdos.index, -atom_pdos['d'], 
                    label="d DOS", color='blue', linewidth=0.5)
            
        plt.fill_between(
                x= self.ePDOS-self.efermi, 
                y1=-atom_tdos, 
                # where= (-1 < t)&(t < 1),
                color= "grey",
                alpha= 0.1)


        locator = AutoMinorLocator()
        dd.yaxis.set_minor_locator(locator)
        dd.xaxis.set_minor_locator(locator)

        dd.set_ylabel('Density of states')  # Add an x-label to the axes.
        dd.set_xlabel(r'$E-E_f$ [eV]')  # Add a y-label to the axes.
        dd.set_title(element +" pDOS")
        dd.legend()  # Add a legend.
        
        dd.vlines(0, ymin=0, ymax=30*1.2, colors='black', ls='--', alpha= 1.0, linewidth=1.0)
        # fig.set_figwidth(12)     #  ширина и
        # fig.set_figheight(6)    #  высота "Figure"
        # dd.set_ylim((-10, 10))
        # dd.set_xlim((-7, 3))
        # plt.savefig(element+'_DOS.png', dpi=1000)
        # plt.savefig('./pics/'+ element+'_DOS.png', dpi=200)
        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        dd.set_ylim((-15, 10))
        dd.set_xlim((-5, 5))
        # plt.savefig('./2pub/pics/pDOS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def print_bands_range(self, band_from=None, band_to=None):
        if band_from is None:
            band_from = 0
        if band_to is None:
            band_to = self.nbandsDFT

        print(f'efermi {self.efermi:.2f}')
        print("-------------SPIN UP---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_up[band_num, : ,1]) :.2f} to  {max(self.hDFT_up[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_up[band_num, : ,1]) -self.efermi :.2f} to  {max(self.hDFT_up[band_num, : ,1]) - self.efermi:.2f}' )
        print("-------------SPIN DN---------------")
        for band_num in range(band_from,band_to):
            print(f'band {band_num+1} eV from  {min(self.hDFT_dn[band_num, : ,1]) :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) :.2f} \
                eV-eF from  {min(self.hDFT_dn[band_num, : ,1]) - self.efermi :.2f} to  {max(self.hDFT_dn[band_num, : ,1]) - self.efermi:.2f}' )


    def get_hr(self):
        self.hDFT_up = self.get_spin_BS(self.directory +'qe/bands_up.dat.gnu')
        self.hDFT_dn = self.get_spin_BS(self.directory +'qe/bands_dn.dat.gnu')
        self.nbandsDFT = self.hDFT_up.shape[0]


    def plot_BS(self):
        
        
        kmax = self.hDFT_up[0, -1 ,0]
        fig, dd = plt.subplots() 
        
        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists/kmax*2.0

        for band in range(self.nbandsDFT):
            if band == 0:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi, label='up', color='red', linewidth=0.7,
                            alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi, label='down', color='blue', linewidth=0.7,
                            alpha=1.0)
            else:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi,  color='red', linewidth=0.7,
                        alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi,  color='blue', linewidth=0.7,
                        alpha=1.0)


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        # dd.set_xlabel('rho')  # Add a y-label to the axes.
        # dd.set_title("pk/p from density")
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(-10, 10)

        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        #plt.savefig('./2pub/pics/BS.png', dpi=200, bbox_inches='tight')

        plt.show()


    def get_qe_kpathBS(self, printQ=False):
        # N_points = 10
        kmax = self.hDFT_up[0, -1 ,0]
        qe2wan =  self.HighSymPointsDists[-1]/kmax*2.0
        N_points_direction = 10
        
        NHSP = len(self.HighSymPointsCoords)
        with open(self.directory + "kpaths/kpath_qe2.dat", "w") as fout2:

            Letter_prev = self.HighSymPointsNames[0]
            dist = 0.0
            k_prev = self.HighSymPointsCoords[0]
            print(Letter_prev)

            for HSP_ind in range(1, NHSP):
                
                Letter_new = self.HighSymPointsNames[HSP_ind]
                k_new = self.HighSymPointsCoords[HSP_ind]
                
                delta_k = k_new - k_prev
                
                num_points = 20 
                for point in range(num_points + (HSP_ind==NHSP-1)):
                    k_to_write = k_prev +   delta_k/(num_points)*point 
                    # print(k_to_write)
                    if point == 0:
                        Letter_to_write =  Letter_prev
                    elif (HSP_ind == NHSP-1 and point == num_points):
                        Letter_to_write =  Letter_new
                    else:
                        Letter_to_write = '.'
                    fout2.write( 
                        f'{Letter_to_write} {k_to_write[0]:.8f}  {k_to_write[1]:.8f} {k_to_write[2]:.8f}  \t {dist/qe2wan:.8f} \n'
                    )


                    k_to_write =     np.array(list(map(int,   k_to_write*N_points_direction)))  
                    dist += LA.norm(self.bcell.T@delta_k/(num_points))
                
                print(Letter_new)
                k_prev = k_new[:]
                Letter_prev = Letter_new 
                        
    # Wannier90 interface 
    def load_wannier(self):
        self.wannier = wannier_loader.Wannier_loader(self.directory, self.name)
        self.wannier.load_kpath('./kpaths/kpath_qe2.dat')
        self.BS_wannier_dn = self.wannier.get_wannier_BS(spin=0)
        self.BS_wannier_up = self.wannier.get_wannier_BS(spin=1)


    def plot_wannier_BS(self):
        kmax = self.hDFT_up[0, -1 ,0]
        qe2wan =  self.HighSymPointsDists[-1]/kmax*2.0
        nwa = self.BS_wannier_dn.shape[1]

        label_ticks = self.HighSymPointsNames
        normal_ticks = self.HighSymPointsDists/kmax*2.0

        fig, dd = plt.subplots()  # Create a figure containing a single axes.
        for band in range(self.nbandsDFT):
            if band == 0:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi, label='up', color='red', linewidth=0.7,
                            alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi, label='down', color='blue', linewidth=0.7,
                            alpha=1.0)
            else:
                dd.plot(self.hDFT_up[band, : ,0], 
                        self.hDFT_up[band, : , 1] - self.efermi,  color='red', linewidth=0.7,
                        alpha=1.0)

                dd.plot(self.hDFT_dn[band, : ,0], 
                        self.hDFT_dn[band, : , 1] - self.efermi,  color='blue', linewidth=0.7,
                        alpha=1.0)


        for band in range(nwa):
            if band == 0:
                
                dd.plot(self.wannier.kpath_dists_qe*2.0,
                        self.BS_wannier_up[ : , band] - self.efermi , label='up', color='r', alpha=0.5, linewidth=3)

                dd.plot(self.wannier.kpath_dists_qe*2.0,
                        self.BS_wannier_dn[ : , band] - self.efermi , label='down', color='b', alpha=0.5, linewidth=3)
                
            else:
                
                dd.plot(self.wannier.kpath_dists_qe*2.0,
                        self.BS_wannier_up[ : , band] - self.efermi , color='r', alpha=0.3, linewidth=3)

                dd.plot(self.wannier.kpath_dists_qe*2.0,
                        self.BS_wannier_dn[ : , band] - self.efermi ,  color='b', alpha=0.3, linewidth=3)


        dd.set_ylabel(r'E - $E_f$ [Ev]')  # Add an x-label to the axes.
        # dd.set_xlabel('rho')  # Add a y-label to the axes.
        # dd.set_title("pk/p from density")
        dd.legend(prop={'size': 8}, loc='upper right', frameon=False)  # Add a legend.
        plt.xticks(normal_ticks, label_ticks)
        dd.yaxis.set_minor_locator(MultipleLocator(1))
        plt.grid(axis='x')
        dd.axhline(y=0, ls='--', color='k')
        plt.xlim(normal_ticks[0], normal_ticks[-1])
        plt.ylim(-15, 15)

        width = 7
        fig.set_figwidth(width)     #  ширина и
        fig.set_figheight(width/1.6)    #  высота "Figure"
        # plt.savefig('./2pub/pics/BS_wannier.png', dpi=200, bbox_inches='tight')

        plt.show()
