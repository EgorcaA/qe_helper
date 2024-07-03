import numpy as np
import pandas as pd
import numpy.linalg as LA
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from pymatgen.electronic_structure.plotter import BSDOSPlotter, BSPlotter, DosPlotter
from pymatgen.io.vasp.outputs import BSVasprun, Vasprun
import qeschema

import pickle 
from  tqdm import tqdm
import os
import re


Ang2Bohr = 1.8897259886
Bohr2Ang = 1./Ang2Bohr



class Wannier_loader():
    def __init__(self, dir, name):
        self.directory = dir # './'
        self.name = name # 'CrTe2'
        self.nwa = 0
        self.load_wannier90()


    def get_wannier_BS(self, spin=0): 
        band_str = []
        self.hks_bs = []
        for k in tqdm(self.k_path_qe):
            hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, spin] 
                          for R_ind, R in enumerate(self.R_coords)], axis=0 )
            self.hks_bs.append(hk)
            band_str.append(np.sort(np.real(np.linalg.eig(hk)[0])))
        band_str = np.array(band_str)
        return band_str
    

    def load_wannier90(self):
        def lload(filename):
            hr = 0
            R_coords = []
            R_weights = []
            with open(self.directory +'wannier/' + filename + '.dat') as f:
                    f.readline()
                    
                    nwa = int(f.readline().strip('\n'))
                    print("nwa ", nwa)
                    self.nwa = nwa
                    Rpts = int(f.readline().strip('\n'))
                    print("Rpts", Rpts)
                    i=1
                    hr = np.zeros((nwa, nwa, Rpts), dtype=complex)

                    R_ind = -1
                    line_ind = 0
                    for line in f:
                        
                        if i< Rpts/15+1:
                            # line.split()
                            R_weights +=  [ int(x) for x in line.split() if x.isnumeric() ]
                            # print(line)
                            i+=1
                        else:

                            hr_string = line.split()
                            if line_ind % nwa**2 == 0:
                                R_coords.append([float(hr_string[0]), float(hr_string[1]), float(hr_string[2])]) 
                                R_ind += 1
                            hr[int(hr_string[3])-1, int(hr_string[4])-1, R_ind] = float(hr_string[5])+ 1j*float(hr_string[6])
                            
                            line_ind +=1 
            return R_coords, hr

        R_coords, hr_up = lload('iron_up_hr')
        _, hr_dn = lload('iron_dn_hr')
        print(hr_up.shape, hr_dn.shape)
        self.complex_hr = np.transpose(np.array([hr_up, hr_dn]), (1,2,3,0))
        self.R_coords = R_coords
        
        # with open(self.directory + '/wannier/hr_mn' + self.name +'.pickle', 'wb') as f:
            # pickle.dump(self.complex_hr, f)
       

    def load_kpath(self, path):
        k_path = []
        kpath_dists = []

        with open(path) as f:
            for line in f:
                kpts_string = line.split()
                k_path.append(np.array([
                    float(kpts_string[1]), float(kpts_string[2]), float(kpts_string[3])
                ]))
                kpath_dists.append(float(kpts_string[4]))

        self.k_path_qe = np.array(k_path)   
        self.kpath_dists_qe = np.array(kpath_dists)   
        

    def load_hr_pickle(self):
        with open(self.directory + '/wannier/hr_mn' + self.name +'.pickle', 'rb') as f:
            self.complex_hr = pickle.load(f)

    def get_dense_hk(self):
        kpoints_adj_serial = np.mgrid[0:1:1.0/10., 0:1:1.0/10., 0:1:1.0/10.].reshape(3,-1).T
        spins = [0, 1]

        hks_spins = []
        for spin in spins:
            hks = []
            for k in tqdm(kpoints_adj_serial):
                hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, spin] 
                            for R_ind, R in enumerate(self.R_coords)], axis=0 )
                hk = np.array(hk).T
                hks.append(hk)
            hks_spins.append(hks)
        self.hks_spins = np.transpose( np.array(hks_spins) , (2,3, 1,0))
        self.kpoints_adj_serial = kpoints_adj_serial
