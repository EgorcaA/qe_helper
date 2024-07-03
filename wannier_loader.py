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
        self.get_crystell_str()
        # self.load_wannier90()
        # self.load_amulet()
        # self.load_wannier90('iron_dn_hr')


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
        self.V= V
        

    def load_amulet(self, filename=None):
        nspin = 0
        nkp = 0
        hdim = 0
        
        h = 0
        kpts = []

        # Open the file containing the Hamiltonian matrix elements
        with open(self.directory + 'wannier/hamilt.am', 'r') as iunhamilt:
            for _ in range(9):
                iunhamilt.readline()
             
            nspin = int(iunhamilt.readline().split()[-1])
            iunhamilt.readline()
            iunhamilt.readline()
            nkp = int(iunhamilt.readline().split()[-1])
            iunhamilt.readline()
            iunhamilt.readline()
            hdim = int(iunhamilt.readline().split()[-1])
            iunhamilt.readline()
            iunhamilt.readline()
            print(nspin, nkp, hdim)
            h = np.zeros((hdim, hdim, nkp, nspin), dtype=complex)
            self.nwa = hdim
            for _ in range(nkp):
                kpvec = np.array(list(map(float, iunhamilt.readline().split())))
                # print(kpvec)
                # print(self.bcell)
                kpts.append([kpvec[0], *LA.inv(self.bcell)@kpvec[1:]*(self.V**0.333)])

            iunhamilt.readline()
            iunhamilt.readline()

            for ispin in range(nspin):
                for ik in range(nkp):
                    for i in range(hdim):
                        for j in range(i, hdim):
                            # Read the real and imaginary parts from the file
                            Hre, Him = map(float, iunhamilt.readline().split())
                            h[i, j, ik, ispin] = complex(Hre, Him)
                            h[j, i, ik, ispin] = np.conj(h[i, j, ik, ispin])
        self.h_amulet = np.array(h)
        self.kpts_amulet = np.array(kpts)

        import plotly.graph_objects as go

        fig = go.Figure()

        Sx = self.kpts_amulet[:, 1].flatten()
        Sy = self.kpts_amulet[:, 2].flatten()
        Sz = self.kpts_amulet[:, 3].flatten()

        # fig.update_xaxes(range=[-1, 1])
        # fig.update_yaxes(range=[-1, 1])
        # fig.update_zaxes(range=[-1, 1])
        fig.add_trace(
            go.Scatter3d(x=Sx, y=Sy, z=Sz,
                        mode='markers',
                        #  size=np.ones(len(SSS))*1
                        marker=dict(
                                    size=3.0,
                                    # color=z,                # set color to an array/list of desired values
                                    # colorscale='Viridis',   # choose a colorscale
                                    opacity=0.7
                                )
                        )
            )
        fig.update_layout(scene = dict(
                            xaxis = dict(nticks=4, range=[-0.5,0.5],),
                            yaxis = dict(nticks=4, range=[-0.5,0.5],),
                            zaxis = dict(nticks=4, range=[-0.5,0.5],),),
                        width=700,
                        legend_orientation="h",
                        legend=dict(x=.5, xanchor="center"),
                        hovermode="x",
                        margin=dict(l=0, r=0, t=0, b=0))


        fig.update_traces(hoverinfo="all", hovertemplate="Аргумент: %{x}<br>Функция: %{y}")
        fig.show()
        # self.get_hr_from_amulet()
        
    def get_hr_from_amulet(self, n=6):

        # x = np.arange(-n , n, 1) #- 1
        # y = np.arange(-n , n, 1) #- 1
        # z = np.arange(-n , n, 1) #- 1
        # X,Y,Z = np.meshgrid(x,y,z)
        # R_coords = np.array([X.flatten(),Y.flatten(),Z.flatten()]).T
        
        R_coords = self.R_coords
        orbitals = range(self.h_amulet.shape[0])
        # hr_amulet = []
        hr = []
        for R in tqdm(R_coords):#[[0.2,0], [0.3,0]]:
            hr.append( np.sum([ np.exp(-2*np.pi*1.j* np.dot(R, self.kpts_amulet[kpt_ind, 1:]) )*self.kpts_amulet[kpt_ind,0]*
                                self.h_amulet[ :, :, kpt_ind, 0] 
                                for kpt_ind in range(len(self.kpts_amulet)) ], axis=0)  )
        self.complex_hr = np.transpose(np.array(hr), (1, 2, 0))[..., np.newaxis]
        self.R_coords = R_coords
        
        #TODO just one spin now
        # self.complex_hr = np.hstack((hr_data[:,:5], 
        #     np.expand_dims(hr_data[:,6]*1.j + hr_data[:,5], axis=0).T, 
        #     np.expand_dims(R_weights_tmp, axis=0).T))
        # with open(self.directory + '/wannier/hr_mn' + self.name +'.pickle', 'wb') as f:
            # pickle.dump(self.complex_hr, f)


    def get_wannier_BS(self): 
        orbitals = range(self.nwa)
        band_str = []
        self.hks_bs = []
        for k in tqdm(self.k_path_qe):
            hk = np.sum( [np.exp(2*np.pi*1.j* np.dot(k, R) )*self.complex_hr[:, :, R_ind, 0] 
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

    def write_amulet(self, filename='hamilt_my'):
        nspin = 2
        nkp = len(self.kpoints_adj_serial)
        print(self.hks_spins.shape)
        # Open the file containing the Hamiltonian matrix elements
        with open(self.directory + 'wannier/' + filename + '.am', 'w') as outhamilt:
            outhamilt.write('&nspin \n')
            outhamilt.write(f'{nspin}\n \n')

            outhamilt.write('&nkp \n')
            outhamilt.write(f'{nkp}\n \n')

            outhamilt.write('&dim \n')
            outhamilt.write(f'{self.nwa}\n \n')

            outhamilt.write('&kpoints \n')
            for ik in range(nkp):
                kpvec = self.kpoints_adj_serial[ik, :]
                k2write = self.bcell@kpvec/(self.V**0.333)
                outhamilt.write(f"{1./nkp:.5f}  {k2write[0]:.5f}  {k2write[1]:.5f}  {k2write[2]:.5f} \n")
            
            outhamilt.write('\n&hamiltonian \n')
            for ispin in range(nspin):
                for ik in range(nkp):
                    for i in range(self.nwa):
                        for j in range(i, self.nwa):
                            # Read the real and imaginary parts from the file
                            outhamilt.write(f'{np.real(self.hks_spins[i, j, ik, ispin]):.5f} {np.imag(self.hks_spins[i, j, ik, ispin]):.5f}\n')
                            
 