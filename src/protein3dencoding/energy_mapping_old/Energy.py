import os
from glob import glob
import subprocess
import shutil
import time
import numpy as np
import pandas as pd
import math
from numba import jit, prange


def generate_enegy_map_features(pqr_parsed_pathes,step_size=2.0):
    gridsize=create_gridbox(pqr_parsed_pathes)
    temp = Energy_in_grid(gridsize, step_size)
    sample_size=len(pqr_parsed_pathes)

    coulom_all_protein = np.zeros((sample_size, temp.measure_distance().shape[1]))
    vdW_all_protein = np.zeros((sample_size, temp.measure_distance().shape[1]))

    file_index = []
    for i, filepath in enumerate(pqr_parsed_pathes):
        st = time.time()
        atom_pqr = pd.read_csv(filepath, index_col=0)
        grid_value = Energy_in_grid(gridsize, step_size, atom_pqr)
        grid_value.set_atom_name()
        coulom_all_protein[i, :] = grid_value.coulombic_fast()
        vdW_all_protein[i, :] = grid_value.van_der_waals_fast()
        file_index.append(filepath[:-4])
        print('elapsed per one protein = ', time.time() - st)

    print('output=','gridsize,file_index,vdw,coulomb')
    return temp.grid,file_index,vdW_all_protein,coulom_all_protein



def create_gridbox(pqr_parsed_pathes):
    pqr_parsed_files = [pd.read_csv(pqr_parsed_file, index_col=0) for pqr_parsed_file in pqr_parsed_pathes]
    x_min = np.min([np.min(pqr_parsed_file.iloc[:, 5]) for pqr_parsed_file in pqr_parsed_files])
    x_max = np.max([np.max(pqr_parsed_file.iloc[:, 5]) for pqr_parsed_file in pqr_parsed_files])
    y_min = np.min([np.min(pqr_parsed_file.iloc[:, 6]) for pqr_parsed_file in pqr_parsed_files])
    y_max = np.max([np.max(pqr_parsed_file.iloc[:, 6]) for pqr_parsed_file in pqr_parsed_files])
    z_min = np.min([np.min(pqr_parsed_file.iloc[:, 7]) for pqr_parsed_file in pqr_parsed_files])
    z_max = np.max([np.max(pqr_parsed_file.iloc[:, 7]) for pqr_parsed_file in pqr_parsed_files])

    gridsize = np.zeros((3, 2))
    gridsize[0, 0] = x_min
    gridsize[0, 1] = x_max
    gridsize[1, 0] = y_min
    gridsize[1, 1] = y_max
    gridsize[2, 0] = z_min
    gridsize[2, 1] = z_max

    width = gridsize[:, 1] - gridsize[:, 0]
    gridsize = np.vstack((np.floor(gridsize[:, 0] - width * 0.1), np.ceil(gridsize[:, 1] + width * 0.1)))
    gridsize = gridsize.T
    return gridsize


class Energy_in_grid():

    '''
    coulom_const = 8.9876 * 10 ** 9
    correct_unit=10**8*6.022*10**23*(1.6021*10**(-19))**2/4.184*10**(-3)
    '''

    default=pd.DataFrame(np.zeros((100,10)))

    def __init__(self, gridsize,step_size,parsed_pqr=default):
        step_size=step_size
        self.atom_pqr = parsed_pqr
        x = np.arange(gridsize[0, 0], gridsize[0, 1], step=step_size)
        y = np.arange(gridsize[1, 0], gridsize[1, 1], step=step_size)
        z = np.arange(gridsize[2, 0], gridsize[2, 1], step=step_size)
        xx, yy, zz = np.meshgrid(x, y, z)
        self.grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
        atom_coord = self.atom_pqr.iloc[:, 5:8].values
        self.distance=self.fast_measure_distance(atom_coord,self.grid)
        self.atom_name=None

    def set_atom_name(self):
        self.atom_name = [atom[0] for atom in self.atom_pqr.iloc[:, 2]]


    def measure_distance(self):
        atom_coord = self.atom_pqr.iloc[:, 5:8].values
        distance = np.zeros((atom_coord.shape[0], self.grid.shape[0]))
        for i, mesh_coord in enumerate(self.grid):
            r2_distance = np.sum(np.square(mesh_coord - atom_coord), axis=1)
            r_distance = np.sqrt(r2_distance)
            distance[:, i] = r_distance
        return distance

    @staticmethod
    @jit('f8[:,:](f8[:,:],f8[:,:])', nopython=True, parallel=True)
    def fast_measure_distance(atom_coord, grid):
        distance = np.zeros((atom_coord.shape[0], grid.shape[0]))
        for i in prange(grid.shape[0]):
            mesh_coord = grid[i, :]
            r2_distance = np.sum(np.square(mesh_coord - atom_coord), axis=1)
            r_distance = np.sqrt(r2_distance)
            distance[:, i] = r_distance
        return distance


    def coulombic(self):
        correct_coulom_const = 3.3202
        charge = self.atom_pqr.iloc[:, 9].values
        rep_charge = np.array([charge] * self.distance.shape[1]).T * correct_coulom_const
        potential = np.sum(rep_charge / self.distance, axis=0)
        return potential


    def coulombic_fast(self):
        correct_coulom_const = 3.3202
        charge = self.atom_pqr.iloc[:, 9].values
        p_coulomb_potential = np.zeros(self.distance.shape[1])
        p_coulomb_potential = self.fast_function_for_coulomb(charge,self.distance,p_coulomb_potential)
        return p_coulomb_potential


    @staticmethod
    @jit('f8[:](f8[:],f8[:,:],f8[:])', nopython=True,parallel=True)
    def fast_function_for_coulomb(charge,distance,p_coulomb_potential):
        correct_coulom_const = 3.3202
        for col_coord in range(distance.shape[1]):
            coulomb=0.0
            for row in prange(charge.shape[0]):
                each_charge=charge[row]
                coulomb_potential = correct_coulom_const*(each_charge / distance[row,col_coord])
                coulomb+=coulomb_potential
            p_coulomb_potential[col_coord]=coulomb
        return p_coulomb_potential


    def van_der_waals(self):
        distance = self.measure_distance()
        atom_name = pd.Series([atom[0] for atom in self.atom_pqr.iloc[:, 2]])
        #atom_name=self.atom_pqr.iloc[:, 2]
        p_vdW_potenial=np.zeros(distance.shape)
        for i,atom in enumerate(atom_name):
            p_vdW_potenial[i,:] = np.apply_along_axis(self.__buckingham_formula, 0, distance[i,:],atom)
        vdW_potenial = np.sum(p_vdW_potenial, axis=0)
        return (vdW_potenial)


    def van_der_waals_fast(self):
        buckingham_dict = {"H": [1.20, 0.067], "C": [1.70, 0.107], "N": [1.55, 0.100], "O": [1.52, 0.111],"S":[1.80,0.183]}
        buckingham_params = np.array([buckingham_dict[atom] for atom in self.atom_name])
        #buckingham_params=np.hstack([buckingham_params,distance])
        p_vdW_potenial = np.zeros(self.distance.shape[1])
        p_vdW_potenial = self.fast_function(buckingham_params,self.distance,p_vdW_potenial)
        return p_vdW_potenial


    # buckingham_params's 1st col=radius, 2nd col=interaction
    @staticmethod
    @jit('f8[:](f8[:,:],f8[:,:],f8[:])', nopython=True,parallel=True)
    def fast_function(buckingham_params,distance,p_vdW_potenial):
        vdWr_C = 1.70
        for col_coord in range(distance.shape[1]):
            vdW=0.0
            for row in prange(buckingham_params.shape[0]):
                buckparams=buckingham_params[row,:]
                temp = (vdWr_C + buckparams[0]) / distance[row,col_coord]
                vdW_potential = buckparams[1] * (-2.25 * (temp) ** 6 + 8.28 * (10 ** 5) * np.exp(-1 / temp / 0.0736))
                vdW+=vdW_potential
            p_vdW_potenial[col_coord]=vdW
        return p_vdW_potenial


    def __buckingham_formula(self, distance,atom):
        vdWr_C = 1.70
        buckingham_dict = {"H": [1.20, 0.067], "C": [1.70, 0.107], "N": [1.55, 0.100], "O": [1.52, 0.111],"S":[1.80,0.183]}
        buckparams = buckingham_dict[atom[0]]
        temp = (vdWr_C + buckparams[0]) / distance
        vdW_potential = buckparams[1] * (-2.25 * (temp) ** 6 + 8.28*(10 ** 5) * np.exp(-1/temp / 0.0736))
        return (vdW_potential)

