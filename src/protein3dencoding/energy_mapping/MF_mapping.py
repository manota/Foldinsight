import numpy as np
import pandas as pd
import glob
from numba import jit, prange
import time
from simtk.openmm import app,openmm
from simtk.openmm.app import PDBFile, ForceField
from pdbfixer import PDBFixer
import joblib
import os
from simtk import unit

class MoleFields():

    def __init__(self,gridbox,pdb_path,ff='amber99sbildn.xml'):
        self.gridbox=gridbox
        self.pdb_path=pdb_path
        self.pdb_xyz,self.pdb_atoms=self.__read_pdb_as_df(pdb_path)
        self.distance_mat=self.__calc_distance(self.pdb_xyz,self.gridbox)
        self.ff=ff

    def set_forcefield_from_out(self,ff):
        self.ff=ff

    def __read_pdb_as_df(self,pdb_path):
        with open(pdb_path, mode='r') as f:
            pdb_df = f.readlines()

        pdb_df = [line.split() for line in pdb_df if 'ATOM' in line]
        pdb_df = pd.DataFrame(pdb_df)
        pdb_xyz = pdb_df.iloc[:, 6:9]
        pdb_xyz.columns = ['x', 'y', 'z']
        pdb_xyz = pdb_xyz.astype(float)
        pdb_atoms = pdb_df.iloc[:, 2]
        return pdb_xyz, pdb_atoms

    def __calc_distance(self, pdb_xyz, gridbox):
        distance=calc_distance_numba(np.array(pdb_xyz),np.array(gridbox))
        return distance

    def __get_ff_params(self,pdb_path, ff):
        pdb = PDBFixer(pdb_path)

        force_field = ForceField(ff)
        system = force_field.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff)

        ff_params = np.zeros((system.getNumParticles(), 3))
        for i in range(system.getNumParticles()):
            forces = [system.getForce(j) for j in range(system.getNumForces())]
            nonbonded_force = [f for f in forces if isinstance(f, openmm.NonbondedForce)][0]
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(i)
            charge=charge.in_units_of(unit.elementary_charge)
            sigma=sigma.in_units_of(unit.angstrom)
            epsilon=epsilon.in_units_of(unit.kilocalorie_per_mole)
            ff_params[i, 0] = charge._value # charge
            ff_params[i, 1] = sigma._value  # sigma
            ff_params[i, 2] = epsilon._value  # epsilon

        return ff_params

    def __extract_virtual_sp3_C_ff(self,ff_params,pdb_atoms):
        virtual_sp3_C_ff = ff_params[np.where(pdb_atoms == 'CA')[0][0], :]
        virtual_sp3_C_ff[0] = 1
        return virtual_sp3_C_ff

    def __get_mean_LJ_params(self,ff_params,virtual_sp3_C_ff):
        mean_sigma = (virtual_sp3_C_ff[1] + ff_params[:, 1]) / 2
        mean_epsilon = np.sqrt(virtual_sp3_C_ff[2] * ff_params[:, 2])
        return mean_sigma,mean_epsilon

    def __set_forcefield_for_mf(self):
        self.ff_params=self.__get_ff_params(self.pdb_path,self.ff)
        self.virtual_sp3_C_ff=self.__extract_virtual_sp3_C_ff(self.ff_params,self.pdb_atoms)
        self.mean_sigma,self.mean_epsilon=self.__get_mean_LJ_params(self.ff_params, self.virtual_sp3_C_ff)
        self.charge=self.ff_params[:,0]

    def create_mfields(self):
        self.__set_forcefield_for_mf()
        vdW = calc_LJ_virtual_sp3(self.distance_mat, self.mean_sigma, self.mean_epsilon)
        coulomb = calc_Coulomb_virtual_sp3(self.distance_mat, self.charge)
        file_name = os.path.basename(self.pdb_path)
        return vdW.T,coulomb.T,file_name


@jit('f8(f8,f8,f8)', nopython=True)
def calc_LJ(distance, sigma, epsilon):
    LJ_part = (sigma / distance) ** 6
    LJ = 4 * epsilon * (LJ_part ** 2 - LJ_part)
    return LJ

@jit('f8(f8,f8)', nopython=True)
def calc_Coulomb(distance, charge):
    Coulomb = 332 * charge / distance
    return Coulomb


@jit('f8[:,:](f8[:,:],f8[:])', nopython=True, parallel=True)
def calc_Coulomb_virtual_sp3(distance_mat, charge):
    Coulomb_potentials = np.zeros((distance_mat.shape[0], 1))
    for i in range(distance_mat.shape[0]):
        Coulomb_atom_virtual_sp3 = 0.0
        for j in prange(charge.shape[0]):
            Coulomb_atom_virtual_sp3 += calc_Coulomb(distance_mat[i, j], charge[j])
        Coulomb_potentials[i, 0] = Coulomb_atom_virtual_sp3
    return Coulomb_potentials


@jit('f8[:,:](f8[:,:],f8[:],f8[:])', nopython=True, parallel=True)
def calc_LJ_virtual_sp3(distance_mat, mean_sigma, mean_epsilon):
    LJ_potentials = np.zeros((distance_mat.shape[0], 1))
    for i in range(distance_mat.shape[0]):
        LJ_atom_virtual_sp3 = 0.0
        for j in prange(mean_epsilon.shape[0]):
            LJ_atom_virtual_sp3 += calc_LJ(distance_mat[i, j], mean_sigma[j], mean_epsilon[j])
        LJ_potentials[i, 0] = LJ_atom_virtual_sp3
    return LJ_potentials


@jit('f8[:,:](f8[:,:],f8[:,:])',nopython=True,parallel=True)
def calc_distance_numba(pdb_xyz,gridbox):
    distance_mat=np.zeros((len(gridbox),len(pdb_xyz)))
    for i in range(len(gridbox)):
        for j in prange(len(pdb_xyz)):
            distance_mat[i,j]=np.sqrt(np.sum((pdb_xyz[j,:]-gridbox[i,:])**2))
    return distance_mat

