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

def save_pdb_refined(pdb_path,save_dir):
    pdbfixer = PDBFixer(pdb_path)
    pdbfixer.findMissingResidues()
    pdbfixer.findMissingAtoms()
    pdbfixer.addMissingAtoms()
    pdbfixer.addMissingHydrogens(7.0)

    os.makedirs(save_dir,exist_ok=True)
    output_path=save_dir+'/'+pdb_path.split('/')[-1][:-4]+'_refined.pdb'

    with open(output_path, 'w') as f:
        PDBFile.writeFile(pdbfixer.topology, pdbfixer.positions, f)

def read_pdb_as_xyz(pdb_path):
    with open(pdb_path, mode='r') as f:
        pdb_df = f.readlines()

    pdb_df = [line.split() for line in pdb_df if 'ATOM' in line]
    pdb_xyz = pd.DataFrame(pdb_df).iloc[:,6:9]
    pdb_xyz.columns=['x','y','z']
    pdb_xyz=pdb_xyz.astype(float)
    return pdb_xyz

def calc_gridsize(salined_pdb_pathes):
    salined_pdb_s = [read_pdb_as_xyz(salined_pdb_path) for salined_pdb_path in salined_pdb_pathes]
    x_min = np.min([np.min(salined_pdb['x']) for salined_pdb in salined_pdb_s])
    x_max = np.max([np.max(salined_pdb['x']) for salined_pdb in salined_pdb_s])
    y_min = np.min([np.min(salined_pdb['y']) for salined_pdb in salined_pdb_s])
    y_max = np.max([np.max(salined_pdb['y']) for salined_pdb in salined_pdb_s])
    z_min = np.min([np.min(salined_pdb['z']) for salined_pdb in salined_pdb_s])
    z_max = np.max([np.max(salined_pdb['z']) for salined_pdb in salined_pdb_s])

    gridsize = np.zeros((3, 2))
    gridsize[0, 0] = x_min
    gridsize[0, 1] = x_max
    gridsize[1, 0] = y_min
    gridsize[1, 1] = y_max
    gridsize[2, 0] = z_min
    gridsize[2, 1] = z_max

    width = gridsize[:, 1] - gridsize[:, 0]
    gridsize = np.vstack((np.floor(gridsize[:, 0] - width * 0.1), np.ceil(gridsize[:, 1] + width * 0.1)))
    # gridsize = np.vstack((np.floor(gridsize[:, 0]), np.ceil(gridsize[:, 1])))
    gridsize = gridsize.T
    return gridsize

def create_gridbox(salined_pdb_pathes,step_size=2):
    gridsize=calc_gridsize(salined_pdb_pathes)
    x = np.arange(gridsize[0, 0], gridsize[0, 1], step=step_size)
    y = np.arange(gridsize[1, 0], gridsize[1, 1], step=step_size)
    z = np.arange(gridsize[2, 0], gridsize[2, 1], step=step_size)
    xx, yy, zz = np.meshgrid(x, y, z)
    gridbox = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    return gridbox