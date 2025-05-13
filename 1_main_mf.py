import glob
import os
import json
import numpy as np
import shutil
import joblib
import time

import pandas as pd
from src.protein3dencoding.salign import sc_align
from src.protein3dencoding.pdb2pqr import running_pqr
from src.protein3dencoding.pdb2pqr import create_pqr_parsed_csv
from src.protein3dencoding.energy_mapping_old import generate_enegy_map_features

import pyarrow.feather as feather
import argparse
import tempfile

def parse_args():
    p = argparse.ArgumentParser(
        description="To get Molecular Field features from PDB files"
    )
    p.add_argument("--input-dir",   default="./01structure",
                   help="Your PDB files directory")
    p.add_argument("--output-dir",     default="./MolecularFields",
                   help="Output directory for Molecular Fields")
    p.add_argument("--step-size", type=float, default=1.0,
                   help="Lattice step size for grid box (Å)")
    p.add_argument("--leave-temp-dir", action="store_true",
                   help="Leave temporary directory")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_dir= args.input_dir
    output_dir=args.output_dir
    step_size=args.step_size

    tmpdir = tempfile.mkdtemp(prefix="MFinter_")

    ### 2. Structural alignment ###
    pdb_pathes_list=sorted(glob.glob(input_dir + '/*.pdb'))
    template_pdb=pdb_pathes_list[0]
    joblib.Parallel(n_jobs=-1)(joblib.delayed(sc_align)(template_pdb,target_pdb) for target_pdb in pdb_pathes_list[1:])
    time.sleep(5)
    unrelaxed_saligned_pdb_pathes_list=glob.glob(input_dir + '/*_fit.pdb')

    saligned_dir=tmpdir+'/02saligned_structure'
    os.makedirs(saligned_dir,exist_ok=True)
    joblib.Parallel(n_jobs=-1)(joblib.delayed(shutil.move)(saligned_pdb,saligned_dir+'/'+os.path.basename(saligned_pdb)) for saligned_pdb in unrelaxed_saligned_pdb_pathes_list)
    time.sleep(5)
    ### 2. Structural alignment ###

    ### 3. PDB2PQR ###
    sc_align_pdb_pathes_list=sorted(glob.glob(saligned_dir + '/*.pdb'))

    pqr_dir=tmpdir+'/03pqr'
    os.makedirs(pqr_dir,exist_ok=True)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(running_pqr)(pdb_path) for pdb_path in sc_align_pdb_pathes_list)
    time.sleep(5)
    joblib.Parallel(n_jobs=-1)(joblib.delayed(shutil.move)(pqr_file_path,pqr_dir+'/'+os.path.basename(pqr_file_path)) for pqr_file_path in glob.glob(saligned_dir+'/*.pqr'))
    time.sleep(5)
    ### 3. PDB2PQR ###

    ### 4. PQR2CSV ###
    pqr_pathes_list=sorted(glob.glob(pqr_dir + '/*.pqr'))

    csv_dir=tmpdir+'/04csv'
    os.makedirs(csv_dir,exist_ok=True)

    joblib.Parallel(n_jobs=-1)(joblib.delayed(create_pqr_parsed_csv)(pqr_file_path) for pqr_file_path in pqr_pathes_list)
    time.sleep(5)
    joblib.Parallel(n_jobs=-1)(joblib.delayed(shutil.move)(csv_file_path,csv_dir+'/'+os.path.basename(csv_file_path)) for csv_file_path in glob.glob(pqr_dir+'/*.csv'))
    time.sleep(5)
    ### 4. PQR2CSV ###


    ### 5. mapping ###
    mapping_dir=tmpdir+'/05mapping'
    os.makedirs(mapping_dir,exist_ok=True)

    pqr_csv_pathes_list=sorted(glob.glob(csv_dir + '/*.csv'))
    gridbox,file_indices,vdW_all_protein,coulom_all_protein=generate_enegy_map_features(pqr_csv_pathes_list, step_size=step_size)
    gridbox_df=pd.DataFrame(gridbox)
    vdW_df=pd.DataFrame(vdW_all_protein)
    coulomb_df=pd.DataFrame(coulom_all_protein)
    file_name_df=pd.DataFrame(file_indices).astype(str)
    gridbox_df.to_csv(mapping_dir+'/gridbox.csv')
    vdW_df.to_csv(mapping_dir+'/vdW.csv')
    coulomb_df.to_csv(mapping_dir+'/coulomb.csv')
    file_name_df.to_csv(mapping_dir+'/file_name.csv')

    feather.write_feather(vdW_df,mapping_dir+'/vdW.feather')
    feather.write_feather(coulomb_df,mapping_dir+'/coulomb.feather')
    time.sleep(5)
    ### 5. mapping ###

    os.makedirs(output_dir,exist_ok=True)
    joblib.Parallel(n_jobs=-1)(joblib.delayed(shutil.move)(mapping_file_path,output_dir+'/'+os.path.basename(mapping_file_path)) for mapping_file_path in glob.glob(mapping_dir+'/*'))
    time.sleep(5)
    if args.leave_temp_dir==False:
        shutil.rmtree(tmpdir)
    else:
        print("Saving directory is",tmpdir)

