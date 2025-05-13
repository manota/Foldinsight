#%%
import pandas as pd
import time
import shutil
import subprocess
import glob
import os
import joblib
import numpy as np
import importlib.resources as pkg_resources

#%%
def running_pqr2(input_pdb_path):
    with pkg_resources.path('protein3dencoding.pdb2pqr','running_pdb2pqr.sh') as f:
        original_sh_path=str(f)
    #original_sh_path='/data2/ota/PycharmProjects/protein3dencoding/protein3dencoding/pdb2pqr/running_pdb2pqr.sh'
    rand_sh_path=original_sh_path[:-3]+str(np.random.random(1))+'.sh'
    pqr_file_path=input_pdb_path[:-4]+'.pqr'
    pdb_order = 'in_pdb=' + input_pdb_path + '\n'
    pqr_order = 'out_pqr=' + pqr_file_path + '\n'

    shutil.copyfile(original_sh_path, rand_sh_path)
    time.sleep(1)
    with open(rand_sh_path, mode='r+') as f:
        file_lines = f.readlines()
        file_lines[1] = pdb_order
        file_lines[2] = pqr_order
        f.seek(0)
        f.writelines(file_lines)
    time.sleep(1)
    command=['sh',rand_sh_path]
    subprocess.run(command)
    os.remove(rand_sh_path)

def running_pqr(input_pdb_path):
    pqr_file_path=input_pdb_path[:-4]+'.pqr'
    command=['pdb2pqr30', '--with-ph=7.0', '--ff=PEOEPB' ,input_pdb_path,pqr_file_path]
    subprocess.run(command)

def create_pqr_parsed_csv(pqr_path):
    output_path=pqr_path[:-4]+'.csv'
    with open(pqr_path, mode='r') as f:
        pqr_mat = f.readlines()

    pqr_mat = [line.split() for line in pqr_mat if 'ATOM' in line]
    pqr_mat = pd.DataFrame(pqr_mat)

    pqr_mat.to_csv(output_path)