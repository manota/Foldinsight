import pandas as pd
import numpy as np
import joblib
import subprocess
import glob
import shutil
import os
import time


def optimize_chimera(input_path,chimera_path='/home/ota/.local/UCSF-Chimera64-1.18/bin/chimera'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    chimera_script=current_dir+'/chimera_script.py'

    rand_chimera_script=chimera_script[:-3]+str(np.random.random(1))+'.py'
    shutil.copyfile(chimera_script,rand_chimera_script)
    output_path=input_path[:-4]+'_opt.pdb'
    with open(rand_chimera_script,mode='r+') as f:
        strings=f.readlines()
        strings[3]='input_file_path='+"'"+input_path+"'"+'\n'
        strings[4]='output_file_path='+"'"+output_path+"'"+'\n'
        f.seek(0)
        f.writelines(strings)
    command=[chimera_path, '--script', rand_chimera_script, '--nogui']
    st=time.time()
    subprocess.run(command)
    print('elapsed time=',(time.time()-st)/60,'min')
    time.sleep(2)
    os.remove(rand_chimera_script)

