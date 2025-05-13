import time
from src.protein3dencoding.optimize_structure import chimera_running
import argparse
import os
import glob
from joblib import Parallel, delayed
import shutil
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="To get Molecular Field features from PDB files"
    )
    p.add_argument("--input-dir",   default="./00ori_structure",
                   help="Your PDB files directory")
    p.add_argument("--output-dir",     default="./01structure",
                   help="Output directory")
    home_path=Path.home()
    matches = list(home_path.glob('*/UCSF*/bin/chimera'))
    matches=[str(p.resolve()) for p in matches]
    p.add_argument("--chimera-path",default=matches[0],
                   help="Path to chimera executable like '$HOME/.local/UCSF-Chimera64-1.19/bin/chimera'")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    input_dir= args.input_dir
    output_dir=args.output_dir
    chimera_path=args.chimera_path
    os.makedirs(output_dir,exist_ok=True)

    pdb_file_pathes_list=sorted(glob.glob(input_dir + '/*.pdb'))
    Parallel(n_jobs=-1)(delayed(chimera_running.optimize_chimera)(pdb_file_path,chimera_path) for pdb_file_path in pdb_file_pathes_list)
    time.sleep(10)
    opt_pdb_file_pathes_list=sorted(glob.glob(input_dir + '/*_opt.pdb'))
    Parallel(n_jobs=-1)(delayed(shutil.move)(pdb_file_path,output_dir+'/'+os.path.basename(pdb_file_path)) for pdb_file_path in opt_pdb_file_pathes_list)


