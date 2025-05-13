#%%
import numpy as np
import pandas as pd
import pyarrow.feather as feather
from src.learnstructure import preprocess
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import copy
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from src.learnstructure import contour_map,preprocess,functions
from scipy.spatial import ConvexHull
import subprocess
import os
import time
import argparse
import tempfile
from pathlib import Path

#%%
property_='functionality'

def parse_args():
    p = argparse.ArgumentParser(
        description="To get Important region of Protein for functionality"
    )
    p.add_argument("--input-X-dir",   default="./MolecularFields",
                   help="Your Molecular Field Mapping directory")
    p.add_argument("--input-Y-path", default="./MolecularFields/y_values.csv",
                   help="Your functionality directory")
    p.add_argument("--representative-pdb-path",   default="./01structure/p_0000.pdb",
                   help="Your representative PDB file path")
    p.add_argument("--output-dir",     default="./results",
                   help="Output directory for Molecular Fields")
    home_path=Path.home()
    matches = list(home_path.glob('*/UCSF*/bin/chimera'))
    matches=[str(p.resolve()) for p in matches]
    p.add_argument("--chimera-path",default=matches[0],
                   help="Path to chimera executable like '$HOME/.local/UCSF-Chimera64-1.19/bin/chimera'")
    p.add_argument("--leave-temp-dir", action="store_true",
                   help="Leave temporary directory")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    mapping_dir= args.input_X_dir
    functionality_path=args.input_Y_path
    representative_pdb_path=args.representative_pdb_path
    output_dir=args.output_dir
    chimera_path=args.chimera_path

    os.makedirs(output_dir,exist_ok=True)

    model_path = tempfile.mkdtemp(prefix="Foldinsight_")

    y_values=np.ravel(pd.read_csv(functionality_path,index_col=0))
    vdW=feather.read_feather(mapping_dir+'/vdW.feather')
    coulomb=feather.read_feather(mapping_dir+'/coulomb.feather')
    coord_df=pd.read_csv(mapping_dir+'/gridbox.csv',index_col=0)

    m_fields=pd.concat([vdW, coulomb], axis=1).values
    m_fields=m_fields.astype(np.float32) ### super minute width ###
    #%%
    result=functions.general_process(m_fields,y_values)
    pls_model=result[0]
    vdW_feature=result[2].feature
    selected_features=result[4]

    vdW_left_feature_pos = np.where(vdW_feature == 1)[0]
    selected_vdW_features = selected_features[selected_features < len(vdW_left_feature_pos)]
    original_vdW_positions = vdW_left_feature_pos[selected_vdW_features]
    pls_vdW_coef=pls_model.coef_.flatten()[:len(original_vdW_positions)]
    sd_vdW=np.std(vdW.iloc[:,original_vdW_positions],axis=0)

    coord_vdW_df=coord_df.iloc[original_vdW_positions,:]
    coord_vdW_df.loc[:,'vdW_coef']=pls_vdW_coef
    coord_vdW_df.loc[:,'vdW_sd']=sd_vdW
    coord_vdW_df.reset_index(inplace=True,drop=True)

    coord_vdW_df.columns=['vdW_coord_x','vdW_coord_y','vdW_coord_z','vdW_coef','vdW_sd']
    coord_vdW_df.to_csv(model_path+'/'+property_+'_vdW_coord.csv')
    coord_df.to_csv(model_path+'/original_coord.csv')

    #%%
    vdW_coord_coef_sd_df=pd.read_csv(model_path+'/'+property_+'_vdW_coord.csv',index_col=0)
    original_coord_df=pd.read_csv(model_path+'/original_coord.csv',index_col=0)
    coord=original_coord_df.values

    b_coef = vdW_coord_coef_sd_df.iloc[:,3].values
    b_SD=vdW_coord_coef_sd_df.iloc[:,4].values
    b_coef_SD= b_coef * b_SD

    vdW_coord= vdW_coord_coef_sd_df.iloc[:, :3].values
    delaunay_pair=contour_map.get_delaunay_pair(vdW_coord)
    average_distance=np.mean([np.sqrt(np.sum((vdW_coord[vertexes[0],:]-vdW_coord[vertexes[1],:])**2)) for vertexes in delaunay_pair])
    coord_and_v=np.hstack([vdW_coord, b_coef_SD.reshape(-1, 1)])

    max_min_list=[99,1]
    max_min_names_list=['max','min']
    max_min_colors_list=['green','yellow']
    total_bild_file_names_list=[]
    for i,q_value in enumerate(max_min_list):
        max_min_name=max_min_names_list[i]
        max_min_color=max_min_colors_list[i]

        v=np.percentile(b_coef_SD, q_value)

        inter_equinox_coord=contour_map.get_inter_equinox_based_delauny(coord_and_v, v,1000)
        limit_distance=2
        clusters=contour_map.get_tree_cluster(inter_equinox_coord,limit_distance)
        clusters=[cluster for cluster in clusters if len(cluster)>3]
        cluster_number=np.zeros(inter_equinox_coord.shape[0])
        cluster_number[:]=-1
        for i,cluster in enumerate(clusters):
            for num in cluster:
                cluster_number[num]=i

        unique_cluster=np.unique(cluster_number)[np.unique(cluster_number)>-1]
        file_names=[]
        for i in unique_cluster:
            try:
                pos=np.where(cluster_number==i)[0]
                cluster_inter_equinox= inter_equinox_coord[pos, :]
                file_name = model_path+'/'+property_ +'_cluster_'+max_min_name + str(int(i)) + '.bild'
                contour_map.make_bild_file(cluster_inter_equinox, file_name,max_min_color)
                hull = ConvexHull(cluster_inter_equinox)
                temp_cluster_inter_equinox = cluster_inter_equinox[hull.vertices, :]
            except:
                continue
            file_names.append(file_name)

        total_bild_file=model_path+'/'+property_ +'_cluster_'+max_min_name+'.bild'
        total_lines=[]
        for file_name in file_names:
            with open(file_name,mode='r') as f:
                lines=f.readlines()
            total_lines=total_lines+lines
            os.remove(file_name)
        with open(total_bild_file,mode='w+') as f:
            f.writelines(total_lines)

        total_bild_file_names_list.append(copy.copy(total_bild_file))

    chimera_cmd_set_file_path='./src/chimera_run/chimera_cmd_set.cmd'
    with open(chimera_cmd_set_file_path,mode='r') as f:
        lines=f.readlines()

    color_list=['#DC143C','#1E90FF','#8A2BE2']
    sp_color=color_list[0]

    lines=[line.replace('\n','') for line in lines]
    lines=[line.replace('representative_pdb_path',representative_pdb_path) for line in lines]
    lines=[line.replace('cluster_max_bild_path',total_bild_file_names_list[0]) for line in lines]
    lines=[line.replace('cluster_min_bild_path',total_bild_file_names_list[1]) for line in lines]
    lines=[line.replace('result_path',output_dir) for line in lines]
    lines=[line.replace('XXXXXXX',sp_color) for line in lines]
    runcommand_lines=[f'runCommand("{line}")'+'\n' for line in lines]
    runcommand_lines=['from chimera import runCommand'+'\n']+runcommand_lines

    chimera_cmd_set_py_path='./src/chimera_run/chimera_cmd_set.py'
    with open(chimera_cmd_set_py_path,mode='w') as f:
        f.writelines(runcommand_lines)

    command=[chimera_path,'--script',chimera_cmd_set_py_path,'--bgopacity']
    subprocess.run(command)
