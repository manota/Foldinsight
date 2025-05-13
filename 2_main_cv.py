#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyarrow.feather as feather
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from src.learnstructure import functions
import joblib
import itertools
import os
import argparse
import tempfile
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="To get Important region of Protein for functionality"
    )
    p.add_argument("--input-X-dir",   default="./MolecularFields",
                   help="Your Molecular Field Mapping directory")
    p.add_argument("--input-Y-path", default="./MolecularFields/y_values.csv",
                   help="Your functionality directory")
    p.add_argument("--output-dir",     default="./results",
                   help="Output directory for Molecular Fields")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    mapping_dir= args.input_X_dir
    functionality_path=args.input_Y_path
    output_dir=args.output_dir

    os.makedirs(output_dir,exist_ok=True)

    # mapping_dir= '/home/ota/PycharmProjects/sensor_241111/MolecularFields'
    # functionality_path='/home/ota/PycharmProjects/sensor_241111/MolecularFields/y_values.csv'
    # output_dir='./results'
    #
    # os.makedirs(output_dir,exist_ok=True)


    y_values=np.ravel(pd.read_csv(functionality_path,index_col=0))
    vdW=feather.read_feather(mapping_dir+'/vdW.feather')
    coulomb=feather.read_feather(mapping_dir+'/coulomb.feather')

    m_fields=pd.concat([vdW, coulomb], axis=1).values
    m_fields=m_fields.astype(np.float32) ### super minute width ###
    kf=KFold(n_splits=20,shuffle=True,random_state=0)

    st=time.time()
    results = joblib.Parallel(n_jobs=20)(joblib.delayed(functions.parallel_cross_val)(m_fields, y_values, train_index, test_index) for train_index, test_index in kf.split(m_fields, y_values))
    elapsed=time.time()-st
    print('elapsed time for cross validation =',elapsed/60,'min')

    cross_y_train=[each_result[0] for each_result in results]
    cross_y_train=list(itertools.chain.from_iterable(cross_y_train))
    cross_y_test=[each_result[1] for each_result in results]
    cross_y_test=list(itertools.chain.from_iterable(cross_y_test))
    cross_pred_y_train=[each_result[2] for each_result in results]
    cross_pred_y_train=list(itertools.chain.from_iterable(cross_pred_y_train))
    cross_pred_y_test=[each_result[3] for each_result in results]
    cross_pred_y_test=list(itertools.chain.from_iterable(cross_pred_y_test))
    cross_pls_comp=[each_result[4] for each_result in results]
    print(cross_pls_comp)

    #calc correlation coefficient
    par = np.polyfit(cross_y_test, cross_pred_y_test, 1, full=True)
    slope = par[0][0]
    intercept = par[0][1]

    variance = np.var(cross_pred_y_test)
    residuals = np.var([(slope * xx + intercept - yy) for xx, yy in zip(cross_y_test, cross_pred_y_test)])
    Rsqr = np.round(1 - residuals / variance, decimals=2)
    print('R = %0.2f'% np.sqrt(Rsqr))

    plt.style.use('seaborn-pastel')
    fig, ax = plt.subplots(figsize=(5, 5)) # figure, axesオブジェクトを作成
    ax.plot([np.min(cross_y_test),np.max(cross_y_test)],[np.min(cross_y_test),np.max(cross_y_test)],color='#757575',linestyle='dashed', alpha=0.5)
    ax.scatter(cross_y_test, cross_pred_y_test, alpha=0.6, edgecolors='black', linewidths=0.5, s=40,color='#DC143C')
    fig.gca().spines['top'].set_visible(False)
    fig.gca().spines['right'].set_visible(False)
    ax.set_ylabel('predicted', fontsize=14)
    ax.set_xlabel('observed', fontsize=14)
    ax.set_title('(R = %0.2f)'% np.sqrt(Rsqr))
    fig.tight_layout()
    plt.show()
    save_path=output_dir+'/observedVSpredicted.svg'
    fig.savefig(save_path,format='svg')
    plt.close()
