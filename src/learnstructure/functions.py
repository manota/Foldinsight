import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pyarrow.feather as feather
from src.learnstructure import preprocess
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import joblib
import copy
import GPy


def parallel_cross_val(m_fields,y_values,train_index,test_index):
    m_fields_train = m_fields[train_index, :]
    m_fields_test = m_fields[test_index, :]
    y_train = y_values[train_index]
    y_test = y_values[test_index]

    y_train_mean=copy.copy(np.mean(y_train))
    y_train_sd=copy.copy(np.std(y_train))
    y_train=(y_train-y_train_mean)/y_train_sd
    y_test=(y_test-y_train_mean)/y_train_sd


    vdW_train, coulomb_train = np.split(m_fields_train, 2, axis=1)
    vdW_test, coulomb_test = np.split(m_fields_test, 2, axis=1)

    ### van der waals potential (START) ###
    vdW_train_obj = preprocess.Preprocessing(vdW_train, y_train)
    vdW_train_obj.detect_small_sd_pos()
    vdW_train_obj.detect_large_skew_pos()
    #vdW_train_obj.determine_cutoff()
    #vdW_train_obj.determine_cutoff(upper_percent=99, lower_percent=1)
    vdW_train_obj.max_cutoff=100
    vdW_train_obj.min_cutoff=-100
    vdW_train_obj.overwrite_too_big_energy()
    vdW_train_obj.overwrite_too_small_energy()
    vdW_train_obj.update_statistics()
    vdW_train_obj.detect_small_sd_pos()
    vdW_train_obj.detect_large_skew_pos()
    centered_vdW_train = vdW_train_obj.output_centering()
    normed_vdW_train = vdW_train_obj.output_normalize()
    vdW_train_params = vdW_train_obj.output_train_params()
    vdW_feature = copy.copy(vdW_train_obj.feature)

    vdW_test_obj = preprocess.Preprocessing(vdW_test, y_test)
    vdW_test_obj.set_train_params(vdW_train_params)
    vdW_test_obj.use_train_params()
    centered_vdW_test = vdW_test_obj.output_centering()
    normed_vdW_test = vdW_test_obj.output_normalize()
    ### van der waals potential (END) ###

    ### coulomb potential (START) ###
    coulomb_train_obj = preprocess.Preprocessing(coulomb_train, y_train)
    #coulomb_train_obj.set_features_manual(vdW_feature)
    coulomb_train_obj.detect_small_sd_pos()
    coulomb_train_obj.detect_large_skew_pos()
    # coulomb_train_obj.determine_cutoff(upper_percent=100, lower_percent=0)
    # coulomb_train_obj.overwrite_too_big_energy()
    # coulomb_train_obj.overwrite_too_small_energy()
    # coulomb_train_obj.detect_small_sd_pos()
    # coulomb_train_obj.detect_large_skew_pos()

    normed_coulomb_train = coulomb_train_obj.output_normalize()
    coulomb_train_params = coulomb_train_obj.output_train_params()

    coulomb_test_obj = preprocess.Preprocessing(coulomb_test, y_test)
    coulomb_test_obj.set_train_params(coulomb_train_params)
    #coulomb_test_obj.use_train_params()
    centered_coulomb_test = coulomb_test_obj.output_centering()
    normed_coulomb_test = coulomb_test_obj.output_normalize()

    estimator = SVR(kernel="linear")
    coulomb_RFE=RFE(estimator, n_features_to_select=100000, step=2500)
    coulomb_RFE=coulomb_RFE.fit(normed_coulomb_train , y_train)
    coulomb_selected_features = np.where(coulomb_RFE.support_ == True)[0]
    normed_coulomb_train = normed_coulomb_train[:, coulomb_selected_features]
    normed_coulomb_test = normed_coulomb_test[:, coulomb_selected_features]

    ### coulomb potential (END) ###

    # %%
    ### molecular fields (START) ###
    normed_m_fields_train = np.hstack([normed_vdW_train, normed_coulomb_train])
    normed_m_fields_test = np.hstack([normed_vdW_test, normed_coulomb_test])
    ### molecular fields (END) ###

    ## feature elimination ##
    st = time.time()
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=2000, step=30)
    selector = selector.fit(normed_m_fields_train, y_train)
    elapsed = (time.time() - st) / 60 / 60
    print('elapsed time in feature elimination =', elapsed, 'hour')
    selected_features = np.where(selector.support_ == True)[0]
    selected_normed_m_fields_train = normed_m_fields_train[:, selected_features]
    selected_normed_m_fields_test = normed_m_fields_test[:, selected_features]

    cross_val_values=np.zeros((298,20))
    inner_kf=KFold(n_splits=20,shuffle=True,random_state=0)
    j=0
    for inner_train_index,inner_test_index in inner_kf.split(selected_normed_m_fields_train,y_train):
        inner_m_train=selected_normed_m_fields_train[inner_train_index,:]
        inner_m_test=selected_normed_m_fields_train[inner_test_index,:]
        inner_y_train=y_train[inner_train_index]
        inner_y_test=y_train[inner_test_index]
        for k,pls_dim in enumerate(range(2,300)):
            model=PLSRegression(n_components=pls_dim)
            model.fit(inner_m_train,inner_y_train)
            pred_v=model.predict(inner_m_test)
            r2=r2_score(inner_y_test,pred_v)
            cross_val_values[k,j]=r2
        j=j+1
    dim_scores = np.apply_along_axis(np.mean, 1, cross_val_values)
    pls_dim = 2 + np.where(dim_scores == np.max(dim_scores))[0][0]

    pls_model = PLSRegression(n_components=int(pls_dim))
    pls_model.fit(selected_normed_m_fields_train, y_train)

    pred_y_train = pls_model.predict(selected_normed_m_fields_train)
    pred_y_test = pls_model.predict(selected_normed_m_fields_test)

    result = [y_train, y_test, pred_y_train, pred_y_test,pls_dim]
    return result



def general_process(m_fields,y_values):
    vdW, coulomb = np.split(m_fields, 2, axis=1)


    ### van der waals potential (START) ###
    vdW_obj = preprocess.Preprocessing(vdW, y_values)
    vdW_obj.detect_small_sd_pos()
    vdW_obj.detect_large_skew_pos()
    vdW_obj.max_cutoff=100
    vdW_obj.min_cutoff=-100
    vdW_obj.overwrite_too_big_energy()
    vdW_obj.overwrite_too_small_energy()
    vdW_obj.update_statistics()
    vdW_obj.detect_small_sd_pos()
    vdW_obj.detect_large_skew_pos()
    normed_vdW = vdW_obj.output_normalize()
    ### van der waals potential (END) ###

    ### coulomb potential (START) ###
    coulomb_obj = preprocess.Preprocessing(coulomb, y_values)
    coulomb_obj.detect_small_sd_pos()
    coulomb_obj.detect_large_skew_pos()

    normed_coulomb = coulomb_obj.output_normalize()

    estimator = SVR(kernel="linear")
    coulomb_RFE=RFE(estimator, n_features_to_select=100000, step=2500)
    coulomb_RFE=coulomb_RFE.fit(normed_coulomb , y_values)
    coulomb_selected_features = np.where(coulomb_RFE.support_ == True)[0]
    normed_coulomb = normed_coulomb[:, coulomb_selected_features]
    ### coulomb potential (END) ###

    ### molecular fields (START) ###
    normed_m_fields = np.hstack([normed_vdW, normed_coulomb])
    ### molecular fields (END) ###

    ## feature elimination ##
    st = time.time()
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=2000, step=30)
    selector = selector.fit(normed_m_fields, y_values)
    elapsed = (time.time() - st) / 60 / 60
    print('elapsed time in feature elimination =', elapsed, 'hour')
    selected_features = np.where(selector.support_ == True)[0]
    selected_normed_m_fields = normed_m_fields[:, selected_features]

    cross_val_values=np.zeros((298,20))
    inner_kf=KFold(n_splits=20,shuffle=True,random_state=0)
    j=0
    for inner_train_index,inner_test_index in inner_kf.split(selected_normed_m_fields,y_values):
        inner_m_train=selected_normed_m_fields[inner_train_index,:]
        inner_m_test=selected_normed_m_fields[inner_test_index,:]
        inner_y_train=y_values[inner_train_index]
        inner_y_test=y_values[inner_test_index]
        for k,pls_dim in enumerate(range(2,300)):
            model=PLSRegression(n_components=pls_dim)
            model.fit(inner_m_train,inner_y_train)
            pred_v=model.predict(inner_m_test)
            r2=r2_score(inner_y_test,pred_v)
            cross_val_values[k,j]=r2
        j=j+1
    dim_scores = np.apply_along_axis(np.mean, 1, cross_val_values)
    pls_dim = 2 + np.where(dim_scores == np.max(dim_scores))[0][0]

    pls_model = PLSRegression(n_components=int(pls_dim))
    pls_model.fit(selected_normed_m_fields, y_values)

    result = [pls_model,pls_dim,vdW_obj,coulomb_obj,selected_features]
    return result
