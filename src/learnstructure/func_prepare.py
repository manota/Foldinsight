import pandas as pd
import numpy as np


def format_data(property_, df,vdW,coulomb):

    train_pos = np.where((np.isnan(df[property_]) == False) & (df['gen'] != 10))[0]
    test_pos = np.where((np.isnan(df[property_]) == False) & (df['gen'] == 10))[0]

    # remove ChR_29_10 & ChR_30_10 for kinetics and spectra because currents too low for accurate measurements
    if property_ == 'green_norm' or property_ == 'kinetics_off':
        test_pos=np.where((np.isnan(df[property_]) == False) & (df['gen'] == 10) & (df['chimera'] != 'ChR_29_10') & (df['chimera'] != 'ChR_30_10'))[0]

    ### training part ###
    y_train=df[property_].values[train_pos]
    log_y_train= np.log(y_train)
    mean_log_y_train=np.mean(log_y_train)
    std_log_y_train=np.std(log_y_train)
    normalized_log_y_train= (log_y_train - mean_log_y_train) / np.std(std_log_y_train)

    vdW_train=vdW[train_pos]
    coulomb_train=coulomb[train_pos]

    ### test part ###
    y_test=df[property_].values[test_pos]
    log_y_test=np.log(y_test)
    normalized_log_y_test= (log_y_test - mean_log_y_train) / np.std(std_log_y_train)

    vdW_test=vdW[test_pos]
    coulomb_test=coulomb[test_pos]

    return y_train,log_y_train,normalized_log_y_train,vdW_train,coulomb_train,y_test,log_y_test,normalized_log_y_test,vdW_test,coulomb_test