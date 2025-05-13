import scipy
import numpy as np
from scipy import stats
from statsmodels.stats.weightstats import DescrStatsW
from sklearn.model_selection import train_test_split
import copy
import pandas as pd
from scipy.stats import skew


class Preprocessing():

    def __init__(self, channel, SN_ratio):
        self.channel = channel
        self.SN_ratio = SN_ratio

        self.processedchannel = np.copy(channel)

        self.feature = np.ones(channel.shape[1])
        self.min_cutoff = -30 # -30kcal~30kcal is only used for feature in the original paper
        self.max_cutoff=30 # -30kcal~30kcal is only used for feature in the original paper
        self.mean_v=self.processedchannel.mean(axis=0)
        self.sd_v=self.processedchannel.std(axis=0)

    def detect_small_sd_pos(self, sml_sd=2):
        del_pos = np.where(self.sd_v < sml_sd)[0]
        self.feature[del_pos] = 0

    def detect_large_skew_pos(self, normal_skew=2.5):
        skewness_v = skew(self.processedchannel)
        del_pos = np.where(np.abs(skewness_v) > normal_skew)[0]
        self.feature[del_pos] = 0

    def determine_cutoff(self, upper_percent=95, lower_percent=5):
        min_cutoff = np.percentile(self.processedchannel, lower_percent)
        max_cutoff = np.percentile(self.processedchannel, upper_percent)
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff

    def overwrite_too_big_energy(self):
        self.processedchannel[self.processedchannel > self.max_cutoff] = self.max_cutoff


    def overwrite_too_small_energy(self):
        self.processedchannel[self.processedchannel < self.min_cutoff] = self.min_cutoff

    def output_normalize(self):
        norm_processedchannel = np.subtract(self.processedchannel, self.mean_v)
        norm_processedchannel = np.divide(norm_processedchannel, self.sd_v)
        feature_pos=np.where(self.feature==1)[0]
        return norm_processedchannel[:,feature_pos]

    def output_centering(self):
        center_processedchannel = np.subtract(self.processedchannel, self.mean_v)
        feature_pos = np.where(self.feature == 1)[0]
        return center_processedchannel[:,feature_pos]

    def output_extracted_raw(self):
        feature_pos = np.where(self.feature == 1)[0]
        return self.processedchannel[:,feature_pos]

    def output_train_params(self):
        use_params = [self.feature, self.min_cutoff, self.max_cutoff,self.mean_v,self.sd_v]
        return use_params

    def update_statistics(self):
        self.mean_v=self.processedchannel.mean(axis=0)
        self.sd_v=self.processedchannel.std(axis=0)

    def set_train_params(self, train_params):
        self.feature = train_params[0]
        self.min_cutoff = train_params[1]
        self.max_cutoff = train_params[2]
        self.mean_v=train_params[3]
        self.sd_v=train_params[4]

    def use_train_params(self):
        self.overwrite_too_big_energy()
        self.overwrite_too_small_energy()

    def show_feature_num(self):
        feature_pos = np.where(self.feature == 1)[0]
        print('number of features=',feature_pos.shape[0])

    def set_features_manual(self,feature):
        self.feature=feature

    def preprocess_default(self):
        self.max_cutoff=100
        self.min_cutoff=-100
        self.detect_small_sd_pos()
        self.detect_large_skew_pos()
        self.overwrite_too_big_energy()
        self.overwrite_too_small_energy()
        self.update_statistics()
        self.detect_small_sd_pos()
        self.detect_large_skew_pos()



