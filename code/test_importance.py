#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:08:54 2017

@author: shenda
"""
import os.path
from collections import defaultdict
from pprint import pprint

import pandas as pd

from BasicCLF import MyXGB

import numpy as np
import dill

if __name__ == "__main__":

    with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = np.array(dill.load(my_input))
        all_label = np.array(dill.load(my_input))
        print('features_all shape: ', all_feature.shape)
        
    # with open('../data/feat_deep_centerwave_v0.1.pkl', 'rb') as my_input:
    #     feat_deep_centerwave = np.array(dill.load(my_input))
    #     print('feat_deep_centerwave shape: ', feat_deep_centerwave.shape)
    #
    # with open('../data/feat_resnet.pkl', 'rb') as my_input:
    #     feat_resnet = np.array(dill.load(my_input))
    #     print('feat_resnet shape: ', feat_resnet.shape)
        
    
    # k-fold cross validation
    # all_feature = np.c_[all_feature, feat_deep_centerwave, feat_resnet]
    all_feature = np.c_[all_feature]
    all_label = np.array(all_label)
    # all_feature[:, 91] = np.zeros_like(all_feature[:, 91]) # Testing if this make 'Variability_Stepping' drop to 0 (it does)

    # Top 20 QRSBasicStats (XGBoost_BasicStats20)
    selected_cols = [57, 51, 50, 59, 61, 55, 58, 62, 60, 52, 72, 66, 65, 75, 69, 68, 70, 67, 74, 73]
    all_feature = all_feature[:, selected_cols]
    # all_label = test_data[:, selected_cols]

    train_data = all_feature
    train_label = all_label
    feature_names = None
    feature_names = [x.strip() for x in open('../data/features_all_v2.5.pkl_feature_list.csv').readlines()]
    feature_names = np.array(feature_names)[selected_cols]
    print("Feature Names =", feature_names)
    name_to_index = {v:k for k,v in enumerate(feature_names)}

    d = defaultdict(list)
    for feat_name in feature_names:
        # d['name'].append(feat_name)
        d['feat_index'].append(name_to_index[feat_name])

    importance_vals = {k : [] for k in feature_names}
    repeats = 10
    for i in range(repeats):
        clf = MyXGB()
        # clf = MyXGB(n_estimators=10, max_depth=10, num_round=10) # Just a quick test

        clf.fit(train_data, train_label, feature_names=feature_names)

        imp_scores = clf.get_importance()
        for feat_name in feature_names:
            score = imp_scores.get(feat_name, 0)
            importance_vals[feat_name].append(score)

    df = pd.DataFrame(d, index=feature_names)
    # df['importance_val'] = 0
    imp_vals = []
    imp_vals_std = []
    for k, v in importance_vals.items():
        # print(k, np.mean(v))
        imp_vals.append(np.mean(v))
        imp_vals_std.append(np.std(v))
    df['importance_val'] = imp_vals
    df['importance_val_std'] = imp_vals_std
    df.sort_values('importance_val', ascending=False).to_csv(f'../stat/features_ranked_ALL.csv')

            #     else:
            #         fout.write('{0},{1}\n'.format(i-1, 0))










