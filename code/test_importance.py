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
import MyEval
import dill

#def TestBasic():
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
    train_data = all_feature
    train_label = all_label
    feature_names = None
    feature_names = [x.strip() for x in open('../data/features_all_v2.5.pkl_feature_list.csv').readlines()]
    name_to_index = {v:k for k,v in enumerate(feature_names)}

    for i in range(5):
        clf = MyXGB()
        # clf = MyXGB(n_estimators=10, max_depth=10, num_round=10) # Just a quick test

        clf.fit(train_data, train_label, feature_names=feature_names)
        print('train done')

        imp_scores = clf.get_importance()

        d = defaultdict(list)
        for feat_name in feature_names:
            d['name'].append(feat_name)
            d['feat_index'].append(name_to_index[feat_name])
            d['importance_val'].append(imp_scores.get(feat_name, 0))
                # fout.write('{0},{1}\n'.format(feat_name, ))

        df = pd.DataFrame(d)
        df.sort_values('importance_val', ascending=False).reset_index(drop=True).to_csv(f'../stat/features_ranked_{i}.csv')

            #     else:
            #         fout.write('{0},{1}\n'.format(i-1, 0))










