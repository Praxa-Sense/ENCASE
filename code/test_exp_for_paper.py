#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 11:08:54 2017

@author: shenda
"""
import os.path
import time
from datetime import timedelta
from pprint import pprint

import dill
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

import ReadData
import MyEval
from BasicCLF import MyLR
from BasicCLF import MyXGB
from Encase import Encase

CSV_NAME = '../stat/res_exp_for_paper_NEW.csv'

def TestExp(all_pid, all_feature, all_label, method, i_iter):
    kf = StratifiedKFold(n_splits=5, shuffle=True)
    i_fold = 5
    print('all feature shape: {0}'.format(all_feature.shape))
    results = []
    for train_index, test_index in kf.split(all_feature, all_label):
        print("train index size=", len(train_index), len(train_index)/all_feature.shape[0])
        train_data = all_feature[train_index]
        train_label = all_label[train_index]
        test_data = all_feature[test_index]
        test_label = all_label[test_index]
        test_pid = all_pid[test_index]

        ### ENCASE
        if method == 'ENCASE_E':
            # Gerard: This is only required if you actually have more than one source of features (we only have E right now)
            # selected_cols = list(range(258, 558))
            # selected_cols = list(range(258, 558))
            # train_data = train_data[:, selected_cols]
            # test_data = test_data[:, selected_cols]
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB()
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        if method == 'ENCASE_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB()
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        if method == 'ENCASE_ECD':
            clf_1 = MyXGB()
            clf_2 = MyXGB()
            clf_3 = MyXGB()
            clf_4 = MyXGB()
            clf_5 = MyXGB()
            clf_final = Encase([clf_1, clf_2, clf_3, clf_4, clf_5])
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'XGBoost_E':
            # selected_cols = list(range(258, 558))
            # train_data = train_data[:, selected_cols]
            # test_data = test_data[:, selected_cols]
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'XGBoost_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'XGBoost_ECD':
            clf_final = MyXGB(n_estimators=100, num_round=50)
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'LR_E':
            selected_cols = list(range(258, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'LR_EC':
            selected_cols = list(range(0, 558))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'LR_ECD':
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)


        elif method == 'SampleEn':
            selected_cols = [300, 301, 302, 303]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'CDF':
            selected_cols = [304, 305, 306]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'MAD':
            selected_cols = [307]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'Variability':
            selected_cols = [346, 347, 348, 349, 350]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyLR()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)

        elif method == 'XGBoost_top10':
            selected_cols = [57, 51, 72, 91, 86, 253, 90, 66, 256, 80]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        elif method == 'XGBoost_BasicStats20':
            selected_cols = [57, 51, 50, 59, 61, 55, 58, 62, 60, 52, 72, 66, 65, 75, 69, 68, 70, 67, 74, 73]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        elif method == 'XGBoost_VariabilityStats':
            selected_cols = [91, 90, 89, 92, 88]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        elif method == 'XGBoost_DeltaRRStats':
            # selected_cols = [86, 84, 79, 80, 82, 85, 78, 81, 76, 77, 83]
            selected_cols = [86, 84, 79, 80, 82, 85, 78, 81]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        elif method == 'XGBoost_Sampen2':
            selected_cols = [42, 43, 44, 45]
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        elif method == 'DummyClassifier':
            clf_final = DummyClassifier()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
            pred = ReadData.Label2Index(pred)
        elif method == 'XGBoost_bin_stat':
            selected_cols = list(range(95, 146 + 1))
            train_data = train_data[:, selected_cols]
            test_data = test_data[:, selected_cols]
            clf_final = MyXGB()
            clf_final.fit(train_data, train_label)
            pred = clf_final.predict(test_data)
            y = ReadData.Label2Index(test_label)
        else:
            raise ValueError("Unknown method ", method)



    # res = MyEval.F14Exp(pred, test_label)
    # print(res)
    #
    # with open(CSV_NAME, 'a') as fout:
    #     fout.write('{0},{1},{2},{3},{4},{5},{6},{7}\n'
    #                .format(method, i_iter, i_fold, res[0], res[1], res[2], res[3], res[4]))

        res = classification_report(y, pred, labels=list(range(4)), output_dict=True)
        pprint(res)
        results.append(res)

    mean_f1_scores = [x['macro avg']['f1-score'] for x in results]
    mean_f1 = np.mean(mean_f1_scores)
    print(f"F1-score: {mean_f1} += {np.std(mean_f1_scores)}")
    with open(CSV_NAME, 'a') as fout:
        scores = []
        for i in range(4):
            f1_cls = np.mean([x[str(i)]['f1-score'] for x in results])
            print(f"F1_{i}-score:", f1_cls)
            scores.append(f1_cls)
        fout.write('{0},{1},{2},{3},{4},{5},{6},{7},{8}\n'
                   .format(method, i_iter, i_fold, scores[0], scores[1], scores[2], scores[2], mean_f1, np.std(mean_f1_scores)))

    i_fold += 1


if __name__ == "__main__":

    with open('../data/features_all_v2.5.pkl', 'rb') as my_input:
        all_pid = dill.load(my_input)
        all_feature = np.array(dill.load(my_input))
        all_label = np.array(dill.load(my_input))
        print('features_all shape: ', all_feature.shape)

    # with open('../data/feat_deep_centerwave_v0.1.pkl', 'rb') as my_input:
    #     feat_deep_centerwave = np.array(dill.load(my_input))
    #     print('feat_deep_centerwave shape: ', feat_deep_centerwave.shape)

    # with open('../data/feat_resnet.pkl', 'rb') as my_input:
    #     feat_resnet = np.array(dill.load(my_input))
    #     print('feat_resnet shape: ', feat_resnet.shape)

    # all_feature = np.c_[all_feature, feat_deep_centerwave, feat_resnet]
    all_feature = np.c_[all_feature]
    all_label = np.array(all_label)
    all_pid = np.array(all_pid)

    # if os.path.exists()

    # Create header if not exists
    if not os.path.exists(CSV_NAME):
        with open(CSV_NAME, 'w') as fout:
            fout.write('Method,n_iter,n_fold,F1_N,F1_A,F1_O,F1_P,F1,F1_std\n')

    # method_list = ['SampleEn', 'CDF', 'MAD', 'Variability', 
    #                'LR_E', 'LR_EC', 'LR_ECD', 
    #                'XGBoost_E', 'XGBoost_EC', 'XGBoost_ECD', 
    #                'ENCASE_E', 'ENCASE_EC', 'ENCASE_ECD']    
    # method_list = ['ENCASE_EC', 'ENCASE_ECD']
    # method_list = ['ENCASE_E']
    # method_list = ['XGBoost_E', 'ENCASE_E']
    # method_list = ['XGBoost_top10']
    # method_list = ['XGBoost_BasicStats20']
    # method_list = ['XGBoost_Sampen2']
    method_list = ['XGBoost_bin_stat']

    for method in method_list:
        t = time.time()
        # for i in range(2):
        TestExp(all_pid, all_feature, all_label, method, 0)
        print(f"Training {method} took {timedelta(seconds=(time.time() - t))}.")

        # Not sure if this is useful, it seems parallelization is already employed by XGBoost?
        # args = [(all_pid, all_feature, all_label, method, i) for i in range(5)]
        # with ThreadPool(5) as pool:
        #     pool.starmap(TestExp, args)
