'''
@Description: 
@Author: voicebeer
@Date: 2020-07-03 00:53:24
@LastEditors: Please set LastEditors
@LastEditTime: 2020-08-04 01:17:37
'''

'''
1. Load source data.
2. Training one classifier to one source data using SVM(other models as alternatives)
'''
# for the model
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

# model storage
import joblib

# standard package
import numpy as np

# utils
import utils
    
dataset_name = 'seed3'

def initiate_cross_sub_reservoir():
    sub_data, sub_label = utils.load_by_session(dataset_name) # 3*14*(m*310)
    sub_data, sub_label = shuffle(sub_data, sub_label, random_state=0)
    for i in range(3):
        for j in range(14):
            clf = LogisticRegression(random_state=0, max_iter=10000)
            clf.fit(utils.normalization(sub_data[i][j]), sub_label[i][j].squeeze())
            print(("clf " + str(i) + " " + str(j)), utils.test(clf, utils.normalization(sub_data[i][j]), sub_label[i][j].squeeze()))
            if dataset_name == 'seed4':
                path = "models/seed4/csu/sesn" + str(i) + "/lr" + str(j) + ".m"
            elif dataset_name == 'seed3':
                path = "models/seed3/csu/sesn" + str(i) + "/lr" + str(j) + ".m"
            joblib.dump(clf, path)

def initiate_cross_ses_reservoir():
    ses_data, ses_label = utils.load_by_subject(dataset_name) # 15*2*(m*310)
    ses_data, ses_label = shuffle(ses_data, ses_label, random_state=0)
    for i in range(15):
        for j in range(2):
            clf = LogisticRegression(random_state=0, max_iter=10000)
            # clf = svm.LinearSVC(max_iter=10000)
            # clf = CalibratedClassifierCV(clf, cv=5)
            clf.fit(utils.normalization(ses_data[i][j]), ses_label[i][j].squeeze())
            print(("clf " + str(i) + " " + str(j)), utils.test(clf, utils.normalization(ses_data[i][j]), ses_label[i][j].squeeze()))
            if dataset_name == 'seed4':
                path = "models/seed4/csn/sub" + str(i) + "/lr" + str(j) + ".m"
            elif dataset_name == 'seed3':
                path = "models/seed3/csn/sub" + str(i) + "/lr" + str(j) + ".m"
            # path = "models/csn/sub" + str(i) + "/lr" + str(j) + ".m"
            joblib.dump(clf, path)

def initiate_cross_sub_ses_reservoir():
    subs_data, subs_label = utils.load_session_data_label(dataset_name, 0)
    subs_data, subs_label = shuffle(subs_data, subs_label, random_state=0)
    # print(len(subs_data[0]))
    for i in range(15):
        clf = LogisticRegression(random_state=0, max_iter=10000)
        clf.fit(utils.normalization(subs_data[i]), subs_label[i].squeeze())
        print("clf: ", utils.test(clf, utils.normalization(subs_data[i]), subs_label[i].squeeze()))
        if dataset_name == 'seed4':
            path = "models/seed4/csun/lr" + str(i) + ".m"
        elif dataset_name == 'seed3':
            path = "models/seed3/csun/lr" + str(i) + ".m"
        # path = "models/csun/lr" + str(i) + ".m"
        joblib.dump(clf, path)

# initiate_cross_ses_reservoir()
# initiate_cross_sub_reservoir()
initiate_cross_sub_ses_reservoir()