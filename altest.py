'''
Author: your name
Date: 2020-08-12 02:43:23
LastEditTime: 2020-08-27 05:48:25
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /FOIT/al.py
'''

'''
学习引擎：svm 
选择引擎： 信息熵or点积？（minimum） 
人工oracle 给100%acc label
'''

'''
Step 1: get labelled data, unlabelled data and test data.
'''
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.metrics import log_loss

# package
import numpy as np
import time

import utils

def al(dataset_name='seed4', FOIT_type='cross-all', rounds=10, batch_size=50):
    data, label = utils.load_source_data(dataset_name=dataset_name, FOIT_type=FOIT_type)
    _, number_label, _ = utils.get_number_of_label_n_trial(dataset_name)
    # data, label = utils.load_session_data_label(dataset_name, 0) # as unlabelled data

    cd_count = 16 if dataset_name=='seed4' else 9 if dataset_name=='seed3' else print('Wrong dataset_name')
    iteration_number = 3 if FOIT_type=='cross-subject' else 15
    accs = [([]) for i in range(iteration_number)]
    times = [([]) for i in range(iteration_number)]
    
    for ite in range(iteration_number):
        session_id = -1
        sub_id = -1
        if FOIT_type == 'cross-subject':
            session_id = ite
            sub_id = 14
        elif FOIT_type == 'cross-session':
            session_id = 2
            sub_id = ite
        elif FOIT_type == 'cross-all':
            session_id = 1
            sub_id = ite
        else:
            print('Wrong FOIT type!')
        # print("Ite: ", ite)
        cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, session_id=session_id, cd_count=cd_count, sub_id=sub_id)
        cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
        ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
        cd_data_min, cd_data_max = np.min(cd_data), np.max(cd_data)
        cd_data = utils.normalization(cd_data) # labelled data
        ud_data = utils.normalization(ud_data) # test data
        if FOIT_type == 'cross-all':
            data_ite, label_ite = data.copy(), label.copy()
            for i in range(len(data)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
            # data_ite, label_ite = shuffle(data, label, random_state=0)
            for i in range(len(data)):
                data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)
            # data_ite = utils.normalization(data_ite)
        elif FOIT_type == 'cross-session':
            data_ite, label_ite = data[ite], label[ite]
            for i in range(len(data_ite)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
                # data_ite[i] = utils.normalization(data_ite[i])
                data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)
            # data_ite = utils.normalization(data_ite)
        else:
            data_ite, label_ite = data[ite], label[ite]
            for i in range(len(data_ite)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
            # data_ite, label_ite = shuffle(data_ite, label_ite, random_state=0)
            for i in range(len(data_ite)):
                # data_ite[i] = utils.normalization(data_ite[i])
                data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)
        # data_ite, label_ite = data.copy(), label.copy()
        # for i in range(len(data)):
        #     data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
        # for i in range(len(data)):
        #     data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)

        # baseline
        clf = svm.LinearSVC(max_iter=30000)
        clf = CalibratedClassifierCV(clf, cv=5)
        since = time.time()
        clf.fit(cd_data, cd_label.squeeze())
        time_baseline = time.time() - since
        scoreA = utils.test(clf, ud_data, ud_label.squeeze())
        accs[ite].append(scoreA)
        times[ite].append(time_baseline)
        
        # select the data from the reservoir iteratively
        s_data_all, s_label_all = utils.stack_list(data_ite, label_ite)
        L_S_data = None
        L_S_label = None
        for i in range(rounds):
            # print("Rounds: ", i)
            # print(type(s_data_all))
            # print(s_data_all.shape)
            s_data_all_predict_proba = clf.predict_proba(s_data_all)
            s_label_all_proba = utils.get_one_hot(s_label_all.squeeze(), number_label)
            confidence = np.zeros((s_label_all_proba.shape[0], 1))
            for i in range(s_label_all_proba.shape[0]):
                confidence[i] = s_label_all_proba[i].dot(s_data_all_predict_proba[i].T)
                # confidence[i] = log_loss(s_label_all_proba[i], s_data_all_predict_proba[i])
            indices = np.argsort(confidence, axis=0) # take the minimum topK indices    
            topK_indices = indices[:batch_size]
            S_data = None
            S_label = None
            for i in topK_indices:
                one_data = s_data_all[i]
                one_label = s_label_all[i]
                if S_data is not None:
                    S_data = np.vstack((S_data, one_data))
                    S_label = np.vstack((S_label, one_label))
                else:
                    S_data = one_data
                    S_label = one_label
            for i in range(len(s_data_all)-1, -1, -1):
                if i in topK_indices:
                    s_data_all = np.delete(s_data_all, i, axis=0)
                    s_label_all = np.delete(s_label_all, i, axis=0)
            if L_S_data is None:
                L_S_data = cd_data.copy()
                L_S_label = cd_label.copy()
            else:
                pass
            L_S_data = np.vstack((L_S_data, S_data))
            L_S_label = np.vstack((L_S_label, S_label))
            L_S_data, L_S_label = shuffle(L_S_data, L_S_label, random_state=0)
            clf.fit(L_S_data, L_S_label.squeeze())
            time_updated_time = time.time() - since
            times[ite].append(time_updated_time)
            scoreTMP = utils.test(clf, ud_data, ud_label.squeeze())
            accs[ite].append(scoreTMP)
    ResultTime = []
    ResultAcc = []
    ResultStd = []
    for i in range(rounds+1):
        tmpTime = []
        tmpAcc = []
        for j in range(iteration_number):
            tmpTime.append(times[j][i])
            tmpAcc.append(accs[j][i])
        ResultTime.append(np.mean(tmpTime))
        ResultAcc.append(np.mean(tmpAcc))
        ResultStd.append(np.std(tmpAcc))
    print("Time: ", ResultTime)
    print("Accs: ", ResultAcc)
    print("Stds: ", ResultStd)
        
if __name__ == "__main__":
    FOIT_type_all = ['cross-all', 'cross-session', 'cross-subject']
    dataset_name_all = ['seed4', 'seed3']
    # FOIT_type_all = ['cross-all']
    # dataset_name_all = ['seed4']
    for dataset_name in dataset_name_all:
        print('Dataset name: {}'.format(dataset_name))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        for FOIT_type in FOIT_type_all:
            print('FOIT type: {}'.format(FOIT_type))
            batch_size = 0
            if FOIT_type=='cross-session' and dataset_name=='seed4':
                batch_size = 150
            else:
                batch_size = 250
            al(dataset_name=dataset_name, FOIT_type=FOIT_type, rounds=10, batch_size=batch_size)

    # a = [3, 4, 2, 7, 5, 9, 0, 1, 6, 8]
    # tmp = np.argsort(a, axis=0)
    # tmp = tmp[0:3]
    # print(tmp)
    # for i in range(len(a)-1, -1, -1):
    #     if i in tmp:
    #         a = np.delete(a, i)
    # print(a)