'''
Author: your name
Date: 2020-08-05 08:55:19
LastEditTime: 2020-12-08 01:28:49
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /FOIT/FOIT_ultra.py
'''
# sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle

# package
import joblib
import numpy as np
import time

# utils
import utils

def FOIT(dataset_name='seed4', rho=1, clf_name='lr', threshold=0.6, with_balance=True, FOIT_type='cross-all'):
    # time and accs
    c0_acc = []
    c0u_acc = []
    foit_acc = []
    time_c0 = []
    time_c0u = []
    time_foit = []

    cd_count = 16 if dataset_name=='seed4' else 9 if dataset_name=='seed3' else print('Wrong dataset_name')
    iteration_number = 3 if FOIT_type=='cross-subject' else 15
    
    number_trial, number_label, labels = utils.get_number_of_label_n_trial(dataset_name)
    data, label = utils.load_source_data(dataset_name=dataset_name, FOIT_type=FOIT_type)

    for ite in range(iteration_number):
        # The data of 15th sub is taken as the cd and ud data in cross-subject for each session (iteration_number=3)
        # The data of 3rd session is taken as the cd and ud data in cross-session for each subject (iteration_number=14)
        # For each iteration, the data of one sub from session 2 is taken as the cd and ud data in cross-all situation
        # print("Iteration: {}".format(ite))
        
        '''
        Parameters
        '''
        session_id = -1
        sub_id = -1
        accs_number = -1
        if FOIT_type == 'cross-subject':
            session_id = ite
            sub_id = 14
            accs_number = 14
        elif FOIT_type == 'cross-session':
            session_id = 2
            sub_id = ite
            accs_number = 2
        elif FOIT_type == 'cross-all':
            session_id = 1
            sub_id = ite
            accs_number = 15
        else:
            print('Wrong FOIT type!')
        
        '''
        Data
        '''
        # data
        cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, session_id=session_id, cd_count=cd_count, sub_id=sub_id)
        cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
        ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
        cd_data_min, cd_data_max = np.min(cd_data), np.max(cd_data)
        cd_data = utils.normalization(cd_data)
        ud_data = utils.normalization(ud_data)
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
            # data_ite = utils.normalization(data_ite)
        # print(len(data_ite), len(label_ite[0]), len(label_ite[0][0]))
        
        '''
        A)
        '''
        if clf_name=='svm':
            clf = svm.LinearSVC(max_iter=30000)
            # clf = svm.SVC(probability=True, max_iter=10000)
        elif clf_name=='lr':
            clf = LogisticRegression(max_iter=30000)
        else:
            print('Unexcepted clf name, using LR as the baseline now.')
            clf = svm.LinearSVC(max_iter=30000)
        clf = CalibratedClassifierCV(clf, cv=5)
        since = time.time()
        clf.fit(cd_data, cd_label.squeeze())
        time_baseline = time.time() - since
        # print('Baseline training complete in {:.4f}'.format(time_baseline))
        scoreA = utils.test(clf, ud_data, ud_label.squeeze())
        # print('Baseline score: {}'.format(scoreA))
        time_c0.append(time_baseline)
        c0_acc.append(scoreA)
        
        '''
        B)
        '''
        accs = []
        clf_sources = []
        for i in range(accs_number):
            if FOIT_type == 'cross-subject':
                path = 'models/' + dataset_name + '/csu/sesn' + str(ite) + '/lr' + str(i) + '.m'
            elif FOIT_type == 'cross-session':
                path = 'models/' + dataset_name + '/csn/sub' + str(ite) + '/lr' + str(i) + '.m'
            else:
                path = 'models/' + dataset_name + '/csun/lr' + str(i) + '.m'
            temp_clf = joblib.load(path)
            clf_sources.append(temp_clf)
            score = utils.test(temp_clf, ud_data, ud_label.squeeze())
            accs.append(score)
        if FOIT_type == 'cross-session':
            pass
        else:
            accs = utils.normalization(accs)
        # print('Accs of classifiers, normalized: {}'.format(accs))

        '''
        C)
        '''
        s_data_all, s_label_all = utils.stack_list(data_ite, label_ite)
        s_data_all_predict_proba = clf.predict_proba(s_data_all)
        s_label_all_proba = utils.get_one_hot(s_label_all.squeeze(), number_label)
        confidence = np.zeros((s_label_all_proba.shape[0], 1))
        for i in range(s_label_all_proba.shape[0]):
            confidence[i] = s_label_all_proba[i].dot(s_data_all_predict_proba[i].T)

        if with_balance:
            ## divide into 4 categories
            data_ite_divided = []
            label_ite_divided = []
            conf_divided = []
            for i in range(number_label):
                data_ite_divided.append([])
                label_ite_divided.append([])
                conf_divided.append([])
            for i in range(len(s_data_all)):
                temp_label = s_label_all[i][0]
                data_ite_divided[temp_label].append(s_data_all[i])
                label_ite_divided[temp_label].append(s_label_all[i])
                conf_divided[temp_label].append(confidence[i])
            indices = []
            for i in range(number_label):
                indices.append(np.argsort(conf_divided[i], axis=0)[::-1])
            topK_indices = [indices[i][:int(rho*len(cd_label)/4)] for i in range(len(indices))]
            S_data = None
            S_label = None
            for i in range(len(topK_indices)):
                for j in topK_indices[i]:
                    # temp_conf = conf_divided[i][j[0]]
                    one_data = data_ite_divided[i][j[0]]
                    one_label= label_ite_divided[i][j[0]]
                    if S_data is not None:
                        S_data = np.vstack((S_data, one_data))
                        S_label = np.vstack((S_label, one_label))
                    else:
                        S_data = one_data
                        S_label = one_label
        elif with_balance is False:
            indices = np.argsort(confidence, axis=0)[::-1]
            topK_indices = indices[:int(rho*len(cd_label))]
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
        else:
            print('Unexcepted value of with balance!')
            raise EnvironmentError
        
        '''
        D)
        '''
        L_S_data = cd_data.copy()
        L_S_label = cd_label.copy()
        L_S_data = np.vstack((L_S_data, S_data))
        L_S_label = np.vstack((L_S_label, S_label))
        L_S_data, L_S_label = shuffle(L_S_data, L_S_label, random_state=0)
        # L_S_data = utils.normalization(L_S_data) # to decide
        clf.fit(L_S_data, L_S_label.squeeze())
        time_updated_baseline = time.time() - since
        # print('Updated baseline training complete in {:.4f}s'.format(time_updated_baseline))
        time_c0u.append(time_updated_baseline)
        scoreD = utils.test(clf, ud_data, ud_label.squeeze())
        # print('Updated model score: {}'.format(scoreD))
        c0u_acc.append(scoreD)

        '''
        E)
        '''
        weight = (len(accs) + 1) / 2
        proba_result_all = clf.predict_proba(ud_data) * weight

        if FOIT_type == 'cross-session':
            weight_for_clfs = utils.decide_which_clf_to_use(scoreD, accs)
            for j in range(len(weight_for_clfs)):
               proba_result_all += clf_sources[j].predict_proba(utils.normalization(ud_data)) * weight_for_clfs[j]
        else:
            for i in range(len(clf_sources)):
                if accs[i] > threshold:
                    proba_result_all += clf_sources[i].predict_proba(ud_data) * accs[i]
        corrects = np.sum(np.argmax(proba_result_all, axis=1) == ud_label.squeeze())
        time_ensemble = time.time() - since
        time_foit.append(time_ensemble)
        scoreE = corrects / len(ud_label)
        # print('Ensembled model score: {}'.format(scoreE))
        foit_acc.append(scoreE)
    # print(c0_acc)
    # print(c0u_acc)
    # print(foit_acc)
    print('Mean acc and std of A: {} {}'.format(np.mean(c0_acc), np.std(c0_acc)))
    print('Mean acc and std of D: {} {}'.format(np.mean(c0u_acc), np.std(c0u_acc)))
    print('Mean acc and std of E: {} {}'.format(np.mean(foit_acc), np.std(foit_acc)))
    print("Time cost for training baseline: ", np.mean(time_c0))
    print("Time cost for training updated baseline: ", np.mean(time_c0u))
    print("Time cost for training FOIT: ", np.mean(time_foit))
    # print('A: ', c0_acc)
    # print('D: ', c0u_acc)
    # print('E: ', foit_acc)

# FOIT(dataset_name='seed4', rho=2, clf_name='lr', threshold=0.4, with_balance=True, FOIT_type='cross-session')

if __name__ == "__main__":
    # FOIT(dataset_name='seed4', FOIT_type='cross-all')
    # rho = 2
    rho = 4
    clf_name = 'svm'
    threshold = 0.6
    with_balance = False
    FOIT_type_all = ['cross-all', 'cross-session', 'cross-subject']
    dataset_name_all = ['seed4', 'seed3']
    # FOIT_type_all = ['cross-all']
    # dataset_name_all = ['seed4']
    for dataset_name in dataset_name_all:
        print('Dataset name: {}'.format(dataset_name))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        for FOIT_type in FOIT_type_all:
            print('FOIT type: {}'.format(FOIT_type))
            FOIT(dataset_name=dataset_name, rho=rho, clf_name=clf_name, threshold=threshold, with_balance=with_balance, FOIT_type=FOIT_type)
            # print()
        # print('\n')

    # FOIT(dataset_name='seed3', rho=4, clf_name='svm', threshold=0.6, with_balance=False, FOIT_type='cross-session')
    