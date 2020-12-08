'''
@Description: for the cross-session-subject
@Author: voicebeer
@Date: 2020-07-03 01:38:14
@LastEditors: Please set LastEditors
@LastEditTime: 2020-08-04 06:48:16
'''
'''
1. Pick one data from session2 as data L
2. Training a SVM L on the data L
'''
# sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import log_loss

# indexes of N largest numbers
import heapq

# model storage
import joblib

# standard package
import numpy as np
import time

# utils
import utils

## to delete
temp_c0 = []
temp_c0u = []
temp_foit = []
temp_time_baseline = []
temp_time_updated_based = []
temp_time_FOIT = []

#
cd_count = 0
dataset_name = 'seed3'
if dataset_name == 'seed3':
    cd_count = 9
elif dataset_name == 'seed4':
    cd_count = 16
else:
    print("Unexcepted dataset name!")
    raise EnvironmentError
number_trial, number_label, labels= utils.get_number_of_label_n_trial(dataset_name)

for sub_number in range(15):
    print("Sub id: ", sub_number)
    session_id = 1
    cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, session_id, cd_count, sub_id=sub_number)
    subs_data, subs_label = utils.load_session_data_label(dataset_name, 0)
    subs_data, subs_label = shuffle(subs_data, subs_label, random_state=0)
    cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
    ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
    '''
    a)
    '''
    clf = svm.LinearSVC(max_iter=10000)
    clf = CalibratedClassifierCV(clf, cv=5)
    since = time.time() 
    clf.fit(utils.normalization(cd_data), cd_label.squeeze())
    time_elapsed_baseline = time.time() - since
    print('Baseline training complete in {:.4f}s'.format(time_elapsed_baseline))
    scoreA = utils.test(clf, utils.normalization(ud_data), ud_label.squeeze())
    print("Baseline score: {}".format(scoreA))
    temp_time_baseline.append(time_elapsed_baseline)
    temp_c0.append(scoreA)

    '''
    b)
    '''
    accs = []
    clf_sources = []
    for i in range(15):
        path = "models/" + dataset_name + "/csun/lr" + str(i) + ".m"
        temp_clf = joblib.load(path)
        clf_sources.append(temp_clf)
        score = utils.test(temp_clf, utils.normalization(ud_data), ud_label.squeeze())
        accs.append(score)
    accs = utils.normalization(accs) # normalize to [0,1]
    print("Accs of classifiers, normalized: {}".format(accs))

    '''
    c)
    '''
    rho = 5
    # rho = 2
    s_data_all, s_label_all = utils.stack_list(subs_data, subs_label)
    s_data_all_predict_proba = clf.predict_proba(utils.normalization(s_data_all))
    s_label_all_proba = utils.get_one_hot(s_label_all.squeeze(), number_label)
    confidence = np.zeros((s_label_all_proba.shape[0], 1))
    for i in range(s_label_all_proba.shape[0]):
        # 内积越大越可信；交叉熵越小越好；欧式距离越小越好
        confidence[i] = s_label_all_proba[i].dot(s_data_all_predict_proba[i].T)
        # confidence[i] = np.linalg.norm(s_label_all_proba[i] - s_data_all_predict_proba[i])
        # confidence[i] = log_loss(s_label_all_proba[i], s_data_all_predict_proba[i])
    
    ### with balance
    ## divide into 4 categories
    # subs_data_divided = []
    # conf_divided = []
    # subs_label_divided = []
    # for i in range(number_label):
    #     subs_data_divided.append([])
    #     conf_divided.append([])
    #     subs_label_divided.append([])
    # for i in range(len(s_data_all)):
    #     temp_label = s_label_all[i][0]
    #     subs_data_divided[temp_label].append(s_data_all[i])
    #     conf_divided[temp_label].append(confidence[i])
    #     subs_label_divided[temp_label].append(s_label_all[i])
    # indices = []
    # for i in range(number_label):
    #     indices.append(np.argsort(conf_divided[i], axis=0)[::-1])
    # topK_indices = [indices[i][:int(rho*len(cd_label)/4)] for i in range(len(indices))]
    # S_data = None
    # S_label = None
    # for i in range(len(topK_indices)):
    #     # print('110: ')
    #     for j in topK_indices[i]:
    #         temp_conf = conf_divided[i][j[0]]
    #         # temp_conf = eval('conf_'+str(i))[j[0]]
    #         print(temp_conf)
    #         one_data = subs_data_divided[i][j[0]]
    #         one_label = subs_label_divided[i][j[0]]
    #         # one_data = eval('subs_data_'+str(i))[j[0]]
    #         # one_label = eval('subs_label_'+str(i))[j[0]]
    #         if S_data is not None:
    #             S_data = np.vstack((S_data, one_data))
    #             S_label = np.vstack((S_label, one_label))
    #         else:
    #             S_data = one_data
    #             S_label = one_label
    # print(S_data.shape)
    # print(S_label.shape)

    ### without balance
    # indices = np.argsort(confidence, axis=0)
    indices = np.argsort(confidence, axis=0)[::-1]
    topK_indices = indices[:int(rho*len(cd_label))]
    S_data = None
    S_label = None
    for i in topK_indices:
        # print(confidence[i])
        one_data = s_data_all[i]
        one_label = s_label_all[i]
        if S_data is not None:
            S_data = np.vstack((S_data, one_data))
            S_label = np.vstack((S_label, one_label))
        else:
            S_data = one_data
            S_label = one_label
    # print(S_data.shape)
    # print(S_label.shape)


    '''
    d)
    '''
    print(utils.count_for_array(cd_label))
    # print(clf.predict(utils.normalization(ud_data)))
    L_S_data = cd_data.copy()
    L_S_label = cd_label.copy()
    # print(S_label)
    L_S_data = np.vstack((L_S_data, S_data))
    L_S_label = np.vstack((L_S_label, S_label))
    # L_S_data, L_S_label = shuffle(L_S_data, L_S_label, random_state=0)
    # print(L_S_data.shape)
    # print(L_S_label.shape)
    # print(cd_label.shape)
    # print(cd_data.shape)
    # print(cd_data[:5])
    # print(L_S_data[:5])
    # model2 = svm.LinearSVC(max_iter=10000)
    # clf2 = CalibratedClassifierCV(model, cv=2)
    # clf2.fit(L_S_data, L_S_label.squeeze())
    print(utils.count_for_array(L_S_label))
    # clf.fit(utils.normalization(cd_data), cd_label.squeeze())
    clf.fit(utils.normalization(L_S_data), L_S_label.squeeze())
    time_updated_baseline = time.time() - since
    print('Updated baseline training complete in {:.4f}s'.format(time_updated_baseline))
    temp_time_updated_based.append(time_updated_baseline)
    # print(clf.predict(utils.normalization(ud_data)))
    # print(ud_label.squeeze())
    scoreD = utils.test(clf, utils.normalization(ud_data), ud_label.squeeze())
    print("Updated model score: {}".format(scoreD))
    # print("a)", scoreD, scoreA)
    temp_c0u.append(scoreD)

    '''
    e)
    '''
    # print(accs)
    weight = (len(accs) + 1) / 2
    proba_result_all = clf.predict_proba(utils.normalization(ud_data)) * weight
    threshold = utils.find_threshold(accs)
    # threshold = 0.4
    # threshold = utils.find_threshold(accs) - np.std(accs)
    # print(accs)

    # 数值大意味着中位数比平均大很多也就是大部分都很好或者个别特别特别好，这种情况就要严格些用中位值；
    # 数值小意味着中位数跟平均差不多，那就是大部分都不是很好，这种情况就可以宽泛些用均值
    # 经过print发现，感觉这个差值大于0.8的话就用中位，小于0.8就用均值。
    # print("249: ", np.median(accs) - np.mean(accs))
    # print("250: ", np.median(accs)/np.mean(accs)) 
    # print(utils.find_threshold(accs))
    # print(np.mean(accs))
    # print(np.std(accs))
    # print(utils.find_threshold(accs) - np.std(accs))
    print("Threshold: ", threshold)
    for i in range(len(clf_sources)):
        if accs[i] > threshold:
            proba_result_all += clf_sources[i].predict_proba(utils.normalization(ud_data)) * accs[i]
    # # print(clf_sources)
    # # print(np.argmax(c_0_proba_result, axis=1))
    # # print(c_0_proba_result.shape)
    # # print(sum(proba_results).shape)
    # # print("152:", sum(proba_results))
    # print(proba_result_all)
    # print(ud_label)
    # print(np.argmax(proba_result_all, axis=1))
    # print(np.argmax(proba_result_all, axis=1))
    corrects = np.sum(np.argmax(proba_result_all, axis=1) == ud_label.squeeze())
    since_FOIT = time.time() - since
    print('FOIT training complete in {:.4f}s'.format(since_FOIT))
    temp_time_FOIT.append(since_FOIT)
    scoreE = corrects/len(ud_label)
    print("Ensembled model score: {}".format(scoreE))
    temp_foit.append(scoreE)
    print() 

print(temp_c0)
print(temp_c0u)
print(temp_foit)
print("A: ", np.mean(temp_c0), np.std(temp_c0))
print("D: ", np.mean(temp_c0u), np.std(temp_c0u))
print("E: ", np.mean(temp_foit), np.std(temp_foit))
print("Time cost for training baseline: ", np.mean(temp_time_baseline))
print("Time cost for training updated baseline: ", np.mean(temp_time_updated_based))
print("Time cost for training FOIT: ", np.mean(temp_time_FOIT))
