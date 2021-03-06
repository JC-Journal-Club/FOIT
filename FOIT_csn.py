# sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle
# from sklearn.metrics import log_loss

# indexes of N largest numbers
import heapq

# model storage
import joblib

# standard package
import numpy as np
import time

# utils
import utils

# time and acc
temp_c0 = []
temp_c0u = []
temp_foit = []
temp_time_baseline = []
temp_time_updated_based = []
temp_time_FOIT = []

#
cd_count = 0
dataset_name = 'seed4'
if dataset_name == 'seed3':
    cd_count = 9
elif dataset_name == 'seed4':
    cd_count = 16
else:
    print("Unexcepted dataset name!")
    raise EnvironmentError
number_trial, number_label, labels= utils.get_number_of_label_n_trial(dataset_name)

ses_data, ses_label = utils.load_by_subject(dataset_name) # 15*2*(m*310)

for sub_number in range(15):
    print("Sub id: ", sub_number)
    # cross-session 取session 3
    cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, session_id=2, cd_count=cd_count, sub_id=sub_number)
    ses_data_sub, ses_label_sub = ses_data[sub_number], ses_label[sub_number] # 2*(m*310)
    ses_data_sub, ses_label_sub = shuffle(ses_data_sub, ses_label_sub, random_state=0)
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
    print("Baseline score: ", scoreA)
    temp_time_baseline.append(time_elapsed_baseline)
    temp_c0.append(scoreA)
    
    '''
    b)
    '''
    accs = [] # two classifiers from the reservoir for each sub
    clf_sources = []
    for j in range(2):
        path = "models/" + dataset_name + "/csn/sub" + str(sub_number) + "/lr" + str(j) + ".m"
        temp_clf = joblib.load(path)
        clf_sources.append(temp_clf)
        score = utils.test(temp_clf, utils.normalization(ud_data), ud_label.squeeze())
        accs.append(score)
    print("Accs of classifiers: {}".format(accs))
    # accs = utils.normalization(accs)
    
    '''
    c)
    '''
    rho = 2

    s_data_all, s_label_all = utils.stack_list(ses_data_sub, ses_label_sub)
    s_data_all_predict_proba = clf.predict_proba(utils.normalization(s_data_all))
    s_label_all_proba = utils.get_one_hot(s_label_all.squeeze(), number_label)
    confidence = np.zeros((s_label_all_proba.shape[0], 1))
    for j in range(s_label_all_proba.shape[0]):
        confidence[j] = s_label_all_proba[j].dot(s_data_all_predict_proba[j].T)
    # print(confidence.shape)
    # subs_data_0, subs_data_1, subs_data_2, subs_data_3 = [], [], [], []
    # conf_0, conf_1, conf_2, conf_3 = [],[],[],[]
    # subs_label_0, subs_label_1, subs_label_2, subs_label_3 = [],[],[],[]
    # for j in range(len(s_data_all)):
    #     temp_label = s_label_all[j][0]
    #     eval('subs_data_' + str(temp_label)).append(s_data_all[j])
    #     eval('conf_' + str(temp_label)).append(confidence[j])
    #     eval('subs_label_' + str(temp_label)).append(s_label_all[j])
    # indices = []
    # for j in range(4):
    #     indices.append(np.argsort(eval('conf_'+str(j)), axis=0)[::-1])
    # topK_indices = [indices[j][:int(rho*len(cd_label)/4)] for j in range(len(indices))]
    # S_data = None
    # S_label = None
    # for k in range(len(topK_indices)):
    #     for j in topK_indices[k]:
    #         temp_conf = eval('conf_'+str(k))[j[0]]
    #         one_data = eval('subs_data_'+str(k))[j[0]]
    #         one_label = eval('subs_label_'+str(k))[j[0]]
    #         if S_data is not None:
    #             S_data = np.vstack((S_data, one_data))
    #             S_label = np.vstack((S_label, one_label))
    #         else:
    #             S_data = one_data
    #             S_label = one_label
    # print(len(cd_label))
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
    L_S_data = cd_data.copy()
    L_S_label = cd_label.copy()
    L_S_data = np.vstack((L_S_data, S_data))
    L_S_label = np.vstack((L_S_label, S_label))
    L_S_data, L_S_label = shuffle(L_S_data, L_S_label, random_state=0)
    print(utils.count_for_array(L_S_label))
    clf.fit(utils.normalization(L_S_data), L_S_label.squeeze())
    time_updated_baseline = time.time() - since
    print('Updated baseline training complete in {:.4f}s'.format(time_updated_baseline))
    temp_time_updated_based.append(time_updated_baseline)
    scoreD = utils.test(clf, utils.normalization(ud_data), ud_label.squeeze())
    print("Updated model score: {}".format(scoreD))
    temp_c0u.append(scoreD)

    '''
    e)
    '''
    weight = 2
    proba_result_all = clf.predict_proba(utils.normalization(ud_data)) * weight
    
    weight_for_clfs = utils.decide_which_clf_to_use(scoreD, accs)
    # weight_for_clfs = utils.decide_which_clf_to_use(scoreA, accs)
    # weight_for_clfs = [1, 1]
    print(weight_for_clfs)
    # print(accs)
    for j in range(len(weight_for_clfs)):
        proba_result_all += clf_sources[j].predict_proba(utils.normalization(ud_data)) * weight_for_clfs[j]
    
    # max_index = np.argmax(accs)
    # proba_result_all += clf_sources[max_index].predict_proba(utils.normalization(ud_data)) * 1

    corrects = np.sum(np.argmax(proba_result_all, axis=1) == ud_label.squeeze())
    since_FOIT = time.time() - since
    print('FOIT training complete in {:.4f}s'.format(since_FOIT))
    temp_time_FOIT.append(since_FOIT)
    scoreE = corrects/len(ud_label)
    print("Ensembled model score: {}".format(scoreE))
    temp_foit.append(scoreE)
    # count = 1
    # threshold = utils.find_threshold(accs)
    # print("249: ", np.median(accs) - np.mean(accs))
    # print("250: ", np.median(accs)/np.mean(accs))
    # print("Threshold: ", threshold)
    # for i in range(len(clf_sources)):
    #     if accs[i] > threshold:
    #         count += 1
    #         proba_result_all += clf_sources[i].predict_proba(utils.normalization(ud_data)) * accs[i]
    # proba_result_all = proba_result_all / count
    # corrects = np.sum(np.argmax(proba_result_all, axis=1) == ud_label.squeeze())
    # since_FOIT = time.time() - since
    # print('FOIT training complete in {:.4f}s'.format(since_FOIT))
    # temp_time_FOIT.append(since_FOIT)
    # temp_foit.append(corrects/len(ud_label))

print(temp_c0)
print(temp_c0u)
print(temp_foit)
print("A: ", np.mean(temp_c0), np.std(temp_c0))
print("D: ", np.mean(temp_c0u), np.std(temp_c0u))
print("E: ", np.mean(temp_foit), np.std(temp_foit))
print("Time cost for training baseline: ", np.mean(temp_time_baseline))
print("Time cost for training updated baseline: ", np.mean(temp_time_updated_based))
print("Time cost for training FOIT: ", np.mean(temp_time_FOIT))

