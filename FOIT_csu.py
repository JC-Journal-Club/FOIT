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

## to delete
temp_c0 = []
temp_c0u = []
temp_foit = []
temp_time_baseline = []
temp_time_updated_based = []
temp_time_FOIT = []

#
# cd_count = 16
cd_count = 0
dataset_name = 'seed4'
if dataset_name == 'seed3':
    cd_count = 9
elif dataset_name == 'seed4':
    cd_count = 16
number_trial, number_label, labels= utils.get_number_of_label_n_trial(dataset_name)

sub_data, sub_label = utils.load_by_session(dataset_name) # 3*14*(m*310)

for ses_number in range(3):
    print("Session id: ", ses_number)
    # cross-subject, 取sub15
    cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, ses_number, cd_count, sub_id=14)
    sub_data_ses, sub_label_ses = sub_data[ses_number], sub_label[ses_number] # 14*(m*310)
    sub_data_ses, sub_label_ses = shuffle(sub_data_ses, sub_label_ses, random_state=0)
    cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
    ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)

    '''
    a)
    '''
    clf = svm.LinearSVC(max_iter=10000)
    # clf = LogisticRegression(max_iter=10000)
    clf = CalibratedClassifierCV(clf, cv=5)
    since = time.time()
    clf.fit(utils.normalization(cd_data), cd_label.squeeze())
    time_elapsed_baseline = time.time() - since
    print('Baseline training complete in {:.4f}s'.format(time_elapsed_baseline))
    scoreA = utils.test(clf, utils.normalization(ud_data), ud_label.squeeze())
    print('Baseline score: ', scoreA)
    temp_time_baseline.append(time_elapsed_baseline)
    temp_c0.append(scoreA)

    '''
    b)
    '''
    accs = [] # 14 classifiers from the reservoir for each session
    clf_sources = []
    for i in range(14):
        path = 'models/' + dataset_name + '/csu/sesn' + str(ses_number) + '/lr' + str(i) + '.m'
        temp_clf = joblib.load(path)
        clf_sources.append(temp_clf)
        score = utils.test(temp_clf, utils.normalization(ud_data), ud_label.squeeze())
        accs.append(score)
    # print('Accs of classifiers: {}'.format(accs))
    accs = utils.normalization(accs)
    print('Accs of classifiers, normalized: {}'.format(accs))

    '''
    c)
    '''
    rho = 0.5
    s_data_all, s_label_all = utils.stack_list(sub_data_ses, sub_label_ses)
    s_data_all_predict_proba = clf.predict_proba(utils.normalization(s_data_all))
    s_label_all_proba = utils.get_one_hot(s_label_all.squeeze(), number_label)
    confidence = np.zeros((s_label_all_proba.shape[0], 1))
    for i in range(s_label_all_proba.shape[0]):
        confidence[i] = s_label_all_proba[i].dot(s_data_all_predict_proba[i].T)
    subs_data_0, subs_data_1, subs_data_2, subs_data_3 = [], [], [], []
    conf_0, conf_1, conf_2, conf_3 = [],[],[],[]
    subs_label_0, subs_label_1, subs_label_2, subs_label_3 = [],[],[],[]
    for i in range(len(s_data_all)):
        temp_label = s_label_all[i][0]
        eval('subs_data_' + str(temp_label)).append(s_data_all[i])
        eval('conf_' + str(temp_label)).append(confidence[i])
        eval('subs_label_' + str(temp_label)).append(s_label_all[i])
    indices = []
    for i in range(4):
        indices.append(np.argsort(eval('conf_'+str(i)), axis=0)[::-1])
        # indices.append(np.argsort(eval('conf_'+str(i)), axis=0))
    topK_indices = [indices[i][:int(rho*len(cd_label)/4)] for i in range(len(indices))]
    S_data = None
    S_label = None
    for i in range(len(topK_indices)):
        for j in topK_indices[i]:
            temp_conf = eval('conf_'+str(i))[j[0]]
            one_data = eval('subs_data_'+str(i))[j[0]]
            one_label = eval('subs_label_'+str(i))[j[0]]
            if S_data is not None:
                S_data = np.vstack((S_data, one_data))
                S_label = np.vstack((S_label, one_label))
            else:
                S_data = one_data
                S_label = one_label
    # print(len(cd_label))
    # print(S_data.shape)
    # print(S_label.shape)

    ### without balance
    # indices = np.argsort(confidence, axis=0)
    # indices = np.argsort(confidence, axis=0)[::-1]
    # topK_indices = indices[:int(rho*len(cd_label))]
    # S_data = None
    # S_label = None
    # for i in topK_indices:
    #     # print(confidence[i])
    #     one_data = s_data_all[i]
    #     one_label = s_label_all[i]
    #     if S_data is not None:
    #         S_data = np.vstack((S_data, one_data))
    #         S_label = np.vstack((S_label, one_label))
    #     else:
    #         S_data = one_data
    #         S_label = one_label

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
    weight = (len(accs) + 1) / 2
    proba_result_all = clf.predict_proba(utils.normalization(ud_data)) * weight
    # threshold = utils.find_threshold(accs)
    threshold = 0.6
    print("Threshold: ", threshold)
    for i in range(len(clf_sources)):
        if accs[i] > threshold:
            proba_result_all += clf_sources[i].predict_proba(utils.normalization(ud_data)) * accs[i]
    corrects = np.sum(np.argmax(proba_result_all, axis=1) == ud_label.squeeze())
    since_FOIT = time.time() - since
    print('FOIT training complete in {:.4f}s'.format(since_FOIT))
    temp_time_FOIT.append(since_FOIT)
    scoreE = corrects/len(ud_label)
    print("Ensembled model score: {}".format(scoreE))
    temp_foit.append(scoreE)

print(temp_c0)
print(temp_c0u)
print(temp_foit)
print("A: ", np.mean(temp_c0), np.std(temp_c0))
print("D: ", np.mean(temp_c0u), np.std(temp_c0u))
print("E: ", np.mean(temp_foit), np.std(temp_foit))
print("Time cost for training baseline: ", np.mean(temp_time_baseline))
print("Time cost for training updated baseline: ", np.mean(temp_time_updated_based))
print("Time cost for training FOIT: ", np.mean(temp_time_FOIT))
