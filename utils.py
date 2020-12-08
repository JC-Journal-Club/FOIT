'''
@Description: 
@Author: voicebeer
@Date: 2020-07-03 00:53:27
LastEditors: Please set LastEditors
LastEditTime: 2020-09-21 01:44:37
'''
# for SEED data loading
import os
import scipy.io as scio

# standard package
import numpy as np
import random
random.seed(0) # temporary
import copy

# normalization function
def normalization(data):
    _range = np.max(data) - np.min(data)
    # print(_range)
    return (data - np.min(data)) / _range

def norm_with_range(data, min_data, max_data):
    _range = max_data - min_data
    return (data - min_data) / _range

dataset_path = {'seed4': 'eeg_feature_smooth', 'seed3': 'ExtractedFeatures'}
path_seed4 = "eeg_feature_smooth"
path_seed3 = "ExtractedFeatures"

'''
For loading data
'''
def get_allmats_name(dataset_name):
    '''
    @Description: get the names of all the .mat files
    @param {type}
    @return: 
        allmats: list (3*15)
    '''
    path = dataset_path[dataset_name]
    sessions = os.listdir(path)
    sessions.sort()
    allmats = []
    for session in sessions:
        if session != '.DS_Store':
            mats = os.listdir(path + '/' + session)
            mats.sort()
            mats_list = []
            for mat in mats:
                mats_list.append(mat)
            allmats.append(mats_list)
    return path, allmats

def default():
    print("Wrong FOIT type!")

def load_source_data(dataset_name='seed4', FOIT_type='cross-all'):
    switch_case = {
        'cross-subject': load_by_session(dataset_name),
        'cross-session': load_by_subject(dataset_name),
        'cross-all': load_session_data_label(dataset_name, 0),
    }
    data, label = switch_case.get(FOIT_type)
    return data, label


def load_by_subject(dataset_name):
    '''
    @Description: load one subject's data and labels, except session 3
    @param {type}: 
    @return: 
        ses_data: list (15*2*(s1,s2)*310)
        ses_label: list (15*2*(851,832)*1)
    '''
    path, allmats = get_allmats_name(dataset_name)
    ses_data = [([0] * 2) for i in range(15)]
    ses_label = [([0] * 2) for i in range(15)]
    for i in range(len(allmats[0])):
        for j in range(len(allmats)-1):
            mat_path = path + "/" + str(j+1) + "/" + allmats[j][i]
            one_sub_data, one_sub_label = get_data_label_frommat(mat_path, dataset_name, j)
            ses_data[i][j] = one_sub_data.copy()
            ses_label[i][j] = one_sub_label.copy()
    return ses_data, ses_label

def load_by_session(dataset_name):
    '''
    @description: load data and label by session, except sub 15 
    @param {type}:
    @return:
        sub_data: list (3*14*(s1,s2,s3)*310)
        sub_label: list (3*14*(851,832,822)*310)
    '''    
    path, allmats = get_allmats_name(dataset_name)
    sub_data = [([0] * 14) for i in range(3)]
    sub_label = [([0] * 14) for i in range(3)]
    for i in range(len(allmats)):
        for j in range(len(allmats[0])-1):
            mat_path = path + "/" + str(i+1) + "/" + allmats[i][j]
            one_sub_data, one_sub_label = get_data_label_frommat(mat_path, dataset_name, i)
            sub_data[i][j] = one_sub_data.copy()
            sub_label[i][j] = one_sub_label.copy()
    return sub_data, sub_label

def load_session_data_label(dataset_name, session_id=0):
    '''
    @Description: load one session's data and labels using session_id
    @param {type}: 
        session_id: int
    @return:
        subs_data: list (15*851*310)
        subs_label: list (15*851*1)
    '''
    path, allmats = get_allmats_name(dataset_name)
    subs_data = [] # 15*851*310
    subs_label = [] # 15*851*1
    for j in range(len(allmats[0])):
        mat_path = path + "/" + str(session_id+1) + "/" + allmats[session_id][j]
        one_sub_data, one_sub_label = get_data_label_frommat(mat_path, dataset_name, session_id)
        subs_data.append(one_sub_data)
        subs_label.append(one_sub_label)
    return subs_data, subs_label

def pick_one_data(dataset_name, session_id=1, cd_count=4, sub_id=0):
    '''
    @Description: pick one data from session 2 (or from other sessions), 
    @param {type}:
        session_id: int
        cd_count: int (to indicate the number of calibration data)
    @return: 
        832 for session 1, 851 for session 0
        cd_data: array (x*310, x is determined by cd_count)
        ud_data: array ((832-x)*310, the rest of that sub data)
        cd_label: array (x*1)
        ud_label: array ((832-x)*1)              
    '''
    path, allmats = get_allmats_name(dataset_name)
    mat_path = path+ "/" + str(session_id+1) + "/" + allmats[session_id][sub_id]
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values()) # 24 * 62 * x * 5
    cd_list = []
    ud_list = []
    number_trial, number_label, labels = get_number_of_label_n_trial(dataset_name)
    session_label_one_data = labels[session_id]
    for i in range(number_label):
        # 根据给定的label值从label链表中拿到全部的index后根据数量随机采样
        cd_list.extend(sample_by_value(session_label_one_data, i, int(cd_count/number_label)))
    ud_list.extend([i for i in range(number_trial) if i not in cd_list])
    cd_label_list = copy.deepcopy(cd_list)
    ud_label_list = copy.deepcopy(ud_list)
    for i in range(len(cd_list)):
        cd_list[i] = mat_de_data[cd_list[i]]
        cd_label_list[i] = labels[session_id][cd_label_list[i]]
    for i in range(len(ud_list)):
        ud_list[i] = mat_de_data[ud_list[i]]
        ud_label_list[i] = labels[session_id][ud_label_list[i]]
    
    # reshape
    cd_data, cd_label = reshape_data(cd_list, cd_label_list)
    ud_data, ud_label = reshape_data(ud_list, ud_label_list)
    
    return cd_data, cd_label, ud_data, ud_label

def sample_by_value(list, value, number):
    '''
    @Description: sample the given list randomly with given value
    @param {type}: 
        list: list
        value: int {0,1,2,3}
        number: number of sampling
    @return: 
        result_index: list
    '''
    result_index = []
    index_for_value = [i for (i,v) in enumerate(list) if v==value]
    result_index.extend(random.sample(index_for_value, number))
    return result_index

def get_data_label_frommat(mat_path, dataset_name, session_id):
    '''
    @Description: load data from mat path and reshape to 851*310
    @param {type}:
        mat_path: String
        session_id: int
    @return: 
        one_sub_data, one_sub_label: array (851*310, 851*1)
    '''
    _, _, labels = get_number_of_label_n_trial(dataset_name)
    mat_data = scio.loadmat(mat_path)
    mat_de_data = {key:value for key, value in mat_data.items() if key.startswith('de_LDS')}
    mat_de_data = list(mat_de_data.values()) # 24 * 62 * x * 5
    one_sub_data, one_sub_label = reshape_data(mat_de_data, labels[session_id])
    return one_sub_data, one_sub_label

'''
Tools
'''
def get_number_of_label_n_trial(dataset_name):
    # global variables
    label_seed4 = [[1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
                    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
                    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]]
    label_seed3 = [[2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0],
                    [2,1,0,0,1,2,0,1,2,2,1,0,1,2,0]]
     # 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1]
    if dataset_name == 'seed3':
        label = 3
        trial = 15
        return trial, label, label_seed3
    elif dataset_name == 'seed4':
        label = 4
        trial = 24
        return trial, label, label_seed4
    else:
        print('Unexcepted dataset name')

def reshape_data(data, label):
    '''
    @Description: reshape data and initiate corresponding label vector
    @param {type}:
        data: list
        label: list
    @return: 
        reshape_data: array (x*310)
        reshape_label: array (x*1)
    '''
    reshape_data = None
    reshape_label = None
    for i in range(len(data)):
        one_data = np.reshape(np.transpose(data[i], (1,2,0)), (-1,310), order='F')
        one_label = np.full((one_data.shape[0],1), label[i])
        if reshape_data is not None:
            reshape_data = np.vstack((reshape_data, one_data))
            reshape_label = np.vstack((reshape_label, one_label))
        else:
            reshape_data = one_data
            reshape_label = one_label
    return reshape_data, reshape_label

def get_one_hot(targets, nb_classes):
    '''
    @Description: get a one-hot encoding vector
    @param {type}:
        targets: list (expected 1xm)
        nb_classes: int
    @return: 
        m x nb_classes
    '''
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def test(model, data, label):
    return model.score(data, label)

def count_for_array(array):
    '''
    @description: count an array
    @param {type}:
        array: list
    @return:
        dict: {0: numbers, 1: numbers, 2: nos, 3: nos}
    '''
    unique, counts = np.unique(array, return_counts = True)
    return dict(zip(unique, counts))

def stack_list(data, label):
    '''
    @description: stack a list into one array
    @param {type}:
        data: list
        label: list
    @return: 
        result_data: array
        result_label: array
    '''
    result_data = None
    result_label = None
    for i in range(len(data)):
        one_data = data[i]
        one_label = label[i]
        if result_data is not None:
            result_data = np.vstack((result_data, one_data))
            result_label = np.vstack((result_label, one_label))
        else:
            result_data = one_data
            result_label = one_label
    return result_data, result_label

def find_threshold(list_accs):
    '''
    @description: find the threshold for ensembling (cross-sub)
    @param {type}:
        list_accs: list
    @return: 
        threshold: int
    '''
    # threshold = np.mean(list_accs)
    # threshold = np.median(list_accs)-0.01
    threshold_of_difference_high = 0.1
    threshold_of_difference_low = 0.048
    difference = np.median(list_accs) - np.mean(list_accs)
    threshold = 0
    if difference>threshold_of_difference_high or difference<threshold_of_difference_low:
        threshold = np.mean(list_accs)
    else:
        threshold = np.median(list_accs)
    return threshold

def decide_which_clf_to_use(scoreD, accs):
    '''
    @Description: decide which clf to use (cross-session)
    @param {type}:
        scoreD: float (used to help decide)
    @return: 
        result: list
    '''
    result = [0, 0]
    diff = scoreD - accs[np.argmin(accs)]
    if scoreD < accs[np.argmin(accs)]:
        return [1, 1]
    elif ((0<=diff) & (diff<0.35)) | ((0.452<diff) & (diff<0.58)):
        result[np.argmax(accs)] = 1
        return result
    else:
        return result

# a = [0.1, 0.2]
# print(decide_which_clf_to_use(0.05, a))
#    diff = np.abs(accs[0]-accs[1])
#     gap = scoreD - accs[np.argmax(accs)]
#     if scoreD < accs[np.argmin(accs)]:
#         return result
#     elif diff > 0.25:
#         return result:
#     elif 0.25 > diff > 0.2:
#         result[np.argmax(accs)] = 1
#         return result

#     if scoreD >= 0.9:
#         return result
#     if accs[0] == accs[1]:
#         if scoreD < accs[np.argmin(accs)] or accs[np.argmin(accs)] >= 0.45:
#             return [1, 1]
#         else:
#             # result[np.argmax(accs)] = 1
#             return result
#     # elif diff <= 0.01:
#     #     return [1, 1]
#     elif diff <= 0.01:
#         return [1, 1]
#     elif diff >= 0.2:
#         result[np.argmax(accs)] = 1
#         return result
#     else:
#         return result

# # data, label = load_by_session('seed3')
# # print(len(data[0][0]))
# # data, label = load_by_session('seed4')
# # print(len(data[0][0]))

# subs_data, subs_label = load_session_data_label('seed3', 1)
# print(len(subs_data[0][0][0]))


# subs_data_divided = []
# for i in range(3):
#     subs_data_divided.append([])
# for i in range(len(subs_data_divided)):
#     subs_data_divided[i].append(2)
#     subs_data_divided[i].append(3)

# print(subs_data_divided)
# print(subs_data_divided[1])
# print(np.argsort(subs_data_divided[1]))

# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt

# ses_number = 1
# dataset_name = 'seed3'
# cd_count = 9
# sub_data, sub_label = load_by_session(dataset_name)
# cd_data, cd_label, ud_data, ud_label = pick_one_data(dataset_name, ses_number, cd_count, sub_id=14)
# sub_data_ses, sub_label_ses = sub_data[ses_number], sub_label[ses_number]
# # sub_data_ses, sub_label_ses = shuffle(sub_data_ses, sub_label_ses, random_state=0)
# # cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
# # ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)

# fig = plt.figure()
# tsne = TSNE(n_components=3, init='pca', random_state=0)
# print(len(sub_data_ses[0]))

# # c = np.random.randn(14)
# # print(colors)
# for i in range(len(sub_data_ses)):
#     trans_data = tsne.fit_transform(sub_data_ses[i])
#     # ax = fig.add_subplot(2, 5, 10)
#     plt.scatter(trans_data[:,0], trans_data[:,1], marker=".")
# # plt.axis('tight')

# trans_cd_data = tsne.fit_transform(cd_data)
# plt.scatter(trans_cd_data[:,0], trans_cd_data[:,1], marker="x")

# plt.show()

# data, label = load_source_data('seed3','cross-all')
# print(len(data[0]))
# print(len(label[0][0]))

# from sklearn.utils import shuffle

# a = [[1, 2], [4, 5, 6], [7, 8, 9]]
# b = [[-1, -2], [-4, -5, -6], [-7, -8, -9]]
# for i in range(len(a)):
#     a[i], b[i] = shuffle(a[i], b[i], random_state=0)
# print(a, b)

# a = [1, 2, 3, 4, 5, 6]
# b = [-1, -2, -3, -4, -5, -6]
# a, b = shuffle(a, b, random_state=0)
# print(a, b)

# a = np.array([[1,2],
#      [3,4],
#      [5,6]])
# b = np.array([1, 2, 3])
# print(a.shape)
# print(a)
# print(b.shape)

# number_of_array = a.shape[0]
# temp_array = list(range(number_of_array))
# print(temp_array)
# temp_index = random.sample(temp_array, 2)
# print(temp_index)
# c = np.array([a[i] for i in temp_index])
# print(c.shape)

# a = [1, 2, 3]
# print(a[:7])

