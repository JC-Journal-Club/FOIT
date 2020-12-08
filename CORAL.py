'''
Author: your name
Date: 2020-08-25 23:37:57
LastEditTime: 2020-08-26 10:12:27
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /FOIT/CORAL.py
'''
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors

# data
import time
import utils
import random
random.seed(0)
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        cov_src = np.cov(Xs.T) + np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + np.eye(Xt.shape[1])
        A_coral = np.dot(scipy.linalg.fractional_matrix_power(cov_src, -0.5),
                         scipy.linalg.fractional_matrix_power(cov_tar, 0.5))
        Xs_new = np.dot(Xs, A_coral).astype(float)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt, testX, testY):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''
        Xs_new = self.fit(Xs, Xt)
        # clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
        clf = svm.LinearSVC(max_iter=30000)
        clf = CalibratedClassifierCV(clf, cv=5)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(testX)
        acc = sklearn.metrics.accuracy_score(testY, y_pred)
        return acc, y_pred

def testCORAL(dataset_name='seed4', FOIT_type='cross-all'):
    data, label = utils.load_source_data(dataset_name=dataset_name, FOIT_type=FOIT_type)
    cd_count = 16 if dataset_name=='seed4' else 9 if dataset_name=='seed3' else print('Wrong dataset_name')
    iteration_number = 3 if FOIT_type=='cross-subject' else 15
    accs = []
    times = []
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
        cd_data, cd_label, ud_data, ud_label = utils.pick_one_data(dataset_name, session_id=session_id, cd_count=cd_count, sub_id=sub_id)
        cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
        ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
        # cd_data_min, cd_data_max = np.min(cd_data), np.max(cd_data)
        cd_data = utils.normalization(cd_data) # labelled data
        ud_data = utils.normalization(ud_data) # test data
        if FOIT_type == 'cross-all':
            data_ite, label_ite = data.copy(), label.copy()
            for i in range(len(data)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
            # data_ite, label_ite = shuffle(data, label, random_state=0)
            for i in range(len(data)):
                data_ite[i] = utils.normalization(data_ite[i])
            # data_ite = utils.normalization(data_ite)
        elif FOIT_type == 'cross-session':
            data_ite, label_ite = data[ite], label[ite]
            for i in range(len(data_ite)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
                data_ite[i] = utils.normalization(data_ite[i])
                # data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)
            # data_ite = utils.normalization(data_ite)
        else:
            data_ite, label_ite = data[ite], label[ite]
            for i in range(len(data_ite)):
                data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
            # data_ite, label_ite = shuffle(data_ite, label_ite, random_state=0)
            for i in range(len(data_ite)):
                data_ite[i] = utils.normalization(data_ite[i])
                # data_ite[i] = utils.norm_with_range(data_ite[i], cd_data_min, cd_data_max)
        s_data_all, s_label_all = utils.stack_list(data_ite, label_ite)
        number_of_data = s_label_all.shape[0]
        temp_array = list(range(number_of_data))
        number_to_sample = 1500 if len(temp_array)<2000 else 2500
        # number_to_sample = 1000
        temp_index = random.sample(temp_array, number_to_sample)
        new_data_all = np.array([s_data_all[i] for i in temp_index])
        new_label_all = np.array([s_label_all[i] for i in temp_index])

        start_time = time.time()
        coral = CORAL()
        acc, ypre = coral.fit_predict(new_data_all, new_label_all.squeeze(), cd_data, cd_label.squeeze(), ud_data, ud_label.squeeze())
        coral_time = time.time() - start_time
        times.append(coral_time)
        accs.append(acc)
    print("Time: ", np.mean(times))
    print("Accs: ", np.mean(accs), np.std(accs))

if __name__ == '__main__':
    FOIT_type_all = ['cross-all', 'cross-session', 'cross-subject']
    dataset_name_all = ['seed4', 'seed3']
    # FOIT_type_all = ['cross-all']
    # dataset_name_all = ['seed4']
    for dataset_name in dataset_name_all:
        print('Dataset name: {}'.format(dataset_name))
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        for FOIT_type in FOIT_type_all:
            print('FOIT type: {}'.format(FOIT_type))
            testCORAL(dataset_name=dataset_name, FOIT_type=FOIT_type)

    # data, label = utils.load_session_data_label('seed4', 0) # as unlabelled data
    # cd_data, cd_label, ud_data, ud_label = utils.pick_one_data('seed4', session_id=1, cd_count=16, sub_id=0)
    # test_data = np.vstack((cd_data, ud_data))
    # test_label = np.vstack((cd_label, ud_label))
    # test_data = utils.normalization(test_data)
    # # cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
    # # ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
    # # cd_data_min, cd_data_max = np.min(cd_data), np.max(cd_data)
    # # cd_data = utils.normalization(cd_data) # labelled data
    # # ud_data = utils.normalization(ud_data) # test data
    # data_ite, label_ite = data.copy(), label.copy()
    # for i in range(len(data)):
    #     data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
    # for i in range(len(data)):
    #     data_ite[i] = utils.normalization(data_ite[i])
    # s_data_all, s_label_all = utils.stack_list(data_ite, label_ite)

    # number_of_data = s_label_all.shape[0]
    # temp_array = list(range(number_of_data))
    # temp_index = random.sample(temp_array, 5000)
    # new_data_all = np.array([s_data_all[i] for i in temp_index])
    # new_label_all = np.array([s_label_all[i] for i in temp_index])

    # start_time = time.time()
    # CORAL = CORAL()
    # acc, ypre = CORAL.fit_predict(new_data_all, new_label_all.squeeze(), test_data, test_label.squeeze())
    # # acc, ypre = CORAL.fit_predict(s_data_all, s_label_all.squeeze(), test_data, test_label.squeeze())
    # coral_time = time.time() - start_time
    # print(acc)
    # print(coral_time)

    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # for i in range(4):
    #     for j in range(4):
    #         if i != j:
    #             src, tar = 'data/' + domains[i], 'data/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #             Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    #             coral = CORAL()
    #             acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
    #             print(acc)