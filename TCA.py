import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
from sklearn.neighbors import KNeighborsClassifier

# data
import time
import utils
import random
random.seed(0) # temporary
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.calibration import CalibratedClassifierCV

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = np.dot(A.T, K).astype(float)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        # clf = KNeighborsClassifier(n_neighbors=1)
        clf = svm.LinearSVC(max_iter=30000)
        clf = CalibratedClassifierCV(clf, cv=5)
        clf.fit(Xs_new, Ys.ravel())
        # Z = np.dot(A.T, K)
        # Z /= np.linalg.norm(Z, axis=0)
        # Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        y_pred = clf.predict(Xt_new)
        acc = sklearn.metrics.accuracy_score(Yt.squeeze(), y_pred)
        return acc, y_pred

def testTCA(dataset_name='seed4', FOIT_type='cross-all'):
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
        tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
        acc, ypre = tca.fit_predict(new_data_all, new_label_all.squeeze(), ud_data, ud_label.squeeze())
        tca_time = time.time() - start_time
        times.append(tca_time)
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
            testTCA(dataset_name=dataset_name, FOIT_type=FOIT_type)

     



    # data, label = utils.load_session_data_label('seed4', 0) # as unlabelled data
    # cd_data, cd_label, ud_data, ud_label = utils.pick_one_data('seed4', session_id=1, cd_count=16, sub_id=2)
    # test_data = np.vstack((cd_data, ud_data))
    # test_label = np.vstack((cd_label, ud_label))
    # test_data = utils.normalization(test_data)
    # # cd_data, cd_label = shuffle(cd_data, cd_label, random_state=0)
    # # ud_data, ud_label = shuffle(ud_data, ud_label, random_state=0)
    # # cd_data_min, cd_data_max = np.min(cd_data), np.max(cd_data)
    # cd_data = utils.normalization(cd_data) # labelled data
    # ud_data = utils.normalization(ud_data) # test data
    # data_ite, label_ite = data.copy(), label.copy()
    # for i in range(len(data)):
    #     data_ite[i], label_ite[i] = shuffle(data_ite[i], label_ite[i], random_state=0)
    # for i in range(len(data)):
    #     data_ite[i] = utils.normalization(data_ite[i])
    # s_data_all, s_label_all = utils.stack_list(data_ite, label_ite)

    # number_of_data = s_label_all.shape[0]
    # temp_array = list(range(number_of_data))
    # temp_index = random.sample(temp_array, 1000)
    # new_data_all = np.array([s_data_all[i] for i in temp_index])
    # new_label_all = np.array([s_label_all[i] for i in temp_index])

    # start_time = time.time()
    # tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    # acc, ypre = tca.fit_predict(new_data_all, new_label_all.squeeze(), ud_data, ud_label.squeeze())
    # tca_time = time.time() - start_time
    # print(acc)
    # print(tca_time)

    # domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    # for i in [2]:
    #     for j in [3]:
    #         if i != j:
    #             src, tar = 'data/' + domains[i], 'data/' + domains[j]
    #             src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
    #             Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']
    #             tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    #             acc, ypre = tca.fit_predict(Xs, Ys, Xt, Yt)
    #             print(acc)
    #             # It should print 0.910828025477707