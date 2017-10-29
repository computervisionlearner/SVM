#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 15:56:17 2017

@author: fs
"""

from time import time
import cv2
import glob
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import random


def get_datasets(PICTURE_PATH, classes =40):
    all_data_set = [] #原始总数据集，二维矩阵n*m，n个样例，m个属性
    all_data_label = [] #总数据对应的类标签
    for label in range(classes):
        folder = os.path.join(PICTURE_PATH,'s%d'%(label+1))
        all_paths = glob.glob(os.path.join(folder,'*.pgm'))
        pictures =[ cv2.imread(path,0).flatten() for path in all_paths ]

        all_data_set.extend(pictures)
        all_data_label.extend([label]*len(all_paths))

    pca = PCA(n_components = 50, svd_solver='auto', whiten=True).fit(all_data_set)
    all_data_pca = pca.transform(all_data_set)
    return all_data_pca, np.array(all_data_label)
#输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def cross_SVM(kernel_name):
    X, y = get_datasets("att_faces")

    kf = KFold(n_splits=9, shuffle = True)
    clf = SVC(kernel=kernel_name)
    precisions = []
    for train, test in kf.split(X):
        clf = clf.fit(X[train], y[train])
        test_pred = clf.predict(X[test])
        precision = np.mean(np.equal(y[test],test_pred))
        precisions.append(precision)
    precision_average = sum(precisions) / len(precisions)
    joblib.dump(clf,kernel_name+'_svm.pkl')
    return precision_average

def compare_svms():
    t0 = time()
    kernel_to_test = ['rbf', 'poly', 'sigmoid', 'linear']

    for kernel_name in kernel_to_test:
        x_label = np.linspace(0.001, 1, 10)
        for i in x_label:
            y_label = [cross_SVM(kernel_name)]*len(x_label)
        plt.plot(x_label, y_label, label=kernel_name)
    print("done in %0.3fs" % (time() - t0))
    plt.xlabel("Gamma")
    plt.ylabel("Precision")
    plt.title('Different Kernels Contrust')
    plt.legend()
    plt.show()


def train_linear_svm():
    X, y = get_datasets("att_faces")
    shuffle = np.random.permutation(400)
    X_train,y_train = X[shuffle[:360]], y[shuffle[:360]]
    X_test, y_test = X[shuffle[360:]], y[shuffle[360:]]

    t0 = time()
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    joblib.dump(clf,'linearsvm_svm.pkl')
    y_pre = clf.predict(X_test)
    precision = np.mean(np.equal(y_test,y_pre))

    print('the precision is: ',precision,'it costs time '+str(time()-t0))

def test_linear_svm():
    X, y = get_datasets("att_faces")
    liner_svms = joblib.load('linearsvm_svm.pkl')
    y_pre = liner_svms.predict(X)
    precision = np.mean(np.equal(y,y_pre))
    print(precision)


if __name__ == '__main__':
    train_linear_svm()
    test_linear_svm()
    compare_svms()





