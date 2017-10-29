#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 20:41:25 2017

@author: fs
"""

from time import time
from PIL import Image
import glob
import numpy as np
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


PICTURE_PATH = "att_faces"

all_data_set = [] #原始总数据集，二维矩阵n*m，n个样例，m个属性
all_data_label = [] #总数据对应的类标签
def get_picture():
    label = 1
   #读取所有图片并一维化
    while (label <= 40):
        for name in glob.glob(PICTURE_PATH + "/s" + str(label) + "/*.pgm"):
            img = Image.open(name)
            #img.getdata()
            #np.array(img).reshape(1, 92*112)
            all_data_set.append( list(img.getdata()) )
            all_data_label.append(label)
        label += 1

get_picture()

n_components = 16#这个降维后的特征值个数如果太大，比如100，结果将极其不准确，为何？？
pca = PCA(n_components = n_components, svd_solver='auto', whiten=True).fit(all_data_set)
#PCA降维后的总数据集
all_data_pca = pca.transform(all_data_set)
#X为降维后的数据，y是对应类标签
X = np.array(all_data_pca)
y = np.array(all_data_label)

#输入核函数名称和参数gamma值，返回SVM训练十折交叉验证的准确率
def SVM(kernel_name, param):
#十折交叉验证计算出平均准确率
#n_splits交叉验证，随机取
    kf = KFold(n_splits=5, shuffle = True)
    precision_average = 0.0
    param_grid = {'C': [1 ,1e3, 5e3, 1e4, 5e4, 1e5]}#自动穷举出最优的C参数
    clf = GridSearchCV(SVC(kernel=kernel_name, class_weight='balanced', gamma = param),param_grid)
    precisions = []
    for train, test in kf.split(X):

        clf = clf.fit(X[train], y[train])
            #print(clf.best_estimator_)
        test_pred = clf.predict(X[test])
        #print classification_report(y[test], test_pred)
        #计算平均准确率

        precision = np.mean(np.equal(y[test],test_pred))
        precisions.append(precision)

    precision_average = sum(precisions) / len(precisions)
   #print (u"准确率为" + str(precision_average))
    return precision_average

t0 = time()
kernel_to_test = ['rbf', 'poly', 'sigmoid']
 #rint SVM(kernel_to_test[0], 0.1)
plt.figure(1)

for kernel_name in kernel_to_test:
    x_label = np.linspace(0.001, 1, 10)
    y_label = []
    for i in x_label:
        y_label.append(SVM(kernel_name, i))

    plt.plot(x_label, y_label, label=kernel_name)


print("done in %0.3fs" % (time() - t0))
plt.xlabel("Gamma")
plt.ylabel("Precision")
plt.title('Different Kernels Contrust')
plt.legend()
plt.show()
