# -*- coding: utf-8 -*-

# 1：ライブラリのインポート--------------------------------
import numpy as np #numpyという行列などを扱うライブラリを利用
import pandas as pd #pandasというデータ分析ライブラリを利用
import matplotlib.pyplot as plt #プロット用のライブラリを利用
from sklearn import svm, metrics, preprocessing #, cross_validation #機械学習用のライブラリを利用
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# 2：データ読み込み--------------
data = np.loadtxt("BostonHousing.csv", delimiter=",",encoding="utf-8_sig")

# 一列目が0.5以上か0.5以下で分類する。
data[data[:,0]>0.5, 0] = 1
data[data[:,0]<=0.5, 0] = 0

 
 
# 4：データの整形-------------------------------------------------------
X= data[:,1:]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

X_std = X_train.std(axis=0)
X_mean = X_train.mean(axis=0)

X_train_stdized = ( X_train-X_mean ) / X_std
X_test_stdized = ( X_test-X_mean ) / X_std
 
# SVM---------------------------------------------------
model = SVC(kernel='rbf', gamma=1/2 , C=1.0,class_weight='balanced', random_state=0) #loss='squared_hinge' #loss="hinge", loss="log"
# RF---------------------------------------------------
# model = RandomForestClassifier(n_estimators=100, max_depth=2)


# trainデータでの学習
model.fit(X_train_stdized, y_train)
# 予測
pred_test = model.predict(X_test_stdized)
test_accuracy = model.score(X_test_stdized, y_test)
## 次のようにしてもよい
## test_accuracy = metrics.accuracy_score(y_test, pred_test)

print(metrics.accuracy_score(y_test, pred_test))

print('Accuracy: %.2f' % test_accuracy)

###################
## k交差検証を用いたハイパーパラメータチューニングを行なう場合
best_score = 0
best_parameters  = {}
# ハイパーパラメータの選択肢をリストとして与える
param_list = [0.001, 0.01, 0.1, 1, 10, 100]
kernel_list = ["linear", "poly", "rbf"]

## SVMの場合
for gamma in param_list:
    for C in param_list:
        for kernel in kernel_list:
            model = SVC(gamma=gamma, C=C, kernel=kernel)
            # cross validation
            scores = cross_val_score(model, X_train_stdized, y_train, cv=5)
            # k 個の評価値の平均を用いる
            score = np.mean(scores)
            #print(score, gamma, C, kernel)
            if score > best_score:
                best_score = score
                best_parameters = {'gamma' : gamma, 'C' : C, 'kernel' : kernel}

model = SVC(**best_parameters)
# best_parameters を使って，training set + validation set に対してモデルを作成する。
model.fit(X_train_stdized, y_train)
# 最後にtest setを使って誤差評価。

test_score = model.score(X_test_stdized, y_test)

print('Best score on validation set: {}'.format(best_score))
print('Best parameters: {}'.format(best_parameters))
print('Test set score with best parameters: {}'.format(test_score))

pred_test =model.predict(X_test_stdized) 
cm = confusion_matrix(y_test, pred_test)
tn, fp, fn, tp = cm.flatten()
print("------","Pos", "Neg")
print("True :",tp, fn)
print("False:", fp, tn)
print("Accuracy", metrics.accuracy_score(y_test, pred_test))
print("Precision", metrics.precision_score(y_test, pred_test))
print("Recall", metrics.recall_score(y_test, pred_test))
print("F1 score", metrics.f1_score(y_test, pred_test))
print("Report", metrics.classification_report(y_test, pred_test))

#判別閾値の表示
print(model.decision_function(X_test_stdized))
## RFの場合
'''
n_est_list = [10, 100, 300, 1000]
max_depth_list = [1,2,3,4,5]
for n_estimators in n_est_list:
    for max_depth in max_depth_list:
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
            # cross validation
            scores = cross_val_score(model, X_train_stdized, y_train, cv=5)
            # k 個の評価値の平均を用いる
            score = np.mean(scores)
            #print(score, gamma, C, kernel)
            if score > best_score:
                best_score = score
                best_parameters = {'gamma' : gamma, 'C' : C, 'kernel' : kernel}

model = SVC(**best_parameters)
# best_parameters を使って，training set + validation set に対してモデルを作成する。
model.fit(X_train_stdized, y_train)
# 最後にtest setを使って誤差評価。
test_score = model.score(X_test_stdized, y_test)

print('Best score on validation set: {}'.format(best_score))
print('Best parameters: {}'.format(best_parameters))
print('Test set score with best parameters: {}'.format(test_score))
'''

