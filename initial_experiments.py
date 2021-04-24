#!/usr/bin/env python
# coding: utf-8

import json
# %%
# In[2]:
import os

import numpy as np
# import pandas as pd
# from nltk.stem.porter import PorterStemmer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC

from config import CONFIG_BY_KEY
from data_loader import DataHelper, DataLoader

# import re
# import time

# %%
# In[6]:

RESULT_FILE = "./output/{}.json"

# %%
# In[7]:


def trainIO(train_index, test_index):

    # Prepare data
    train_input, train_output = data.getSplit(train_index)

    test_input, test_output = data.getSplit(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output,
                            config, data)

    train_input = np.empty((len(train_input), 0))
    test_input = np.empty((len(test_input), 0))

    if config.use_target_text:

        if config.use_bert:
            train_input = np.concatenate(
                [train_input,
                 datahelper.getTargetBertFeatures(mode='train')],
                axis=1)
            test_input = np.concatenate(
                [test_input,
                 datahelper.getTargetBertFeatures(mode='test')],
                axis=1)
        else:
            train_input = np.concatenate([
                train_input,
                np.array([
                    datahelper.pool_text(utt)
                    for utt in datahelper.vectorizeUtterance(mode='train')
                ])
            ],
                                         axis=1)
            test_input = np.concatenate([
                test_input,
                np.array([
                    datahelper.pool_text(utt)
                    for utt in datahelper.vectorizeUtterance(mode='test')
                ])
            ],
                                        axis=1)

    if config.use_target_video:
        train_input = np.concatenate(
            [train_input,
             datahelper.getTargetVideoPool(mode='train')], axis=1)
        test_input = np.concatenate(
            [test_input,
             datahelper.getTargetVideoPool(mode='test')], axis=1)

    if config.use_target_audio:
        train_input = np.concatenate(
            [train_input,
             datahelper.getTargetAudioPool(mode='train')], axis=1)
        test_input = np.concatenate(
            [test_input,
             datahelper.getTargetAudioPool(mode='test')], axis=1)

    if train_input.shape[1] == 0:
        print("Invalid modalities")
        exit(1)

    # Aux input

    if config.use_author:
        train_input_author = datahelper.getAuthor(mode="train")
        test_input_author = datahelper.getAuthor(mode="test")

        train_input = np.concatenate([train_input, train_input_author], axis=1)
        test_input = np.concatenate([test_input, test_input_author], axis=1)

    if config.use_context:
        if config.use_bert:
            train_input_context = datahelper.getContextBertFeatures(
                mode="train")
            test_input_context = datahelper.getContextBertFeatures(mode="test")
        else:
            train_input_context = datahelper.getContextPool(mode="train")
            test_input_context = datahelper.getContextPool(mode="test")

        train_input = np.concatenate([train_input, train_input_context],
                                     axis=1)
        test_input = np.concatenate([test_input, test_input_context], axis=1)

    train_output = datahelper.oneHotOutput(mode="train",
                                           size=config.num_classes)
    test_output = datahelper.oneHotOutput(mode="test", size=config.num_classes)

    return train_input, train_output, test_input, test_output


# %%
# In[8]:

config = CONFIG_BY_KEY['tav']
data = DataLoader(config)

# %%

# print(data.data_input[0])
results = []
print('test')
train_input, train_output, test_input, test_output = (None, ) * 4
for fold, (train_index, test_index) in enumerate(data.getStratifiedKFold()):
    config.fold = fold + 1
    print("Present Fold: {}".format(config.fold))

    train_input, train_output, test_input, test_output = trainIO(
        train_index, test_index)

    # print(train_input.shape)
    # print(train_output.shape)

    # print(test_input)
    # print(test_output)
    break
    # clf = svm_train(train_input, train_output)
    # result_dict, result_str = svm_test(clf, test_input, test_output)

    # results.append(result_dict)

# %%

# print(data.data_input[0])
results = []
print('complete test, Gaussian Naive Bayes')
train_input, train_output, test_input, test_output = (None, ) * 4
for fold, (train_index, test_index) in enumerate(data.getStratifiedKFold()):
    config.fold = fold + 1
    print("Present Fold: {}".format(config.fold))

    train_input, train_output, test_input, test_output = trainIO(
        train_index, test_index)

    y_true = test_output[:, 1].astype(int)

    # Using Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb.fit(train_input, train_output[:, 1])
    y_pred = gnb.predict(test_input)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    # return classification_report(y_true, y_pred, output_dict=True, digits=3)
    # ,result_string
    # print(gnb.score(train_input, train_output))  # 66.84
    # print(gnb.score(test_input, test_output))  # 64.49

    # result_dict, result_str = gnb.score(test_input, test_output)
# clf = svm_train(train_input, train_output)
# result_dict, result_str = svm_test(clf, test_input, test_output)

# results.append(result_dict)

# %%
# In[10]:

train_output = train_output[:, 1]
test_output = test_output[:, 1]

# %%
# In[11]:

# model 1:-
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(train_input, train_output)
# getting the score of train and test data
print(lsvc.score(train_input, train_output))  # 56.15 Failed to converge
print(lsvc.score(test_input, test_output))  # 53.62

# %%
# In[12]:

# model 2:-
# Using Gaussuan Naive Bayes
gnb = GaussianNB()
gnb.fit(train_input, train_output)
print(gnb.score(train_input, train_output))  # 66.84
print(gnb.score(test_input, test_output))  # 64.49

# %%
# In[13]:

# model 3:-
# Logistic Regression
lr = LogisticRegression()
lr.fit(train_input, train_output)
print(lr.score(train_input, train_output))  # 100
print(lr.score(test_input, test_output))  # 76.08

# %%
# In[14]:

# Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=0)
rfc.fit(train_input, train_output)
print(rfc.score(train_input, train_output))  # 98.91
print(rfc.score(test_input, test_output))  # 69.56

# %%
# In[15]:


def svm_train(train_input, train_output):
    clf = make_pipeline(
        StandardScaler() if config.svm_scale else FunctionTransformer(
            lambda x: x, validate=False),
        svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf'))

    return clf.fit(train_input, np.argmax(train_output, axis=1))


# %%
# In[16]:


def svm_test(clf, test_input, test_output):

    probas = clf.predict(test_input)
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


# %%
# In[17]:

model_name = 'tav'
results = []
for fold, (train_index, test_index) in enumerate(data.getStratifiedKFold()):

    # Present fold
    config.fold = fold + 1
    print("Present Fold: {}".format(config.fold))

    train_input, train_output, test_input, test_output = trainIO(
        train_index, test_index)

    clf = svm_train(train_input, train_output)
    result_dict, result_str = svm_test(clf, test_input, test_output)

    results.append(result_dict)

# Dumping result to output
if not os.path.exists(os.path.dirname(RESULT_FILE)):
    os.makedirs(os.path.dirname(RESULT_FILE))
with open(RESULT_FILE.format(model_name), 'w') as file:
    json.dump(results, file)

# %%
# In[]:
