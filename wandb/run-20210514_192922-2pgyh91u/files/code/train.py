import argparse
import json
import os
import re
# import sys
# from scipy.sparse.construct import random
# import IPython
# import ipdb
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC
# from fastai.vision import Learner
# from fastai.vision import *
# from fastai.vision.all import *
from fastai.vision.learner import Learner
import torch
import torch.nn.functional as F
from torch import nn
import wandb

import config
from config import CONFIG_BY_KEY
from data_loader import DataLoader
from data_loader import DataHelper

# breakpoint()
# %%wandb
wandb.login()
WANDB_NOTEBOOK_NAME = 'train.py'


class FullyConnectedNN(nn.Module):
    def __init__(self):
        # call constructor from superclass
        super().__init__()

        # define network layers
        self.fc1 = nn.Linear(3099, 250, bias=True)
        self.fc2 = nn.Linear(250, 2, bias=True)
        # self.fc3 = nn.Linear(120, 2, bias=True)

    def forward(self, xb):
        # define forward pass
        x = xb.float()
        # x = xb.view(250, -1) resize?
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc2(x))
        # self.lin3(x)
        return x


def fcnn_train(train_input, train_output):
    clf = FullyConnectedNN()
    # print(clf)
    fcnn_learner = Learner(data=data,
                           model=clf,
                           loss_func=nn.CrossEntropyLoss(),
                           metrics=accuracy)
    fcnn_learner.fit_one_cycle(5, 1e-2)


def lsvc_train(train_input, train_output):
    # clf = make_pipeline(
    # StandardScaler(),
    # svm.SVC(C=config.svm_c, gamma='scale', kernel='rbf'))
    clf = LinearSVC(C=run_config.lsvc_c, max_iter=run_config.lsvc_max_iter)
    # print(sys.executable)
    return clf.fit(train_input, train_output[:, 1].astype(int))


def lsvc_test(clf, test_input, test_output):

    y_pred = clf.predict(test_input)
    y_true = test_output[:, 1].astype(int)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    # print(confusion_matrix(y_true, y_pred))
    # print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


def lr_train(train_input, train_output):
    # lr = LogisticRegression(solver='saga',
    #                         max_iter=10000,
    #                         penalty='elasticnet',
    #                         l1_ratio=1)
    lr = LogisticRegression()
    return lr.fit(train_input, train_output[:, 1].astype(int))


def lr_test(clf, test_input, test_output):

    y_pred = clf.predict(test_input)
    y_true = test_output[:, 1].astype(int)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    print(confusion_matrix(y_true, y_pred))
    print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


def svm_train(train_input, train_output):
    clf = make_pipeline(
        StandardScaler() if run_config.svm_scale else FunctionTransformer(
            lambda x: x, validate=False),
        svm.SVC(C=run_config.svm_c,
                gamma=run_config.svm_gamma,
                kernel=run_config.svm_kernel))

    return clf.fit(train_input, np.argmax(train_output, axis=1))


def svm_test(clf, test_input, test_output):

    probas = clf.predict(test_input)
    y_pred = probas
    y_true = np.argmax(test_output, axis=1)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    # print(confusion_matrix(y_true, y_pred))
    # print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


def gauss_train(train_input, train_output):
    gnb = GaussianNB()
    return gnb.fit(train_input, train_output[:, 1].astype(int))


def gauss_test(clf, test_input, test_output):

    y_pred = clf.predict(test_input)
    y_true = test_output[:, 1].astype(int)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    # print(confusion_matrix(y_true, y_pred))
    # print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


def rfc_train(train_input, train_output):
    rfc = RandomForestClassifier(n_estimators=10, random_state=0)
    return rfc.fit(train_input, train_output[:, 1].astype(int))


def rfc_test(clf, test_input, test_output):

    y_pred = clf.predict(test_input)
    y_true = test_output[:, 1].astype(int)

    # To generate random scores
    # y_pred = np.random.randint(2, size=len(y_pred))

    # To generate majority baseline
    # y_pred = [0]*len(y_pred)

    result_string = classification_report(y_true, y_pred, digits=3)
    # print(confusion_matrix(y_true, y_pred))
    # print(result_string)
    return classification_report(y_true, y_pred, output_dict=True,
                                 digits=3), result_string


def trainIO(train_index, test_index):

    # Prepare data
    train_input, train_output = data.getSplit(train_index)
    test_input, test_output = data.getSplit(test_index)

    datahelper = DataHelper(train_input, train_output, test_input, test_output,
                            run_config, data)

    train_input = np.empty((len(train_input), 0))
    test_input = np.empty((len(test_input), 0))

    if run_config.use_target_text:

        if run_config.use_bert:
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

    if run_config.use_target_video:
        train_input = np.concatenate(
            [train_input,
             datahelper.getTargetVideoPool(mode='train')], axis=1)
        test_input = np.concatenate(
            [test_input,
             datahelper.getTargetVideoPool(mode='test')], axis=1)

    if run_config.use_target_audio:
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

    if run_config.use_author:
        train_input_author = datahelper.getAuthor(mode="train")
        test_input_author = datahelper.getAuthor(mode="test")

        train_input = np.concatenate([train_input, train_input_author], axis=1)
        test_input = np.concatenate([test_input, test_input_author], axis=1)

    if run_config.use_context:
        if run_config.use_bert:
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
                                           size=run_config.num_classes)
    test_output = datahelper.oneHotOutput(mode="test",
                                          size=run_config.num_classes)

    return train_input, train_output, test_input, test_output


def trainSpeakerIndependent(model_name=None):

    run_config.fold = "SI"

    (train_index, test_index) = data.getSpeakerIndependent()
    train_input, train_output, test_input, test_output = trainIO(
        train_index, test_index)

    train_func = CLF_MAP[args.clf][0]
    test_func = CLF_MAP[args.clf][1]
    clf = train_func(train_input, train_output)
    test_func(clf, test_input, test_output)


def trainSpeakerDependent(model_name=None):

    print(vars(run_config))

    config_params = {
        k: v
        for k, v in config.Config.__dict__.items()
        if not (k.startswith('__') and k.endswith('__'))
    }
    # IPython.embed()
    # breakpoint()
    wandb.init(config=config_params, project="multimodal-sarcasm")
    wandb.config.update({"config_key": args.config_key})
    wandb.run.name = run_config.run_name + re.sub(r'^.*?-', '-',
                                                  wandb.run.name)
    # breakpoint()
    # wandb.config.svm_c=config.svm_c

    # Load data
    data = DataLoader(run_config)
    # breakpoint()
    # labels = ['Non-Sarcastic', 'Sarcastic']

    # Iterating over each fold
    results = []
    for fold, (train_index,
               test_index) in enumerate(data.getStratifiedKFold()):

        # Present fold
        run_config.fold = fold + 1
        print("Present Fold: {}".format(run_config.fold))

        train_input, train_output, test_input, test_output = trainIO(
            train_index, test_index)

        breakpoint()
        train_func = CLF_MAP[run_config.model][0]
        # test_func = CLF_MAP[args.clf][1]
        clf = train_func(train_input, train_output)

        y_pred = clf.predict(test_input)
        # y_probas = clf.predict_proba(test_input)
        y_test = test_output[:, 1].astype(int)
        # y_train = train_output[:, 1].astype(int)
        # importances = clf.feature_importances_
        # indices = np.argsort(importances)[::-1]

        # To generate random scores
        # y_pred = np.random.randint(2, size=len(y_pred))

        # To generate majority baseline
        # y_pred = [0]*len(y_pred)
        # result_str = classification_report(y_true, y_pred, digits=3)
        # print(confusion_matrix(y_true, y_pred))
        # print(result_string)
        result_dict = classification_report(y_test,
                                            y_pred,
                                            output_dict=True,
                                            digits=3)

        results.append(result_dict)
        # wandb.sklearn.plot_classifier(clf,
        #                               train_input,
        #                               test_input,
        #                               y_train,
        #                               y_test,
        #                               y_pred,
        #                               y_probas,
        #                               labels,
        #                               model_name='args.clf',
        #                               feature_names=None)
        #
    # Dumping result to output
    if not os.path.exists(os.path.dirname(RESULT_FILE)):
        os.makedirs(os.path.dirname(RESULT_FILE))
    with open(RESULT_FILE.format(model_name), 'w') as file:
        json.dump(results, file)


def printResult(model_name=None):

    results = json.load(open(RESULT_FILE.format(model_name), "rb"))

    weighted_precision, weighted_recall = [], []
    weighted_fscores = []

    print("#" * 20)
    for fold, result in enumerate(results):
        weighted_fscores.append(result["weighted avg"]["f1-score"])
        weighted_precision.append(result["weighted avg"]["precision"])
        weighted_recall.append(result["weighted avg"]["recall"])

        print("Fold {}:".format(fold + 1))
        print(
            "Weighted Precision: {}  Weighted Recall: {}  Weighted F score: {}"
            .format(result["weighted avg"]["precision"],
                    result["weighted avg"]["recall"],
                    result["weighted avg"]["f1-score"]))
    print("#" * 20)
    print("Avg :")
    precision = np.mean(weighted_precision)
    recall = np.mean(weighted_recall)
    fscore = np.mean(weighted_fscores)

    wandb.log({
        'weighted_precision': precision,
        'weighted_recall': recall,
        'weighted_fscore': fscore
    })

    print("Weighted Precision: {:.3f}  Weighted Recall: {:.3f}\
        Weighted F score: {:.3f}".format(precision, recall, fscore))


CLF_MAP = {
    'lsvc': [lsvc_train, lsvc_test],
    'lr': [lr_train, lr_test],
    'rfc': [rfc_train, rfc_test],
    'gauss': [gauss_train, gauss_test],
    'svm': [svm_train, svm_test],
    'fcnn': [fcnn_train, None]
}

# a=CLF_MAP['lr'][0]

# print(a(train_input, train_output))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-key',
                        default='',
                        choices=list(CONFIG_BY_KEY.keys()))
    # parser.add_argument('clf', choices=list(CLF_MAP.keys()))
    return parser.parse_args()


args = parse_args()
print("Args:", args)

RESULT_FILE = "./output/lsvc{}.json"

# Load config
run_config = CONFIG_BY_KEY[args.config_key]

# Load data
data = DataLoader(run_config)

if __name__ == "__main__":

    if run_config.speaker_independent:
        trainSpeakerIndependent(model_name=run_config.model)
    else:
        for _ in range(run_config.runs):
            trainSpeakerDependent(model_name=run_config.model)
            printResult(model_name=run_config.model)
