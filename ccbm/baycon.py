import json

import pandas as pd
import torch
import sys
import os
import baycon.baycon.bayesian_generator as baycon
from baycon.common.DataAnalyzer import *
from baycon.common.Target import Target
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


def execute(X, Y, model, feature_names, target, initial_instance, initial_prediction, categorical_features=[], actionable_features=[]):
    data_analyzer = DataAnalyzer(X, Y, feature_names, target, categorical_features, actionable_features)
    X, Y = data_analyzer.data()
    counterfactuals, ranker, best_instance = baycon.run(initial_instance, initial_prediction, target, data_analyzer, model)
    #append counterfactuals and best instance
    counterfactuals = np.array(counterfactuals)
    counterfactuals = np.concatenate((counterfactuals, np.array([best_instance])), axis=0)
    predictions = np.array([])
    try:
        predictions = model(torch.Tensor(counterfactuals), concepts=True)
        predictions = predictions[1].argmax(dim=-1).detach().numpy()

    except ValueError:
        pass
    return counterfactuals, best_instance, predictions, initial_instance, initial_prediction, data_analyzer, ranker, model

def sample_different_label(y, n_tasks, already_seen=set()):
        not_y = []
        if n_tasks == 1:
            return 1 - y
        else:
            for el in y:
                possible_labels = list(range(n_tasks))
                for el2 in already_seen:
                    possible_labels.remove(el2)
                if el in possible_labels:
                    possible_labels.remove(el)
                if len(possible_labels) == 0:
                    possible_labels = list(range(n_tasks))
                    possible_labels.remove(el)
                not_y_tmp = np.random.choice(possible_labels)
                not_y.append(not_y_tmp)
                already_seen.add(not_y_tmp)
            return np.array(not_y)

def evaluate_baycon(
        model, 
        train_dl,
        test_dl,
        accelerator,
        n_tasks
):
    not_y_list = []
    c_list = []
    y_list = []
    y_preds_list = []
    # populate train set
    for x, c_true, y_true in train_dl:
        x = x.to(accelerator)
        y_numpy = y_true.detach().argmax(dim=-1).numpy()
        outputs = model(x)
        c_preds, _ = outputs[0], outputs[1].detach().numpy()
        # Warning: should concepts be binary
        if not model.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
        c_bool = c_preds.detach().numpy()
        c_list.append(c_bool)
        y_list.append(y_numpy)
    c_list = np.concatenate(c_list, axis=0)
    c_bool = pd.DataFrame(c_list)
    feature_names = c_bool.columns
    c_bool = c_bool.values
    y_list = np.concatenate(y_list, axis=0)
    y_numpy = pd.DataFrame(y_list).values
    num_classes = len(np.unique(y_numpy))
    # populate test set
    c_list = []
    label_seen = set()
    for x, c_true, y_true in test_dl:
        x = x.to(accelerator)
        y_numpy_tmp = y_true.detach().argmax(dim=-1).numpy()
        not_y = sample_different_label(y_numpy_tmp, num_classes, label_seen)
        label_seen.add(not_y[0])
        print(label_seen)
        outputs = model(x)
        c_preds, y_preds = outputs[0], outputs[1]
        # Warning: should concepts be binary
        if not model.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
        c_bool_tmp = c_preds.float().detach().numpy()
        y_preds = y_preds.argmax(dim=-1).detach().numpy()
        not_y_list.append(not_y)
        c_list.append(c_bool_tmp)
        y_preds_list.append(y_preds)
    c_list = np.concatenate(c_list, axis=0)
    not_y_list = np.concatenate(not_y_list, axis=0)
    not_y = pd.DataFrame(not_y_list).values
    c_bool_test = pd.DataFrame(c_list)
    c_bool_test = c_bool_test.values
    y_preds_list = np.concatenate(y_preds_list, axis=0)
    y_preds = pd.DataFrame(y_preds_list).values

    
    all_cf_preds = []
    all_best_cf = []
    y_prime_pred = []
    y_prime = []
    cf_correct = 0
    total = 0
    total_time = 0
    hamming_dist = 0
    euclidean_dist = 0
    for i in tqdm(range(len(not_y))):
        t = Target(target_type="classification", target_feature="y", target_value=int(not_y[i]))
        start = time.time()
        counterfactuals, best_cf, predictions, initial_instance, initial_prediction, data_analyzer, ranker, model = execute((c_bool > 0.5).astype(int), y_numpy,
                                                                                                                            model, feature_names, t,
                                                                                                                            (c_bool_test[i] > 0.5).astype(int), y_preds[i])
        total_time += time.time() - start
        all_cf_preds.append(counterfactuals)
        # take the counterfactuals with the lower hamming distance to the initial instance
        if len(counterfactuals) != 0:
            h = np.sum(np.abs((counterfactuals > 0.5).astype(float) - (initial_instance > 0.5).astype(float)), axis=1)
            best_cf = counterfactuals[np.argmin(h)]
            predictions = predictions[np.argmin(h)]
        else:
            print('empty')
        if len(best_cf) != 0:
            all_best_cf.append([best_cf])
        else:
            all_best_cf.append([np.array([-1,-1])])
            print('Best not found')
        if (predictions == not_y[i]):
            cf_correct += 1
        y_prime_pred.append(np.expand_dims(np.eye(num_classes)[predictions], axis=0))
        y_prime.append(np.eye(num_classes)[not_y[i]])
        total += 1 
        hamming_dist += np.abs((best_cf > 0.5).astype(float) - (c_bool_test[i] > 0.5).astype(float)).sum()
        euclidean_dist += np.linalg.norm(best_cf - c_bool_test[i])
    all_cf_preds = np.concatenate(all_cf_preds, axis=0)
    all_cf_preds = torch.Tensor(all_cf_preds).squeeze(dim=0)
    all_best_cf = np.concatenate(all_best_cf, axis=0)
    all_best_cf = torch.Tensor(np.array(all_best_cf))
    cf_accuracy = cf_correct/total
    print(y_prime_pred)
    y_prime_pred = np.concatenate(y_prime_pred, axis=0)
    y_prime_pred = torch.Tensor(np.array(y_prime_pred))
    print(y_prime_pred.shape)
    print(y_prime)
    y_prime = np.concatenate(y_prime, axis=0)
    y_prime = torch.Tensor(np.array(y_prime))
    print(y_prime.shape)
    cf_roc = roc_auc_score(y_prime, y_prime_pred)
    hamming_dist = hamming_dist/total
    euclidean_dist = euclidean_dist/total
    return total_time, all_cf_preds, all_best_cf, y_prime_pred, cf_accuracy, cf_roc, not_y, hamming_dist, euclidean_dist