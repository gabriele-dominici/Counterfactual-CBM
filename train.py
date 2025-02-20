import torch
import math
import shap
import numpy as np
import pytorch_lightning as pl
from ccbm.utils import randomize_class, save_set_c_and_cf, FEATURE_NAMES, DATASET, CLASS_TO_VISUALISE
from sklearn.metrics import roc_auc_score
from pytorch_lightning import seed_everything
from ccbm.metrics import (variability, 
                          difference_over_union, 
                          intersection_over_union, 
                          cf_in_distribution, distance_train)
from torch.utils.data import TensorDataset
from ccbm.baycon import evaluate_baycon
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def train(net,
          train_dl, test_dl, 
          epochs, device, learning_rate, emb_size, c_cf_set, concept_labels, batch_size,
          log_dir, figures_dir, results_dir, 
          fold, model, seed, wandb_logger):
    if model == 'Oracle':
        return train_oracle(net, test_dl, c_cf_set, concept_labels, wandb_logger)
    elif model == 'DeepNN':
        return train_nn(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, wandb_logger)
    elif model in ['StandardCBM', 'StandardDCR']:
        return train_cbm(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, wandb_logger)
    elif model in ['CFCBM', 'CFDCR']:
        return train_cfcbm(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger)
    elif model in ['VAECF', 'CCHVAE']:
        return train_vae(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, batch_size, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger)
    elif model in ['VCNET']:
        return train_conceptvcnet(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger)
    elif model == 'BayCon':
        return train_baycon(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, device, batch_size, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger)
    else:
        raise ValueError(f'Unknown model {model}')
    
def train_model(model, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger):
    print(f'Running {model}, epochs={epochs}, learning_rate={learning_rate}')
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max", save_weights_only=True)
    trainer = pl.Trainer(max_epochs=epochs,
                            # accumulate_grad_batches=20,
                            # devices=1, accelerator="gpu",
                            enable_checkpointing=True,
                            limit_train_batches=1.0,
                            limit_val_batches=1.0,
                            logger=wandb_logger,
                            callbacks=checkpoint_callback,
                            accelerator=accelerator)
    seed_everything(seed, workers=True)
    
    try:
        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)
    except Exception as e:
        print(e)
    model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
    print(f"Best train acc: {checkpoint_callback.best_model_score}, "
            f"Epoch: {torch.load(checkpoint_callback.best_model_path)['epoch']}")
    return model

def train_oracle(net, test_dl, c_cf_set, concept_labels, wandb_logger):
    task_accuracy = task_cf_accuracy = task_accuracy_perturbed = task_accuracy_int = concept_accuracy = -1
    explanations, explanation_cf = [], []

    c_cf_total = torch.empty(0, c_cf_set.shape[-1])
    c_total = torch.empty(0, c_cf_set.shape[-1])
    cf_time_total = 0
    for _, c_test, y_test in test_dl:
        cf_time, c_cf = net.find_counterfactuals(c_test, y_test, c_cf_set, concept_labels)
        c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)
        cf_time_total += cf_time
    cf_variability = variability(c_cf_total, c_cf_set)
    cf_iou = intersection_over_union(c_cf_total, c_cf_set)
    cf_dou = difference_over_union(c_cf_total, c_cf_set)
    cf_found = 1
    cf_in_pred = 1
    cf_in_train = 1

    pdist = torch.nn.PairwiseDistance(p=2)
    euclidean_distance = pdist(c_total, c_cf_total).mean().item()
    hamming_distance = torch.norm((c_total>0.5).float() - (c_cf_total>0.5).float(), p=0, dim=-1).mean().item() 

    result = {}
    result['task_accuracy'] = task_accuracy
    result['task_cf_accuracy'] = task_cf_accuracy
    result['task_accuracy_perturbed'] = task_accuracy_perturbed
    result['task_accuracy_int'] = task_accuracy_int
    result['concept_accuracy'] = concept_accuracy
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time_total
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['explanations'] = explanations
    result['explanation_cf'] = explanation_cf
    
    return result, net

def train_nn(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, wandb_logger):
    train_dl, val_dl = train_dl
    net = train_model(net, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)
    y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])

    for X_test, _, y_test in test_dl:
        y_preds = net.forward(X_test)
        y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
        y_total = torch.cat((y_total, y_test), dim=0)

    task_accuracy = roc_auc_score(y_total.cpu(), y_preds_total.detach().cpu())

    concept_accuracy = task_cf_accuracy = task_accuracy_perturbed = task_accuracy_int = -1
    explanations, explanation_cf = [], []
    cf_variability = cf_iou = cf_dou = cf_found = cf_time = -1
    euclidean_distance = hamming_distance = -1
    cf_in_pred = cf_in_train = -1
    cf_time_total = -1

    result = {}
    result['task_accuracy'] = task_accuracy
    result['task_cf_accuracy'] = task_cf_accuracy
    result['task_accuracy_perturbed'] = task_accuracy_perturbed
    result['task_accuracy_int'] = task_accuracy_int
    result['concept_accuracy'] = concept_accuracy
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time_total
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['explanations'] = explanations
    result['explanation_cf'] = explanation_cf

    return result, net

def train_cbm(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, wandb_logger):

    #if train is tuple
    if len(train_dl) == 2:
        train_dl, val_dl = train_dl
    else: 
        val_dl = test_dl

    net = train_model(net, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)

    y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])

    for X_test, c_test, y_test in test_dl:

        c_preds, y_preds, _ = net.forward(X_test)

        if not net.bool_concepts:
            c_preds = torch.sigmoid(c_preds)

        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)
        y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
        y_total = torch.cat((y_total, y_test), dim=0)

    concept_accuracy = roc_auc_score(c_total.cpu(), c_preds_total.detach().cpu())
    concept_acc = (c_preds_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
    p_c = (c_preds_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
    task_accuracy = roc_auc_score(y_total.cpu(), y_preds_total.detach().cpu())
    task_cf_accuracy = task_accuracy_perturbed = task_accuracy_int = -1
    explanations = []
    explanation_cf = []
    cf_variability = cf_iou = cf_dou = cf_found = cf_time = -1
    euclidean_distance = hamming_distance = -1
    cf_in_pred = cf_in_train = -1
    cf_time_total = -1

    result = {}
    result['task_accuracy'] = task_accuracy
    result['task_cf_accuracy'] = task_cf_accuracy
    result['task_accuracy_perturbed'] = task_accuracy_perturbed
    result['task_accuracy_int'] = task_accuracy_int
    result['concept_accuracy'] = concept_accuracy
    result['cf_concept_acc_int_0.0'] = concept_acc
    result['p_c'] = p_c
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time_total
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['explanations'] = explanations
    result['explanation_cf'] = explanation_cf

    return result, net

def train_cfcbm(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger):
    train_dl, val_dl = train_dl
    net = train_model(net, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)
    net.actual_resample = net.resample

    c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_cf_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_cf_target_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    c_cf_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    z2_total = torch.empty(0, net.emb_size)
    z3_total = torch.empty(0, net.emb_size)

    for X_test, c_test, y_test in test_dl:
        y_prime = randomize_class(y_test, include=False)
        if y_test.shape[-1] == 1:
            y_prime = None
        (c_preds, y_preds, explanations,
        c_cf, y_cf, y_cf_target, explanation_cf, 
        p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
        pcprime_z3, py_c, py_c_cf, pc_z2, c_cf_true, weights, z2, z3, c_pred_d) = net.forward(X_test, test=True, y_prime=y_prime, explain=True, include=False, inference=True)

        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)
        y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
        y_total = torch.cat((y_total, y_test), dim=0)
        y_cf_total = torch.cat((y_cf_total, y_cf), dim=0)
        y_cf_target_total = torch.cat((y_cf_target_total, y_cf_target), dim=0)
        c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)
        z2_total = torch.cat((z2_total, z2), dim=0)
        z3_total = torch.cat((z3_total, z3), dim=0)

    concept_accuracy = roc_auc_score(c_total.cpu(), c_preds_total.detach().cpu())
    concept_acc = (c_preds_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
    p_c = (c_preds_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
    task_accuracy = roc_auc_score(y_total.cpu(), y_preds_total.detach().cpu())
    task_cf_accuracy = roc_auc_score(y_cf_target_total.squeeze().cpu(), y_cf_total.detach().cpu())
    cf_variability = variability(c_cf_total.cpu(), c_preds_total.cpu())
    cf_iou = intersection_over_union(c_cf_total.cpu(), c_preds_total.cpu())
    cf_dou = difference_over_union(c_cf_total.cpu(), c_preds_total.cpu())
    cf_in_pred = cf_in_distribution(c_cf_total.cpu(), c_preds_total.cpu())
    cf_in_train = cf_in_distribution(c_cf_total.cpu(), c_cf_set.cpu())
    mean_distance_train = distance_train(c_cf_total.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())
    task_accuracy_perturbed, task_accuracy_int = -1, -1

    pdist = torch.nn.PairwiseDistance(p=2)
    euclidean_distance = pdist(c_preds_total.cpu(), c_cf_total.cpu()).mean().item()
    hamming_distance = torch.norm((c_preds_total>0.5).float().cpu() - (c_cf_total>0.5).float().cpu(), p=0, dim=-1).mean().item() 

    cf_time_total, cf_found = net.counterfactual_times(test_dl, accelerator, rerun=False)

    result = {}
    result['task_accuracy'] = task_accuracy
    result['task_cf_accuracy'] = task_cf_accuracy
    result['task_accuracy_perturbed'] = task_accuracy_perturbed
    result['task_accuracy_int'] = task_accuracy_int
    result['concept_accuracy'] = concept_accuracy
    result['cf_concept_acc_int_0.0'] = concept_acc
    result['p_c'] = p_c
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time_total
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['explanations'] = explanations
    result['explanation_cf'] = explanation_cf
    result['mean_distance_train'] = mean_distance_train

    save_set_c_and_cf(c_preds_total, y_preds_total, y_cf_total, c_cf_total, model, fold, log_dir)

    n_times_list = [5, 10, 100]
    for n in n_times_list: 

        c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        y_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        for X_test, c_test, y_test in test_dl:
            y_prime = randomize_class(y_test, include=False)
            (c_preds, y_preds, explanations,
            c_cf, y_cf, y_cf_target, explanation_cf, 
            p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
            pcprime_z3, py_c, py_c_cf, pc_z2, c_cf_true, weights, z2, z3, c_pred_d) = net.forward(X_test, test=True, y_prime=y_prime, explain=True, include=False, resample=n)

            c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
            c_total = torch.cat((c_total, c_test), dim=0)
            y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
            y_total = torch.cat((y_total, y_test), dim=0)

        concept_accuracy = roc_auc_score(c_total.cpu(), c_preds_total.detach().cpu())
        p_c = (c_preds_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
        concept_acc = (c_preds_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
        task_accuracy = roc_auc_score(y_total.cpu(), y_preds_total.detach().cpu())

        result[f'task_accuracy_{n}'] = task_accuracy
        result[f'concept_accuracy_{n}'] = concept_accuracy
        result[f'cf_concept_acc_int_0.0_{n}'] = concept_acc
        result[f'p_c_{n}'] = p_c
            

        cf_time, cf_found, cf, c_preds_total, y_cf_total = net.counterfactual_times(test_dl, accelerator, rerun=False, n_times=n, return_cf=True)
        
        cf_variability = variability(cf.cpu(), c_preds_total.cpu())
        cf_iou = intersection_over_union(cf.cpu(), c_preds_total.cpu())
        cf_dou = difference_over_union(cf.cpu(), c_preds_total.cpu())
        cf_in_pred = cf_in_distribution(cf.cpu(), c_preds_total.cpu())
        cf_in_train = cf_in_distribution(cf.cpu(), c_cf_set.cpu())
        mean_distance_train = distance_train(cf.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())
        euclidean_distance = pdist(c_preds_total.cpu(), cf.cpu()).mean().item()
        hamming_distance = torch.norm((c_preds_total>0.5).float().cpu() - (cf>0.5).float().cpu(), p=0, dim=-1).mean().item() 

        result[f'cf_time_{n}'] = cf_time
        result[f'cf_variability_{n}'] = cf_variability
        result[f'cf_iou_{n}'] = cf_iou
        result[f'cf_dou_{n}'] = cf_dou
        result[f'cf_found_{n}'] = cf_found
        result[f'cf_in_pred_{n}'] = cf_in_pred
        result[f'cf_in_train_{n}'] = cf_in_train
        result[f'mean_distance_train_{n}'] = mean_distance_train
        result[f'euclidean_distance_{n}'] = euclidean_distance
        result[f'hamming_distance_{n}'] = hamming_distance
    
    p_list = [0.1, 0.2, 0.5, 1.0]
    for p in p_list:
        c_preds_total_noise = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        y_prime_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        y_cf_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        c_cf_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        for X_test, c_test, y_test in test_dl:
            y_prime = y_test.clone()
            y_cf_logits, c_cf, y_preds, c_preds = net.predict_counterfactuals(
                X_test, y_prime=y_prime, resample=1, return_cf=True, auto_intervention=p, full=True
            )

            c_preds_total_noise = torch.cat((c_preds_total_noise, c_preds), dim=0)
            c_total = torch.cat((c_total, c_test), dim=0)
            y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
            y_prime_total = torch.cat((y_prime_total, y_test), dim=0)
            y_cf_total = torch.cat((y_cf_total, y_cf_logits), dim=0)
            c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)
        
        task_accuracy = roc_auc_score(y_prime_total.cpu(), y_preds_total.detach().cpu())
        task_accuracy_cf = roc_auc_score(y_prime_total.cpu(), y_cf_total.detach().cpu())
        cf_concept_acc = (c_cf_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
        p_c = (c_cf_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
        hamming_distance = torch.norm((c_preds_total_noise>0.5).float().cpu() - (c_cf_total>0.5).float().cpu(), p=0, dim=-1).mean().item() 
        cf_in_pred = cf_in_distribution(c_cf_total.cpu(), c_preds_total.cpu())
        cf_in_train = cf_in_distribution(c_cf_total.cpu(), c_cf_set.cpu())
        mean_distance_train = distance_train(c_cf_total.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())

        result[f'task_accuracy_int_{p}'] = task_accuracy
        result[f'cf_found_int_{p}'] = task_accuracy_cf
        result[f'cf_in_pred_int_{p}'] = cf_in_pred
        result[f'cf_in_train_int_{p}'] = cf_in_train
        result[f'mean_distance_train_{p}'] = mean_distance_train
        result[f'hamming_distance_int_{p}'] = hamming_distance
        result[f'cf_concept_acc_int_{p}'] = cf_concept_acc
        result[f'p_c_{p}'] = p_c

    return result, net

def train_vae(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, batch_size, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger):

    cbm, vae = net
    train_dl, val_dl = train_dl
    result, cbm = train_cbm(cbm, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)
    
    y_test_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    for X_test, c_test, y_test in train_dl:
        c_preds, y_preds, _ = cbm.forward(X_test)
        if not cbm.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        y_test_total = torch.cat((y_test_total, y_test), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)

    train_data = TensorDataset(c_preds_total, c_total, y_test_total)
    train_dl = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True)

    y_test_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    c_preds_total = torch.empty(0, train_dl.dataset[0][0].shape[-1])
    c_total = torch.empty(0, train_dl.dataset[0][0].shape[-1])
    for X_test, c_test, y_test in test_dl:
        c_preds, y_preds, _ = cbm.forward(X_test)

        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        y_test_total = torch.cat((y_test_total, y_test), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)

    test_data = TensorDataset(c_preds_total, c_total, y_test_total)
    test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=True)

    vae.set_model(cbm.reasoner)

    vae = train_model(vae, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, wandb_logger)

    cf_time, cf_found, c_cf, c_preds, y_cf, y_cf_target = vae.counterfactual_times(test_dl, accelerator, return_cf=True)
    cf_variability = variability(c_cf, c_preds)
    cf_iou = intersection_over_union(c_cf, c_preds)
    cf_dou = difference_over_union(c_cf, c_preds)
    cf_in_pred = cf_in_distribution(c_cf, c_preds)
    cf_in_train = cf_in_distribution(c_cf, c_cf_set)

    pdist = torch.nn.PairwiseDistance(p=2)
    euclidean_distance = pdist(c_preds, c_cf).mean().item()
    hamming_distance = torch.norm((c_preds>0.5).float() - (c_cf>0.5).float(), p=0, dim=-1).mean().item() 
    mean_distance_train = distance_train(c_cf.cpu(), c_cf_set.cpu(), y_cf.detach().cpu(), concept_labels.cpu())

    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['mean_distance_train'] = mean_distance_train

    save_set_c_and_cf(c_preds_total, y_test_total, y_cf, c_cf, model, fold, log_dir)

    p_list = [0.1, 0.2, 0.5, 1.0]
    for p in p_list:
        c_preds_total_noise = torch.empty(0, test_dl.dataset[0][0].shape[-1])
        c_total = torch.empty(0, test_dl.dataset[0][0].shape[-1])
        y_preds_total = torch.empty(0, test_dl.dataset[0][-1].shape[-1])
        y_prime_total = torch.empty(0, test_dl.dataset[0][-1].shape[-1])
        y_cf_total = torch.empty(0, test_dl.dataset[0][-1].shape[-1])
        c_cf_total = torch.empty(0, test_dl.dataset[0][0].shape[-1])
        for c_test, c_true, y_test in test_dl:
            y_prime = y_test.clone()
            output = vae.generate_cf(
                c_test, y_prime=y_prime, auto_intervention=p, c_true=c_true,
            )
            if len(output) == 4: 
                y_cf_logits, c_cf, y_preds, c_preds = output
            elif len(output) == 6:
                y_cf_logits, c_cf, y_prime, c_preds, y_preds, c_true = output

            c_preds_total_noise = torch.cat((c_preds_total_noise, c_preds), dim=0)
            c_total = torch.cat((c_total, c_true), dim=0)
            y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
            y_prime_total = torch.cat((y_prime_total, y_prime), dim=0)
            y_cf_total = torch.cat((y_cf_total, y_cf_logits), dim=0)
            c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)
        
        task_accuracy = roc_auc_score(y_prime_total.cpu(), y_preds_total.detach().cpu())
        task_accuracy_cf = roc_auc_score(y_prime_total.cpu(), y_cf_total.detach().cpu())
        cf_concept_acc = (c_cf_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
        p_c = (c_cf_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
        hamming_distance = torch.norm((c_preds_total_noise>0.5).float().cpu() - (c_cf_total>0.5).float().cpu(), p=0, dim=-1).mean().item() 
        cf_in_pred = cf_in_distribution(c_cf_total.cpu(), c_preds_total.cpu())
        cf_in_train = cf_in_distribution(c_cf_total.cpu(), c_cf_set.cpu())
        mean_distance_train = distance_train(c_cf_total.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())

        result[f'task_accuracy_int_{p}'] = task_accuracy
        result[f'cf_found_int_{p}'] = task_accuracy_cf
        result[f'cf_in_pred_int_{p}'] = cf_in_pred
        result[f'cf_in_train_int_{p}'] = cf_in_train
        result[f'hamming_distance_int_{p}'] = hamming_distance
        result[f'mean_distance_train_{p}'] = mean_distance_train
        result[f'cf_concept_acc_int_{p}'] = cf_concept_acc
        result[f'p_c_{p}'] = p_c


    return result, (cbm, vae)

def train_conceptvcnet(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger):
    train_dl, val_dl = train_dl
    net = train_model(net, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)

    y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    y_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    # c_cf_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    for X_test, c_test, y_test in test_dl:
        c_preds, y_preds, _, _ = net.forward(X_test)
        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        c_total = torch.cat((c_total, c_test), dim=0)
        y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
        y_total = torch.cat((y_total, y_test), dim=0)
        # c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)

    task_accuracy = roc_auc_score(y_total.cpu(), y_preds_total.detach().cpu())
    concept_accuracy = roc_auc_score(c_total, c_preds_total.detach())
    concept_acc = (c_preds_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
    p_c = (c_preds_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
    task_accuracy_perturbed = task_accuracy_int = -1
    explanations, explanation_cf = [], []

    cf_time, cf_found, c_cf_pred, y_cf_target, c_preds, y_cf_total = net.counterfactual_times(test_dl, accelerator)
    
    cf_variability = variability(c_cf_pred, c_preds)
    cf_iou = intersection_over_union(c_cf_pred, c_preds)
    cf_dou = difference_over_union(c_cf_pred, c_preds)
    cf_in_pred = cf_in_distribution(c_cf_pred, c_preds)
    cf_in_train = cf_in_distribution(c_cf_pred, c_total)

    pdist = torch.nn.PairwiseDistance(p=2)
    euclidean_distance = pdist(c_preds, c_cf_pred).mean().item()
    hamming_distance = torch.norm((c_preds>0.5).float() - (c_cf_pred>0.5).float(), p=0, dim=-1).mean().item() 
    mean_distance_train = distance_train(c_cf_pred.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())

    result = {}
    result['task_accuracy'] = task_accuracy
    result['task_cf_accuracy'] = cf_found
    result['task_accuracy_perturbed'] = task_accuracy_perturbed
    result['task_accuracy_int'] = task_accuracy_int
    result['concept_accuracy'] = concept_accuracy
    result['cf_concept_acc_int_0.0'] = concept_acc
    result['p_c'] = p_c
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['mean_distance_train'] = mean_distance_train
    result['explanations'] = explanations
    result['explanation_cf'] = explanation_cf

    save_set_c_and_cf(c_preds, y_preds_total, y_cf_target, c_cf_pred, model, fold, log_dir)

    p_list = [0.1, 0.2, 0.5, 1.0]
    for p in p_list:
        c_preds_total_noise = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        c_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        y_prime_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        y_cf_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
        c_cf_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
        for X_test, c_test, y_test in test_dl:
            y_prime = y_test.clone()
            y_cf_logits, c_cf, y_preds, c_preds = net.predict_counterfactuals(
                X_test, y_cf_target=y_prime, auto_intervention=p
            )

            c_preds_total_noise = torch.cat((c_preds_total_noise, c_preds), dim=0)
            c_total = torch.cat((c_total, c_test), dim=0)
            y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)
            y_prime_total = torch.cat((y_prime_total, y_test), dim=0)
            y_cf_total = torch.cat((y_cf_total, y_cf_logits), dim=0)
            c_cf_total = torch.cat((c_cf_total, c_cf), dim=0)
        
        task_accuracy = roc_auc_score(y_prime_total.cpu(), y_preds_total.detach().cpu())
        task_accuracy_cf = roc_auc_score(y_prime_total.cpu(), y_cf_total.detach().cpu())
        cf_concept_acc = (c_cf_total > 0.5).float().eq(c_total).float().all(dim=-1).float().mean()
        p_c = (c_cf_total > 0.5).float().eq(c_total).float().mean(dim=-1).float().mean()
        hamming_distance = torch.norm((c_preds_total_noise>0.5).float().cpu() - (c_cf_total>0.5).float().cpu(), p=0, dim=-1).mean().item() 
        cf_in_pred = cf_in_distribution(c_cf_total.cpu(), c_preds_total.cpu())
        cf_in_train = cf_in_distribution(c_cf_total.cpu(), c_cf_set.cpu())
        mean_distance_train = distance_train(c_cf_total.cpu(), c_cf_set.cpu(), y_cf_total.detach().cpu(), concept_labels.cpu())

        result[f'task_accuracy_int_{p}'] = task_accuracy
        result[f'cf_found_int_{p}'] = task_accuracy_cf
        result[f'cf_in_pred_int_{p}'] = cf_in_pred
        result[f'cf_in_train_int_{p}'] = cf_in_train
        result[f'mean_distance_train_{p}'] = mean_distance_train
        result[f'hamming_distance_int_{p}'] = hamming_distance
        result[f'cf_concept_acc_int_{p}'] = cf_concept_acc
        result[f'p_c_{p}'] = p_c
 

    return result, net

def train_baycon(net, epochs, learning_rate, seed, train_dl, test_dl, results_dir, accelerator, batch_size, c_cf_set, concept_labels, model, fold, log_dir, wandb_logger):
    train_dl, val_dl = train_dl
    result, net = train_cbm(net, epochs, learning_rate, seed, train_dl, val_dl, results_dir, accelerator, wandb_logger)
    
    c_preds_total = torch.empty(0, train_dl.dataset[0][1].shape[-1])
    y_preds_total = torch.empty(0, train_dl.dataset[0][-1].shape[-1])
    for X_test, c_test, y_test in test_dl:
        c_preds, y_preds, _ = net.forward(X_test)
        if not net.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
            y_preds = torch.sigmoid(y_preds)
        c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
        y_preds_total = torch.cat((y_preds_total, y_preds), dim=0)

    cap = max(50, int(train_dl.dataset[0][-1].shape[-1]*1.5))
    X_test, c_test, y_test = test_dl.dataset.tensors
    X_train, c_train, y_train = train_dl.dataset.tensors
    test_data = TensorDataset(X_test[:cap], c_test[:cap], y_test[:cap])
    test_dl = torch.utils.data.DataLoader(test_data, batch_size, shuffle=True, pin_memory=True)
    
    (cf_time, 
     all_cf_preds, all_best_cf, y_prime_pred,
     task_cf_accuracy, cf_found,
     y_cf_target, hamming_distance, euclidean_distance) = evaluate_baycon(net, train_dl, test_dl, accelerator, train_dl.dataset[0][-1].shape[-1])

    cf_variability = variability(all_best_cf, c_preds_total)
    cf_iou = intersection_over_union(all_best_cf, c_preds_total)
    cf_dou = difference_over_union(all_best_cf, c_preds_total)
    cf_in_pred = cf_in_distribution(all_best_cf, c_preds_total)
    cf_in_train = cf_in_distribution(all_best_cf, c_train)
    mean_distance_train = distance_train(all_best_cf, c_cf_set, y_prime_pred, concept_labels)

    result['task_cf_accuracy'] = task_cf_accuracy
    result['cf_variability'] = cf_variability
    result['cf_iou'] = cf_iou
    result['cf_dou'] = cf_dou
    result['cf_found'] = cf_found
    result['cf_in_pred'] = cf_in_pred
    result['cf_in_train'] = cf_in_train
    result['cf_time'] = cf_time
    result['euclidean_distance'] = euclidean_distance
    result['hamming_distance'] = hamming_distance
    result['mean_distance_train'] = mean_distance_train

    save_set_c_and_cf(c_preds_total, y_preds_total, torch.tensor(y_cf_target), all_best_cf, model, fold, log_dir)

    return result, net
    

