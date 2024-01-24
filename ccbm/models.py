from abc import abstractmethod

import numpy as np
import torch
import math
import time
from torch import optim
from ccbm.utils import randomize_class, sample_bernoulli
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import pytorch_lightning as pl
from torch.distributions import kl_divergence
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss, BCELoss, ModuleList, BCEWithLogitsLoss, KLDivLoss
from torch_explain.nn.semantics import Logic, ProductTNorm
import torch.nn.functional as F 
from tqdm import tqdm
import random

from ccbm.layers import ConceptReasoningLayer
from ccbm.metrics import variability, intersection_over_union, difference_over_union

EPS = 1e-9

class Oracle():
    def find_counterfactuals(self, c_test, y_test, c, y, y_cf_target=None):
        start = time.time()
        binary = False
        if len(y.shape) == 1:
            # y = F.one_hot(y, num_classes=2)
            binary = True
        c_test = c_test.detach().unsqueeze(dim=1)
        c_test_extended = c_test.repeat(1, c.shape[0], 1).to(c.device)
        randperm = torch.randperm(c.shape[0])
        c = c[randperm]   
        y = y[randperm]
        c_extended = c.repeat(c_test.shape[0], 1, 1)
        dist = torch.sum(torch.abs(c_test_extended - c_extended), dim=-1)
        if y_cf_target is not None:
            y_target = y_cf_target
        else:
            if binary:
                y_target = 1 - y_test
            else:
                y_target = randomize_class(y_test, include=False)
        if len(y_target.shape) == 1 or y_target.shape[1] == 1:
            y_target = y_target.squeeze().long()
            y_target = F.one_hot(y_target, num_classes=2)
            y_test = y_test.squeeze().long()
            y_test = F.one_hot(y_test, num_classes=2)
            y = y.squeeze().long()
            y = F.one_hot(y, num_classes=2)
        y_test = y_test.argmax(dim=-1)
        y = y.argmax(dim=-1)
        y_target = y_target.argmax(dim=-1)
        y_target = y_target.unsqueeze(dim=1)
        y_target_extended = y_target.repeat(1, y.shape[0]).to(y.device)
        y_extended = y.repeat(y_target.shape[0], 1)
        # to exclude the true class examples
        # filter = y_test == y
        # dist[filter] = 10000000
        # to exclude example of others classes
        filter = y_target_extended != y_extended
        dist[filter] = 10000000
        cf_indexes = torch.argsort(dist, dim=-1)[:, 0]
        c_cf = c[cf_indexes]
        time_taken = time.time() - start
        return time_taken, c_cf
        

class NeuralNet(pl.LightningModule):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__()
        self.input_features = input_features
        self.n_classes = n_classes
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        if n_classes > 1:
            self.cross_entropy = CrossEntropyLoss(reduction="mean")
        else:
            self.cross_entropy = BCEWithLogitsLoss()
        self.bce = BCELoss(reduction="mean")
        self.bce_log = BCEWithLogitsLoss(reduction="mean")
        self.randomize_class = randomize_class

    @abstractmethod
    def forward(self, X):
        raise NotImplementedError

    @abstractmethod
    def _unpack_input(self, I):
        raise NotImplementedError
    
    
    def _sample_different_label(self, y):
        not_y = []
        if self.n_classes == 1:
            return torch.ones_like(y).long().to(y.device) - y
        else:
            y = y.argmax(dim=1)
            for el in y:
                possible_labels = list(range(self.n_classes))
                possible_labels.remove(el)
                not_y.append(np.random.choice(possible_labels))
            not_y = torch.Tensor(not_y).long().to(y.device)
            return one_hot(not_y, self.n_classes)

    def training_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)

        y_preds = self.forward(X)

        loss = self.cross_entropy(y_preds.squeeze(), y_true.float().argmax(dim=-1))
        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_preds.cpu().detach())
        print(f'Epoch {self.current_epoch}: task accuracy: {task_accuracy:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, _, y_true = self._unpack_input(I)
        y_preds = self.forward(X)
        loss = self.cross_entropy(y_preds.squeeze(), y_true.float().argmax(dim=-1))
        val_acc = roc_auc_score(y_true.cpu(), y_preds.cpu().detach())
        self.log("val_acc", val_acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return optimizer


class StandardE2E(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, learning_rate: float = 0.01):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes)
        )

    def _unpack_input(self, I):
        return I

    def forward(self, X, explain=False):
        return self.model(X)


class StandardCBM(StandardE2E):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, bool_concepts: bool = True, deep: bool = True):
        super().__init__(input_features, n_classes, emb_size, learning_rate)
        self.n_concepts = n_concepts
        self.concept_names = concept_names
        self.task_names = task_names
        self.task_weight = task_weight
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_features, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, emb_size),
            torch.nn.LeakyReLU(),
        )
        self.relation_classifiers = torch.nn.Sequential(torch.nn.Linear(emb_size, n_concepts))
        if deep:
            self.reasoner = torch.nn.Sequential(
            torch.nn.Linear(n_concepts, emb_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(emb_size, n_classes)
            )
        else:
            self.reasoner = torch.nn.Sequential(
                torch.nn.Linear(n_concepts, n_classes),
            )
        self.bool_concepts = bool_concepts
        self.classification_loss = self.bce if bool_concepts else self.bce_log
        self.classification_threshold = 0.5 if bool_concepts else 0

    def forward(self, X, explain=False, concepts=False):
        explanation = None
        if concepts:
            c_preds = X
        else:
            embeddings = self.encoder(X)
            c_preds = self.relation_classifiers(embeddings)
            if self.bool_concepts:
                c_preds = torch.sigmoid(c_preds)
        y_preds = self.reasoner(c_preds)
        return c_preds, y_preds, explanation

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        concept_loss = self.classification_loss(c_preds, c_true.float())
        try:
            task_loss = self.cross_entropy(y_preds, y_true.float().argmax(dim=-1))
        except:
            task_loss = self.cross_entropy(y_preds,  y_true.float())
        loss = concept_loss + self.task_weight*task_loss

        if not self.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
            y_preds = torch.sigmoid(y_preds)

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_preds.squeeze().cpu().detach())
        concept_accuracy = roc_auc_score(c_true.cpu(), c_preds.squeeze().cpu().detach())
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, _ = self.forward(X)

        if not self.bool_concepts:
            c_preds = torch.sigmoid(c_preds)

        concept_loss = self.classification_loss(c_preds, c_true.float())
        try:
            task_loss = self.cross_entropy(y_preds, y_true.float().argmax(dim=-1))
        except:
            task_loss = self.cross_entropy(y_preds,  y_true.float())
        loss = concept_loss + self.task_weight*task_loss
        val_acc = (roc_auc_score(y_true.cpu(), y_preds.cpu()) + roc_auc_score(c_true.cpu(), c_preds.cpu())) / 2
        self.log("val_acc", val_acc)
        return loss


# write function to rescale a tensor A in [0, 1] to [vmin, vmax] where vmin and vmax are the min and max values of another tensor B
def rescale(A, B):
    return A * (0.5*B.max() - 0.5*B.min()) + 0.5*B.min()


class CounterfactualCBM_V3(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, concept_set: torch.Tensor, concept_labels: torch.Tensor,
                 shield: torch.Tensor = None, train_intervention: bool = False,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, bool_concepts: bool = False, deep: bool = True, reconstruction=False, dataset='dsprites',
                 bool_cf=False, resample: int = 0, bernulli: bool = False):
        if bool_cf:
            bool_concepts = True
        bool_concepts = True
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight, bool_concepts, deep)
        
        self.bernulli = bernulli
        self.dataset = dataset
        self.concept_list = concept_set
        self.concept_set = set([tuple(el) for el in self.concept_list.cpu().detach().numpy()])
        self.concept_labels = concept_labels
        self.resample = resample
        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU())
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + 2 * n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + 2 * n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_concepts))
        self.epoch = 0
        self.deep = deep
        if deep:
            self.reasoner = torch.nn.Sequential(torch.nn.Linear(n_concepts, n_concepts), torch.nn.LeakyReLU(), torch.nn.Linear(n_concepts, n_classes))
        else:
            self.reasoner = torch.nn.Sequential(torch.nn.Linear(n_concepts, n_classes))

        self.shield = shield
        self.train_intervention = train_intervention
        self.classification_loss = self.bce if bool_concepts else self.bce_log
        self.classification_threshold = 0.5 if bool_concepts else 0
        self.reconstruction = reconstruction
        self.bool_cf = bool_cf
        
    def _predict_concepts(self, embeddings, predictor):
        c_preds = predictor(embeddings)
        if self.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
        return c_preds
    
    def counterfactual_times(
        self,
        test_dl,
        accelerator,
        rerun=False,
        n_times = 1,
        return_cf = False,
        auto_intervention=0
    ):
        total = 0
        found = 0
        total_time = 0
        y_prime_pred = torch.empty((0, self.n_classes))
        y_prime_list = torch.empty((0, self.n_classes))
        cf_pred_total = torch.empty((0, self.n_concepts))
        c_preds_total = torch.empty((0, self.n_concepts))
        correct = 0
        for x, c, y in test_dl:
            x = x
            y = y 
            start = time.time() 
            y_prime = self.randomize_class(y, include=False)
            if y.shape[-1] == 1:
                y_prime = 1 - y
            y_cf_logits, c_cf_pred, c_preds = self.predict_counterfactuals(
                x, y_prime=y_prime, resample=n_times, return_cf=True, full=True
            )
            time_taken = time.time() - start
            total_time += time_taken

            y_prime_pred = torch.cat((y_prime_pred, y_cf_logits), dim=0)
            y_prime_list = torch.cat((y_prime_list, y_prime), dim=0)
            c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
            if return_cf:
                cf_pred_total = torch.cat((cf_pred_total, c_cf_pred), dim=0)
            
            pred = torch.argmax(y_cf_logits, dim=-1)
            y_p = torch.argmax(y_prime, dim=-1)

            filter = pred != y_p
            total += y.shape[0]
            correct += (pred == y_p).sum()
        roc_cf = roc_auc_score(y_prime_list.cpu().detach().numpy(), y_prime_pred.cpu().detach().numpy())
        if return_cf:
            return total_time, roc_cf, cf_pred_total, c_preds_total, y_prime_pred
        return total_time, roc_cf
    
    def _predict_task(self, c, embeddings=None):
        return self.reasoner(c)
    
    def filter_counterfactuals(self, c_prime_pred, c_pred, y_cf_pred, y_prime, dist_samples, distribution):
        # check if counterfactuals have the right predictions and then 
        # takes the ones with the minimum hamming distance between c_prime_pred and c_pred
        # c_prime_pred is of shape (resampling, batch_size, n_concepts)
        # c_pred is of shape (batch_size, n_concepts)
        # y_cf_pred is of shape (resampling, batch_size, n_classes)
        # y_prime is of shape (batch_size, n_classes)
        # returns c_prime_pred and y_cf_pred of shape (batch_size, n_concepts) and (batch_size, n_classes)
        y_prime = y_prime.unsqueeze(0).repeat(c_prime_pred.shape[0], 1, 1)
        c_pred = c_pred.unsqueeze(0).repeat(c_prime_pred.shape[0], 1, 1)
        filter_y = y_prime.argmax(dim=-1) != y_cf_pred.argmax(dim=-1)
        hamming_distance = torch.norm((c_prime_pred > 0.5).float() - (c_pred > 0.5).float(), p=0, dim=-1)
        sampling_probability = distribution.log_prob(dist_samples).exp().mean(dim=-1)
        hamming_distance_tmp = hamming_distance.clone()
        sampling_probability_tmp = sampling_probability.clone()
        hamming_distance_tmp[filter_y] = 1000
        hamming_distance_tmp[hamming_distance == 0] = 1000
        sampling_probability_tmp[filter_y] = -1
        sampling_probability_tmp[hamming_distance == 0] = -1
        # min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        min_sampling_probability, min_sampling_probability_idx = torch.max(sampling_probability_tmp, dim=0)
        c_prime_pred = c_prime_pred[min_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        y_cf_pred = y_cf_pred[min_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        # # min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        # # min_hamming_distance, min_hamming_distance_idx = torch.max(hamming_distance_tmp, dim=0)
        # max_sampling_probability, max_sampling_probability_idx = torch.max(sampling_probability_tmp, dim=0)
        # c_prime_pred = c_prime_pred[max_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        # y_cf_pred = y_cf_pred[max_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        return c_prime_pred, y_cf_pred
    
    def predict_counterfactuals(self, X, y_prime=None, resample=1, return_cf=False, auto_intervention=0, full=False):
        h = self.encoder(X)
        z2_mu = self.concept_mean_predictor(h)
        z2_log_var = self.concept_var_predictor(h)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()
        # z2 = z2_mu

        c_pred = torch.sigmoid(self.concept_predictor(z2)*4)

        if auto_intervention > 0:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_pred.shape[1])[:int(math.ceil(auto_intervention*c_pred.shape[1]))]
            # flip
            c_pred[:, index] = 1 - c_pred[:, index]

        y_pred = self._predict_task(c_pred, h)

        if y_prime is None:
            y_prime = self.randomize_class((y_pred).float())

        # q(z3|z2, c, y, y')
        z2_c_y_y_prime = torch.cat((z2, c_pred, y_pred, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        z3 = qz3_z2_c_y_y_prime.rsample(torch.Size([resample]))

        if resample == 1:
            z3 = z3_mu

        c_prime_pred = torch.sigmoid(self.concept_predictor(z3)*4)


        y_cf_pred = self._predict_task(c_prime_pred, h)

        if resample > 1:
            c_prime_pred, y_cf_pred = self.filter_counterfactuals(c_prime_pred, c_pred, y_cf_pred, y_prime, z3, qz3_z2_c_y_y_prime)  
            y_cf_pred = self._predict_task(c_prime_pred, h)
        else:
            y_cf_pred = y_cf_pred.squeeze(0)

        if auto_intervention > 0:
            return y_cf_pred, c_prime_pred, y_pred, c_pred

        if return_cf:
            if full: 
                return y_cf_pred, c_prime_pred, c_pred
            return y_cf_pred, c_prime_pred

        return y_cf_pred

    def extract_counterfactual_explanation(self, y_preds, class_id, c_preds, c_cf, embeddings):
        if not self.bool_concepts:
            y_preds = torch.sigmoid(y_preds)
            c_preds = torch.sigmoid(c_preds)
        class_mask = y_preds[:, class_id] > self.classification_threshold
        c_preds_class = (c_preds[class_mask] > self.classification_threshold).int()
        c_preds_class_cf = (c_cf[class_mask] > self.classification_threshold).int()
        return c_preds_class, c_preds_class_cf
    
    def sample_again(self, cf_pred, pcprime_z3, y_pred, y_target, counter):
        if counter == 0:
            return cf_pred
        filter = []
        cf_pred_clone = cf_pred.clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set and y_pred[i].argmax() == y_target[i].argmax():
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        new_samples = sample_bernoulli(pcprime_z3)
        cf_pred[filter] = new_samples[filter]
        return cf_pred, filter
    
    def sample_again_z(self, cf_pred, qz3, y_pred, y_target, counter):
        if counter == 0:
            return cf_pred
        filter = []
        cf_pred_clone = (cf_pred > 0.5).float().clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set and y_pred[i].argmax() == y_target[i].argmax():
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        new_samples = qz3.rsample()
        new_cf = torch.sigmoid(self.concept_predictor(new_samples))
        cf_pred[filter] = new_cf[filter]
        return cf_pred, filter, new_samples

    
    def check_overscribed_not_ind(self, c_pred, c_pred_init, z3, z3_init):
        filter = []
        cf_pred_clone = (c_pred > 0.5).float().clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set:
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        print(filter.sum())
        c_pred[filter] = c_pred_init[filter]
        z3[filter] = z3_init[filter]
        return c_pred, z3
    
    def select_cf(self, c_cf, c, y_cf, y_target):
        filter = y_cf.argmax(dim=-1) == y_target.argmax(dim=-1)
        filter[0, :] = True
        print(filter.shape)
        right_cf = c_cf[filter]
        print(right_cf.shape)
        weights = torch.abs(filter.float()-1).sum(dim=-1)
        return right_cf[0, :, :]
    
    def dont_change(self, c_prime_pred, c_pred):
        changes = torch.abs((c_prime_pred > 0.5).float() - (c_pred > 0.5).float())
        c_prime_pred_tmp = c_prime_pred*changes + c_pred*(1-changes)
        changes = (torch.abs(c_prime_pred - c_pred) < 0.01).float()
        c_prime_pred = c_prime_pred*changes + c_prime_pred_tmp*(1-changes)
        return c_prime_pred

    def filter_y_c(self, y_pred, c_pred, z2):
        
        y_pred_argmax = y_pred.argmax(dim=-1).T.detach().numpy()

        def extract_most_freq(x):
            unique, count = np.unique(x, return_counts=True)
            index = np.argmax(count)
            result = unique[index]
            return result
        def extract_highest_sum(x):
            sum = np.sum(x, axis=0)
            result = np.argmax(sum, axis=-1)
            return result

        # index = extract_highest_sum(y_pred.detach().numpy())
        index = np.apply_along_axis(extract_most_freq, arr=y_pred_argmax, axis=-1)
        index = torch.Tensor(index)

        y_pred_argmax_index = y_pred.argmax(dim=-1).eq(index).float().argmax(dim=0).squeeze(0)

        y_pred_selected = y_pred[y_pred_argmax_index, list(range(y_pred.shape[1])), :]
        c_pred_selected = c_pred[y_pred_argmax_index, list(range(y_pred.shape[1])), :]
        z2_selected = z2[y_pred_argmax_index, list(range(y_pred.shape[1])), :]

        return y_pred_selected, c_pred_selected, z2_selected

    def forward(self, X, c=None, c_cf=None, y_prime=None, explain=False, explanation_mode='local', auto_intervention=0, test=False, y_true=None, include=True, resample=1, inference=False):
        if auto_intervention > 0 and y_true is not None:
            X = torch.cat((X, X[:int(y_true.shape[0])]))
        h = self.encoder(X)
        z2_mu = self.concept_mean_predictor(h)
        z2_log_var = self.concept_var_predictor(h)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()

        if inference:
            z2 = z2_mu
        # p(c|z2)
        pc_z2 = torch.distributions.Bernoulli(logits=torch.zeros(self.concept_predictor(z2).shape))
        c_pred = torch.sigmoid(self.concept_predictor(z2)*4)

        # p(z2)
        p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

        # p(y|c)
        if self.bernulli:
            if test:
                py_c = torch.distributions.Bernoulli(logits = self.reasoner(c_pred))
            else:
                py_c = torch.distributions.Bernoulli(logits = self.reasoner(c.float()))
            y_pred = self._predict_task(c_pred, h)
        else:
            py_c = None
            if test:
                y_pred = self._predict_task(c_pred, h)
            else:
                y_pred = self._predict_task(c.float(), h)

        if y_prime is None:
            if y_pred.shape[-1] == 1:
                y_prime = 1 - (y_pred > 0).float()
            else:
                y_prime = self.randomize_class((y_pred).float(), include=include)
        if auto_intervention > 0 and y_true is not None:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_pred.shape[1])[:int(math.ceil(auto_intervention*c_pred.shape[1]))]
            # flip
            c_pred_inv = c_pred.clone().detach()
            c_pred_inv[-int(y_true.shape[0]):, index] = 1 - c_pred[-int(y_true.shape[0]):, index]
            c_pred_inv = c_pred_inv[-int(y_true.shape[0]):]
            y_prime_inv = y_true
            c_pred = c_pred[:-int(y_true.shape[0])]
            c_pred = torch.cat((c_pred, c_pred_inv), dim=0)
            y_prime = y_prime[:-int(y_true.shape[0])]
            y_prime = torch.cat((y_prime, y_prime_inv), dim=0)
            

        # q(z3|z2, c, y, y')
        z2_c_y_y_prime = torch.cat((z2, c_pred, y_pred, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        z3 = qz3_z2_c_y_y_prime.rsample(sample_shape=torch.Size())

        if inference:
            z3 = z3_mu

        # p(c'|z3)
        pcprime_z3 = torch.distributions.Bernoulli(logits=self.concept_predictor(z3))
        c_prime_pred = torch.sigmoid(self.concept_predictor(z3)*4)

        oracle = Oracle()
        if c is not None:
            c_cf = c

        # p(y'|c')
        if self.bernulli:
            if test:
                py_c_cf = torch.distributions.Bernoulli(logits = self.reasoner(c_prime_pred))
            else:
                py_c_cf = torch.distributions.Bernoulli(logits = self.reasoner(c_cf.float())) #TODO c or c_pred?
            # y_cf_pred = py_c_cf.sample()
            # y_cf_pred = sample_bernoulli(py_c_cf)
            y_cf_pred = self._predict_task(c_prime_pred, h)
        else:
            if test:
                y_cf_pred = self._predict_task(c_prime_pred, h)
            else:
                y_cf_pred = self._predict_task(c_cf.float(), h)
            py_c_cf = None


        weights = torch.ones(c_prime_pred.shape[0])

        # p(z3|z2, c, y)
        z2_c_y = torch.cat((z2, c_pred, y_pred), dim=1)
        z3_mu = self.concept_mean_z3_predictor(z2_c_y)
        z3_log_var = self.concept_var_z3_predictor(z2_c_y)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        pz3_z2_c_y = torch.distributions.Normal(z3_mu, z3_sigma)

        # extract explanations
        explanation, explanation_cf = {}, {}
        if explain:
            for class_id in range(self.n_classes):
                c_preds_explanation, c_preds_explanation_cf = self.extract_counterfactual_explanation(y_pred, class_id, c_pred, c_prime_pred, h)
                explanation_cf[class_id] = {'explanation': c_preds_explanation, 'counterfactual': c_preds_explanation_cf}

        return c_pred, y_pred, explanation, c_prime_pred, y_cf_pred, y_prime, explanation_cf, p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime, pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, None
    
    def training_step(self, I, batch_idx):
        if batch_idx == 0:
            self.epoch += 1
        X, c_true, y_true = self._unpack_input(I)
        self.actual_resample = 0
        if self.reconstruction:
            intervention_list = [0.1, 0.2, 0.5]
            # random int between 0 and 2
            rand_int = random.randint(0,2)
            p = intervention_list[rand_int]
        else:
            p = 0
        (c_pred, y_pred, explanation,
         c_prime, y_cf, y_prime, explanation_cf, 
         p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
         pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, _) = self.forward(X, c_true, y_true=y_true[:int(X.shape[0]*0.2)], test=True, auto_intervention=p)
        int_concept_loss, int_task_loss = 0, 0
        int_concept_accuracy, int_task_accuracy = 0, 0

        if self.reconstruction:
            z2 = z2[:-int(X.shape[0]*0.2)]
            z3 = z3[:-int(X.shape[0]*0.2)]
            c_true = torch.cat((c_true, c_true[:int(X.shape[0]*0.2)]), dim=0)
            c_prime_int = c_prime[-int(X.shape[0]*0.2):]
            c_prime = c_prime[:-int(X.shape[0]*0.2)]
            y_pred = y_pred[:-int(X.shape[0]*0.2)]
            c_pred = c_pred[:-int(X.shape[0]*0.2)]

        # compute loss

        # KL( p(z2) || q(z2|x))
        kl_loss_z2 = torch.distributions.kl_divergence(p_z2, qz2_x).mean()

        # KL( p(z3|z2,c,y) || q(z3|z2,c,y,y'))
        kl_loss_z3 = torch.distributions.kl_divergence(pz3_z2_c_y, qz3_z2_c_y_y_prime).mean()

        kl_loss_dist = torch.distributions.kl_divergence(p_z2, pz3_z2_c_y).mean()

        kl_q_dist = torch.distributions.kl_divergence(qz2_x, qz3_z2_c_y_y_prime).mean()

        dist_loss = torch.norm(z2 - z3, p=2, dim=1).mean()
        hamming_loss = torch.abs(torch.norm((c_pred > 0.5).float() - (c_prime > 0.5).float(), p=0, dim=-1)).mean()

        if self.reconstruction:
            reconstruction_loss = self.bce_log(c_prime_int, c_true[:int(X.shape[0]*0.2)].float())
            print(reconstruction_loss)
        else:
            reconstruction_loss = 0

        # E_{c' ~ q(c'|c,y')}[-log p(c'|c))
        # admissibility_cf = torch.mean(- pcprime_z3.log_prob(c_cf.float()))
        admissibility_cf = 0

        if self.bernulli:
            # -log(p(y|c))
            task_loss = torch.mean(- py_c.log_prob(y_true.float()))
            # E_{c' ~ q(c'|c,y')}[ -log(p(y'|c'))]
            task_loss_cf = torch.mean(- py_c_cf.log_prob(y_prime.float()))
        else:
            try:
                task_loss = self.cross_entropy(y_pred, y_true.float().argmax(dim=-1))
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float().argmax(dim=-1))
            except:
                task_loss = self.cross_entropy(y_pred, y_true.float())
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float())

        # E_{c' ~ q(z2 | x)}[-log(p(c|z2))]
        concept_loss = self.bce(c_pred, c_true.float())
        # loss = torch.nn.MSELoss()
        # concept_loss = loss(c_pred, c_true.float())
        # concept_loss = torch.mean(- pc_z2.log_prob(c_true.float()))
        # concept_loss = torch.mean(- p_z2.log_prob(c_true.float()))

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_pred.squeeze().cpu().detach())
        c_true = c_true[:int(c_pred.shape[0])]
        concept_accuracy = roc_auc_score(c_true.cpu(), c_pred.squeeze().cpu().detach())

        try:
            task_cf_accuracy = roc_auc_score(y_prime.squeeze().cpu(), y_cf.squeeze().cpu().detach(), average='micro')
        except:
            task_cf_accuracy = 0

        
        # dsprites loss
        if self.dataset == 'dsprites':
            if self.deep:
                loss = 0.7*task_loss + 10*concept_loss + 0.3*task_loss_cf + 1.2*(kl_loss_z2  + kl_loss_z3) + 1.0*kl_loss_dist + 0.6*dist_loss + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf
            else:
                if y_pred.shape[-1] == 1:
                    loss = 0.7*task_loss + 10*concept_loss + 0.5*task_loss_cf + 1.2*(kl_loss_z2  + kl_loss_z3) + 1.0*kl_loss_dist + 0.35*dist_loss + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
                else:
                    loss = 0.7*task_loss + 10*concept_loss + 0.5*task_loss_cf + 1.2*(kl_loss_z2  + kl_loss_z3) + 1.0*kl_loss_dist + 0.5*dist_loss + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
        # mnist loss 
        else:
            loss = 1*task_loss + 10*concept_loss + 0.2*task_loss_cf + 2*(kl_loss_z2  + kl_loss_z3) + 1.7*kl_loss_dist + 0.55*dist_loss # + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf  
          
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        self.actual_resample = self.resample

        (c_pred, y_pred, explanation,
         c_prime, y_cf, y_prime, explanation_cf, 
         p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
         pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, _) = self.forward(X, test=True)

        if self.reconstruction:
            z2 = z2[:-int(X.shape[0]*0.2)]
            z3 = z3[:-int(X.shape[0]*0.2)]
            c_true = torch.cat((c_true, c_true[:int(X.shape[0]*0.2)]), dim=0)
            c_prime_int = c_prime[-int(X.shape[0]*0.2):]
            c_prime = c_prime[:-int(X.shape[0]*0.2)]
            y_pred = y_pred[:-int(X.shape[0]*0.2)]
            c_pred = c_pred[:-int(X.shape[0]*0.2)]

        # compute loss

        # KL( p(z2) || q(z2|x))
        kl_loss_z2 = torch.distributions.kl_divergence(p_z2, qz2_x).mean()

        # KL( p(z3|z2,c,y) || q(z3|z2,c,y,y'))
        kl_loss_z3 = torch.distributions.kl_divergence(pz3_z2_c_y, qz3_z2_c_y_y_prime).mean()

        kl_loss_dist = torch.distributions.kl_divergence(p_z2, pz3_z2_c_y).mean()

        kl_q_dist = torch.distributions.kl_divergence(qz2_x, qz3_z2_c_y_y_prime).mean()

        dist_loss = torch.norm(z2 - z3, p=2, dim=1).mean()
        hamming_loss = torch.abs(torch.norm((c_pred > 0.5).float() - (c_prime > 0.5).float(), p=0, dim=-1)).mean()

        if self.reconstruction:
            reconstruction_loss = self.bce_log(c_prime_int, c_true[:int(X.shape[0]*0.2)].float())
        else:
            reconstruction_loss = 0

        # E_{c' ~ q(c'|c,y')}[-log p(c'|c))
        # admissibility_cf = torch.mean(- pcprime_z3.log_prob(c_cf.float()))
        admissibility_cf = 0

        if self.bernulli:
            # -log(p(y|c))
            task_loss = torch.mean(- py_c.log_prob(y_true.float()))
            # E_{c' ~ q(c'|c,y')}[ -log(p(y'|c'))]
            task_loss_cf = torch.mean(- py_c_cf.log_prob(y_prime.float()))
        else:
            try:
                task_loss = self.cross_entropy(y_pred, y_true.float().argmax(dim=-1))
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float().argmax(dim=-1))
            except:
                task_loss = self.cross_entropy(y_pred, y_true.float())
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float())

        # E_{c' ~ q(z2 | x)}[-log(p(c|z2))]
        # concept_loss = self.bce(c_pred, c_true.float())
        concept_loss = torch.mean(- pc_z2.log_prob(c_true.float()))
        # concept_loss = torch.mean(- p_z2.log_prob(c_true.float()))


        
        # dsprites loss
        #loss = 0.7*task_loss + 10*concept_loss + 0.3*task_loss_cf + 1.2*(kl_loss_z2  + kl_loss_z3) + 1.0*kl_loss_dist + 0.5*dist_loss + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
        # # mnist loss 
        # if self.reconstruction:
        #     loss = 0.7*task_loss + 10*concept_loss + 0.2*task_loss_cf + 2*(kl_loss_z2  + kl_loss_z3) + 2.0*kl_loss_dist + 0.55*dist_loss + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
        # else:
        #     # loss = 0.7*task_loss + 10*concept_loss + 0.2*task_loss_cf + 0.9*(kl_loss_z2  + kl_loss_z3) + 1*kl_loss_dist + 0.63*dist_loss + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
        loss = 1*task_loss + 10*concept_loss + 0.2*task_loss_cf + 2*(kl_loss_z2  + kl_loss_z3) + 1.5*kl_loss_dist + 0.4*dist_loss # + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
        #     # loss = 1*task_loss + 10*concept_loss + 0.15*task_loss_cf + 1*kl_loss_z2  + 3*kl_loss_z3 + 2*kl_loss_dist + 0.0*dist_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf 
              

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_pred.squeeze().cpu().detach())
        concept_accuracy = roc_auc_score(c_true.cpu(), c_pred.squeeze().cpu().detach())
        try:
            # task_cf_accuracy = roc_auc_score(y_prime.squeeze().cpu(), y_cf.squeeze().cpu().detach(), average='micro')
            task_cf_accuracy = (y_prime.squeeze().cpu() == (y_cf > 0).float().squeeze().cpu().detach()).float().mean()
        except:
            task_cf_accuracy = 0
        
        val_acc = np.float32((task_accuracy + concept_accuracy + task_cf_accuracy) / 3)
        self.log("val_acc", val_acc)
        return loss
    
class CounterfactualCBM_V3_1(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, concept_set: torch.Tensor, concept_labels: torch.Tensor,
                 shield: torch.Tensor = None, train_intervention: bool = False,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, bool_concepts: bool = False, deep: bool = True, reconstruction=False,
                 bool_cf=False, resample: int = 0, bernulli: bool = False):
        if bool_cf:
            bool_concepts = True
        bool_concepts = True
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight, bool_concepts, deep)
        
        self.bernulli = bernulli
        self.concept_list = concept_set
        self.concept_set = set([tuple(el) for el in self.concept_list.cpu().detach().numpy()])
        self.concept_labels = concept_labels
        self.resample = resample
        self.encoder = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU())
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_mean_z3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_z3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_mean_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + 2 * n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_var_qz3_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size + n_concepts + 2 * n_classes, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.concept_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_concepts))
        self.epoch = 0
        if deep:
            self.reasoner = torch.nn.Sequential(torch.nn.Linear(n_concepts, n_concepts), torch.nn.LeakyReLU(), torch.nn.Linear(n_concepts, n_classes))
        else:
            self.reasoner = torch.nn.Sequential(torch.nn.Linear(n_concepts, n_classes))

        self.shield = shield
        self.train_intervention = train_intervention
        self.classification_loss = self.bce if bool_concepts else self.bce_log
        self.classification_threshold = 0.5 if bool_concepts else 0
        self.reconstruction = reconstruction
        self.bool_cf = bool_cf
        
    def _predict_concepts(self, embeddings, predictor):
        c_preds = predictor(embeddings)
        if self.bool_concepts:
            c_preds = torch.sigmoid(c_preds)
        return c_preds
    
    def counterfactual_times(
        self,
        test_dl,
        accelerator,
        rerun=False,
        n_times = 1,
        return_cf = False,
        auto_intervention=0
    ):
        total = 0
        found = 0
        total_time = 0
        y_prime_pred = torch.empty((0, self.n_classes))
        y_prime_list = torch.empty((0, self.n_classes))
        cf_pred_total = torch.empty((0, self.n_concepts))
        c_preds_total = torch.empty((0, self.n_concepts))
        correct = 0
        for x, c, y in test_dl:
            x = x
            y = y 
            start = time.time() 
            y_prime = self.randomize_class(y, include=False)
            y_cf_logits, c_cf_pred, c_preds = self.predict_counterfactuals(
                x, y_prime=y_prime, resample=n_times, return_cf=True, full=True
            )
            time_taken = time.time() - start
            total_time += time_taken

            y_prime_pred = torch.cat((y_prime_pred, y_cf_logits), dim=0)
            y_prime_list = torch.cat((y_prime_list, y_prime), dim=0)
            c_preds_total = torch.cat((c_preds_total, c_preds), dim=0)
            if return_cf:
                cf_pred_total = torch.cat((cf_pred_total, c_cf_pred), dim=0)
            
            pred = torch.argmax(y_cf_logits, dim=-1)
            y_p = torch.argmax(y_prime, dim=-1)

            filter = pred != y_p
            total += y.shape[0]
            correct += (pred == y_p).sum()
        roc_cf = roc_auc_score(y_prime_list.cpu().detach().numpy(), y_prime_pred.cpu().detach().numpy())
        if return_cf:
            return total_time, roc_cf, cf_pred_total, c_preds_total, y_prime_pred
        return total_time, roc_cf
    
    def _predict_task(self, c, embeddings=None):
        return self.reasoner(c)
    
    def filter_counterfactuals(self, c_prime_pred, c_pred, y_cf_pred, y_prime, dist_samples, distribution):
        # check if counterfactuals have the right predictions and then 
        # takes the ones with the minimum hamming distance between c_prime_pred and c_pred
        # c_prime_pred is of shape (resampling, batch_size, n_concepts)
        # c_pred is of shape (batch_size, n_concepts)
        # y_cf_pred is of shape (resampling, batch_size, n_classes)
        # y_prime is of shape (batch_size, n_classes)
        # returns c_prime_pred and y_cf_pred of shape (batch_size, n_concepts) and (batch_size, n_classes)
        y_prime = y_prime.unsqueeze(0).repeat(c_prime_pred.shape[0], 1, 1)
        c_pred = c_pred.unsqueeze(0).repeat(c_prime_pred.shape[0], 1, 1)
        filter_y = y_prime.argmax(dim=-1) != y_cf_pred.argmax(dim=-1)
        hamming_distance = torch.norm((c_prime_pred > 0.5).float() - (c_pred > 0.5).float(), p=0, dim=-1)
        sampling_probability = distribution.log_prob(dist_samples).exp().mean(dim=-1)
        hamming_distance_tmp = hamming_distance.clone()
        sampling_probability_tmp = sampling_probability.clone()
        hamming_distance_tmp[filter_y] = 1000
        hamming_distance_tmp[hamming_distance == 0] = 1000
        sampling_probability_tmp[filter_y] = -1
        sampling_probability_tmp[hamming_distance == 0] = -1
        # min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        min_sampling_probability, min_sampling_probability_idx = torch.max(sampling_probability_tmp, dim=0)
        c_prime_pred = c_prime_pred[min_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        y_cf_pred = y_cf_pred[min_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        # # min_hamming_distance, min_hamming_distance_idx = torch.min(hamming_distance_tmp, dim=0)
        # # min_hamming_distance, min_hamming_distance_idx = torch.max(hamming_distance_tmp, dim=0)
        # max_sampling_probability, max_sampling_probability_idx = torch.max(sampling_probability_tmp, dim=0)
        # c_prime_pred = c_prime_pred[max_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        # y_cf_pred = y_cf_pred[max_sampling_probability_idx, torch.tensor(list(range(c_pred.shape[1]))), :].squeeze(0)
        return c_prime_pred, y_cf_pred
    
    def predict_counterfactuals(self, X, y_prime=None, resample=1, return_cf=False, auto_intervention=0, full=False):
        h = self.concept_mean_predictor(X)
        z2_mu = self.concept_mean_predictor(X)
        z2_log_var = self.concept_var_predictor(X)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()
        z2 = z2_mu

        c_pred_d = torch.sigmoid(self.concept_predictor(z2)*4)
        c_pred = torch.sigmoid(self.concept_predictor(h)*4)

        if auto_intervention > 0:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_pred.shape[1])[:int(math.ceil(auto_intervention*c_pred.shape[1]))]
            # flip
            c_pred[:, index] = 1 - c_pred[:, index]
            c_pred_d[:, index] = 1 - c_pred[:, index]

        y_pred = self._predict_task(c_pred, h)

        if y_prime is None:
            y_prime = self.randomize_class((y_pred).float())

        # q(z3|z2, c, y, y')
        z2_c_y_y_prime = torch.cat((z2, c_pred_d, y_pred, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        
        if resample == 1:
            z3 = z3_mu
        else:
            z3 = qz3_z2_c_y_y_prime.rsample(torch.Size([resample]))

        c_prime_pred = torch.sigmoid(self.concept_predictor(z3)*4)

        y_cf_pred = self._predict_task(c_prime_pred, h)

        if resample > 1:
            c_prime_pred, y_cf_pred = self.filter_counterfactuals(c_prime_pred, c_pred_d, y_cf_pred, y_prime, z3, qz3_z2_c_y_y_prime)  
            y_cf_pred = self._predict_task(c_prime_pred, h)
        # else:
        #     y_cf_pred = y_cf_pred.squeeze(0)

        if auto_intervention > 0:
            return y_cf_pred, c_prime_pred, y_pred, c_pred

        if return_cf:
            if full: 
                return y_cf_pred, c_prime_pred, c_pred
            return y_cf_pred, c_prime_pred

        return y_cf_pred

    def extract_counterfactual_explanation(self, y_preds, class_id, c_preds, c_cf, embeddings):
        if not self.bool_concepts:
            y_preds = torch.sigmoid(y_preds)
            c_preds = torch.sigmoid(c_preds)
        class_mask = y_preds[:, class_id] > self.classification_threshold
        c_preds_class = (c_preds[class_mask] > self.classification_threshold).int()
        c_preds_class_cf = (c_cf[class_mask] > self.classification_threshold).int()
        return c_preds_class, c_preds_class_cf
    
    def sample_again(self, cf_pred, pcprime_z3, y_pred, y_target, counter):
        if counter == 0:
            return cf_pred
        filter = []
        cf_pred_clone = cf_pred.clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set and y_pred[i].argmax() == y_target[i].argmax():
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        new_samples = sample_bernoulli(pcprime_z3)
        cf_pred[filter] = new_samples[filter]
        return cf_pred, filter
    
    def sample_again_z(self, cf_pred, qz3, y_pred, y_target, counter):
        if counter == 0:
            return cf_pred
        filter = []
        cf_pred_clone = (cf_pred > 0.5).float().clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set and y_pred[i].argmax() == y_target[i].argmax():
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        new_samples = qz3.rsample()
        new_cf = torch.sigmoid(self.concept_predictor(new_samples))
        cf_pred[filter] = new_cf[filter]
        return cf_pred, filter, new_samples

    
    def check_overscribed_not_ind(self, c_pred, c_pred_init, z3, z3_init):
        filter = []
        cf_pred_clone = (c_pred > 0.5).float().clone().cpu().detach().numpy()
        for i in range(cf_pred_clone.shape[0]):
            if tuple(cf_pred_clone[i].astype(int)) in self.concept_set:
                filter += [False]
            else:
                filter += [True]
        filter = torch.Tensor(filter).bool()
        print(filter.sum())
        c_pred[filter] = c_pred_init[filter]
        z3[filter] = z3_init[filter]
        return c_pred, z3
    
    def select_cf(self, c_cf, c, y_cf, y_target):
        filter = y_cf.argmax(dim=-1) == y_target.argmax(dim=-1)
        filter[0, :] = True
        print(filter.shape)
        right_cf = c_cf[filter]
        print(right_cf.shape)
        weights = torch.abs(filter.float()-1).sum(dim=-1)
        return right_cf[0, :, :]
    
    def dont_change(self, c_prime_pred, c_pred):
        changes = torch.abs((c_prime_pred > 0.5).float() - (c_pred > 0.5).float())
        c_prime_pred_tmp = c_prime_pred*changes + c_pred*(1-changes)
        changes = (torch.abs(c_prime_pred - c_pred) < 0.01).float()
        c_prime_pred = c_prime_pred*changes + c_prime_pred_tmp*(1-changes)
        return c_prime_pred

    def filter_y_c(self, y_pred, c_pred, z2):
        
        y_pred_argmax = y_pred.argmax(dim=-1).T.detach().numpy()

        def extract_most_freq(x):
            unique, count = np.unique(x, return_counts=True)
            index = np.argmax(count)
            result = unique[index]
            return result
        def extract_highest_sum(x):
            sum = np.sum(x, axis=0)
            result = np.argmax(sum, axis=-1)
            return result

        # index = extract_highest_sum(y_pred.detach().numpy())
        index = np.apply_along_axis(extract_most_freq, arr=y_pred_argmax, axis=-1)
        index = torch.Tensor(index)

        y_pred_argmax_index = y_pred.argmax(dim=-1).eq(index).float().argmax(dim=0).squeeze(0)

        y_pred_selected = y_pred[y_pred_argmax_index, list(range(y_pred.shape[1])), :]
        c_pred_selected = c_pred[y_pred_argmax_index, list(range(y_pred.shape[1])), :]
        z2_selected = z2[y_pred_argmax_index, list(range(y_pred.shape[1])), :]

        return y_pred_selected, c_pred_selected, z2_selected

    def forward(self, X, c=None, c_cf=None, y_prime=None, explain=False, explanation_mode='local', auto_intervention=0, test=False, y_true=None, include=True, resample=1, inference=False):
        if auto_intervention > 0 and y_true is not None:
            X = torch.cat((X, X[:int(y_true.shape[0])]))
        h = self.concept_mean_predictor(X)
        z2_mu = self.concept_mean_predictor(X)
        z2_log_var = self.concept_var_predictor(X)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)

        if inference:
            z2 = z2_mu
        else:
            z2 = qz2_x.rsample()
        # p(c|z2)
        # pc_z2 = torch.distributions.Bernoulli(logits=self.concept_predictor(z2)*4)
        pc_z2 = torch.distributions.Bernoulli(logits=torch.zeros(self.concept_predictor(z2).shape))
        # pc_z2 = torch.distributions.ContinuousBernoulli(logits=self.concept_predictor(z2))
        # c_pred = pc_z2.rsample()
        # c_pred = sample_bernoulli(pc_z2)
        c_pred_d = torch.sigmoid(self.concept_predictor(z2)*4)
        c_pred = torch.sigmoid(self.concept_predictor(h)*4)

        # if self.actual_resample > 0:
        #     c_pred_init = c_pred
        #     resample = self.actual_resample
        #     n_errors = -1
        #     while resample != 0 and n_errors != 0:
        #         c_pred, errors = self.sample_again(c_pred, pc_z2, resample)
        #         resample -= 1
        #         n_errors = errors.sum()
        #     print(errors.sum())
        #     c_pred = self.check_overscribed_not_ind(c_pred, c_pred_init)

        # p(z2)
        p_z2 = torch.distributions.Normal(torch.zeros_like(qz2_x.mean), torch.ones_like(qz2_x.mean))

        # p(y|c)
        if self.bernulli:
            if test:
                py_c = torch.distributions.Bernoulli(logits = self.reasoner(c_pred))
            else:
                py_c = torch.distributions.Bernoulli(logits = self.reasoner(c.float()))
            # y_pred = py_c.sample()
            # y_pred = sample_bernoulli(py_c)
            y_pred = self._predict_task(c_pred, h)
        else:
            py_c = None
            if test:
                y_pred = self._predict_task(c_pred, h)
            else:
                y_pred = self._predict_task(c.float(), h)

        # if resample > 1:
        #     y_pred, c_pred, z2 = self.filter_y_c(y_pred, c_pred, z2)
        # elif resample == 1:
        #     y_pred = y_pred.squeeze(0)
        #     c_pred = c_pred.squeeze(0)
        #     z2 = z2.squeeze(0)

        if y_prime is None:
            y_prime = self.randomize_class((y_pred).float(), include=include)
        if auto_intervention > 0 and y_true is not None:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_pred.shape[1])[:int(math.ceil(auto_intervention*c_pred.shape[1]))]
            # flip
            c_pred_inv = c_pred_d.clone().detach()
            c_pred_inv[-int(y_true.shape[0]):, index] = 1 - c_pred_d[-int(y_true.shape[0]):, index]
            c_pred_inv = c_pred_inv[-int(y_true.shape[0]):]
            # c_pred[:int(y_true.shape[0]), index] = 1 - c_pred[:int(y_true.shape[0]), index]
            y_prime_inv = y_true
            c_pred_d = c_pred_d[:-int(y_true.shape[0])]
            c_pred_d = torch.cat((c_pred_d, c_pred_inv), dim=0)
            y_prime = y_prime[:-int(y_true.shape[0])]
            y_prime = torch.cat((y_prime, y_prime_inv), dim=0)
            

        # q(z3|z2, c, y, y')
        z2_c_y_y_prime = torch.cat((z2, c_pred_d, y_pred, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        
        if inference:
            z3 = z3_mu
        else:
            z3 = qz3_z2_c_y_y_prime.rsample(sample_shape=torch.Size())

        # p(c'|z3)
        pcprime_z3 = torch.distributions.Bernoulli(logits=self.concept_predictor(z3))
        # pcprime_z3 = None
        # pcprime_z3 = torch.distributions.ContinuousBernoulli(logits=self.concept_predictor(z3)) # Continuous Bernoulli
        # c_prime_pred = pcprime_z3.rsample()
        # c_prime_pred = sample_bernoulli(pcprime_z3)
        c_prime_pred = torch.sigmoid(self.concept_predictor(z3)*4)

        # c_prime_pred = self.dont_change(c_prime_pred, c_pred)

        oracle = Oracle()
        if c is not None:
            # _, c_cf = oracle.find_counterfactuals(c, y_pred, self.concept_list, self.concept_labels, y_prime)
            # c_cf = c_cf.to(X.device)
            c_cf = c
        # p(y'|c')

        if self.bernulli:
            if test:
                py_c_cf = torch.distributions.Bernoulli(logits = self.reasoner(c_prime_pred))
            else:
                py_c_cf = torch.distributions.Bernoulli(logits = self.reasoner(c_cf.float())) #TODO c or c_pred?
            # y_cf_pred = py_c_cf.sample()
            # y_cf_pred = sample_bernoulli(py_c_cf)
            y_cf_pred = self._predict_task(c_prime_pred, h)
        else:
            if test:
                y_cf_pred = self._predict_task(c_prime_pred, h)
            else:
                y_cf_pred = self._predict_task(c_cf.float(), h)
            py_c_cf = None
        
        # y_prime = y_prime.repeat(self.resample, 1, 1)
        # c_prime_pred = self.select_cf(c_prime_pred, c_pred, y_cf_pred, y_prime)

        # c_cf = self.explore(c_pred, y_prime, qz3_z2_c_y_y_prime, h)

        weights = torch.ones(c_prime_pred.shape[0])

        # p(z3|z2, c, y)
        z2_c_y = torch.cat((z2, c_pred_d, y_pred), dim=1)
        z3_mu = self.concept_mean_z3_predictor(z2_c_y)
        z3_log_var = self.concept_var_z3_predictor(z2_c_y)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        pz3_z2_c_y = torch.distributions.Normal(z3_mu, z3_sigma)

        # extract explanations
        explanation, explanation_cf = {}, {}
        if explain:
            for class_id in range(self.n_classes):
                c_preds_explanation, c_preds_explanation_cf = self.extract_counterfactual_explanation(y_pred, class_id, c_pred, c_prime_pred, h)
                explanation_cf[class_id] = {'explanation': c_preds_explanation, 'counterfactual': c_preds_explanation_cf}

        return c_pred, y_pred, explanation, c_prime_pred, y_cf_pred, y_prime, explanation_cf, p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime, pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, c_pred_d
    
    def training_step(self, I, batch_idx):
        if batch_idx == 0:
            self.epoch += 1
        if self.epoch == 1000:
            for param in self.encoder.parameters():
                param.requires_grad = False
            # for param in self.concept_mean_predictor.parameters():
            #     param.requires_grad = False
            for param in self.concept_predictor.parameters():
                param.requires_grad = False
            # for param in self.reasoner.parameters():
            #     param.requires_grad = False
        X, c_true, y_true = self._unpack_input(I)
        # self.actual_resample = self.resample
        self.actual_resample = 0
        if self.reconstruction:
            intervention_list = [0.1, 0.2, 0.5]
            # random int between 0 and 2
            rand_int = random.randint(0,2)
            p = intervention_list[rand_int]
        else:
            p = 0
        if self.epoch > -1:
            inference = False
        else:
            inference = True
        (c_pred, y_pred, explanation,
         c_prime, y_cf, y_prime, explanation_cf, 
         p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
         pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, c_pred_d) = self.forward(X, c_true, y_true=y_true[:int(X.shape[0]*0.2)], test=True, auto_intervention=p, inference=inference)
        int_concept_loss, int_task_loss = 0, 0
        int_concept_accuracy, int_task_accuracy = 0, 0

        if self.reconstruction:
            z2 = z2[:-int(X.shape[0]*0.2)]
            z3 = z3[:-int(X.shape[0]*0.2)]
            c_true = torch.cat((c_true, c_true[:int(X.shape[0]*0.2)]), dim=0)
            c_prime_int = c_prime[-int(X.shape[0]*0.2):]
            c_prime = c_prime[:-int(X.shape[0]*0.2)]
            y_pred = y_pred[:-int(X.shape[0]*0.2)]
            c_pred = c_pred[:-int(X.shape[0]*0.2)]

        # compute loss

        # KL( p(z2) || q(z2|x))
        kl_loss_z2 = torch.distributions.kl_divergence(p_z2, qz2_x).mean()

        # KL( p(z3|z2,c,y) || q(z3|z2,c,y,y'))
        kl_loss_z3 = torch.distributions.kl_divergence(pz3_z2_c_y, qz3_z2_c_y_y_prime).mean()

        kl_loss_dist = torch.distributions.kl_divergence(p_z2, pz3_z2_c_y).mean()

        kl_q_dist = torch.distributions.kl_divergence(qz2_x, qz3_z2_c_y_y_prime).mean()

        dist_loss = torch.norm(z2 - z3, p=2, dim=1).mean()
        hamming_loss = torch.abs(torch.norm((c_pred > 0.5).float() - (c_prime > 0.5).float(), p=0, dim=-1)).mean()

        if self.reconstruction:
            reconstruction_loss = self.bce_log(c_prime_int, c_true[:int(X.shape[0]*0.2)].float())
            print(reconstruction_loss)
        else:
            reconstruction_loss = 0

        # E_{c' ~ q(c'|c,y')}[-log p(c'|c))
        # admissibility_cf = torch.mean(- pcprime_z3.log_prob(c_cf.float()))
        admissibility_cf = 0

        if self.bernulli:
            # -log(p(y|c))
            task_loss = torch.mean(- py_c.log_prob(y_true.float()))
            # E_{c' ~ q(c'|c,y')}[ -log(p(y'|c'))]
            task_loss_cf = torch.mean(- py_c_cf.log_prob(y_prime.float()))
        else:
            try:
                task_loss = self.cross_entropy(y_pred, y_true.float().argmax(dim=-1))
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float().argmax(dim=-1))
            except:
                task_loss = self.cross_entropy(y_pred, y_true.float())
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float())

        # E_{c' ~ q(z2 | x)}[-log(p(c|z2))]
        concept_loss = self.bce(c_pred, c_true.float())
        concept_loss2 = self.bce(c_pred_d, c_true.float())

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_pred.squeeze().cpu().detach())
        c_true = c_true[:int(c_pred.shape[0])]
        concept_accuracy = roc_auc_score(c_true.cpu(), c_pred.squeeze().cpu().detach())

        try:
            task_cf_accuracy = roc_auc_score(y_prime.squeeze().cpu(), y_cf.squeeze().cpu().detach(), average='micro')
        except:
            task_cf_accuracy = 0

        
        loss = self.task_weight*task_loss + 1*concept_loss + 1*concept_loss2 + 0.02*task_loss_cf + 0.2*(kl_loss_z2  + kl_loss_z3) + 0.2*kl_loss_dist + 0.03*dist_loss # + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf       

        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        self.actual_resample = self.resample

        if self.epoch > -1:
            inference = False
        else:
            inference = True

        (c_pred, y_pred, explanation,
         c_prime, y_cf, y_prime, explanation_cf, 
         p_z2, qz2_x, pz3_z2_c_y, qz3_z2_c_y_y_prime,
         pcprime_z3, py_c, py_c_cf, pc_z2, c_cf, weights, z2, z3, c_pred_d) = self.forward(X, test=True, inference=inference)

        if self.reconstruction:
            z2 = z2[:-int(X.shape[0]*0.2)]
            z3 = z3[:-int(X.shape[0]*0.2)]
            c_true = torch.cat((c_true, c_true[:int(X.shape[0]*0.2)]), dim=0)
            c_prime_int = c_prime[-int(X.shape[0]*0.2):]
            c_prime = c_prime[:-int(X.shape[0]*0.2)]
            y_pred = y_pred[:-int(X.shape[0]*0.2)]
            c_pred = c_pred[:-int(X.shape[0]*0.2)]

        # compute loss

        # KL( p(z2) || q(z2|x))
        kl_loss_z2 = torch.distributions.kl_divergence(p_z2, qz2_x).mean()

        # KL( p(z3|z2,c,y) || q(z3|z2,c,y,y'))
        kl_loss_z3 = torch.distributions.kl_divergence(pz3_z2_c_y, qz3_z2_c_y_y_prime).mean()

        kl_loss_dist = torch.distributions.kl_divergence(p_z2, pz3_z2_c_y).mean()

        kl_q_dist = torch.distributions.kl_divergence(qz2_x, qz3_z2_c_y_y_prime).mean()

        dist_loss = torch.norm(z2 - z3, p=2, dim=1).mean()
        hamming_loss = torch.abs(torch.norm((c_pred > 0.5).float() - (c_prime > 0.5).float(), p=0, dim=-1)).mean()

        if self.reconstruction:
            reconstruction_loss = self.bce_log(c_prime_int, c_true[:int(X.shape[0]*0.2)].float())
        else:
            reconstruction_loss = 0

        # E_{c' ~ q(c'|c,y')}[-log p(c'|c))
        # admissibility_cf = torch.mean(- pcprime_z3.log_prob(c_cf.float()))
        admissibility_cf = 0

        if self.bernulli:
            # -log(p(y|c))
            task_loss = torch.mean(- py_c.log_prob(y_true.float()))
            # E_{c' ~ q(c'|c,y')}[ -log(p(y'|c'))]
            task_loss_cf = torch.mean(- py_c_cf.log_prob(y_prime.float()))
        else:
            try:
                task_loss = self.cross_entropy(y_pred, y_true.float().argmax(dim=-1))
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float().argmax(dim=-1))
            except:
                task_loss = self.cross_entropy(y_pred, y_true.float())
                task_loss_cf = self.cross_entropy(y_cf, y_prime.float())

        # E_{c' ~ q(z2 | x)}[-log(p(c|z2))]
        # concept_loss = self.bce(c_pred, c_true.float())
        concept_loss = torch.mean(- pc_z2.log_prob(c_true.float()))
        # concept_loss = torch.mean(- p_z2.log_prob(c_true.float()))

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_pred.squeeze().cpu().detach())
        c_true = c_true[:int(c_pred.shape[0])]
        concept_accuracy = roc_auc_score(c_true.cpu(), c_pred.squeeze().cpu().detach())

        try:
            task_cf_accuracy = roc_auc_score(y_prime.squeeze().cpu(), y_cf.squeeze().cpu().detach(), average='micro')
        except:
            task_cf_accuracy = 0

        
        loss = 1*task_loss + 10*concept_loss + 0.2*task_loss_cf + 2*(kl_loss_z2  + kl_loss_z3) + 1.5*kl_loss_dist + 0.4*dist_loss # + reconstruction_loss # + 0.0*hamming_loss + 0*kl_q_dist # + 2*admissibility_cf  

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_pred.squeeze().cpu().detach())
        concept_accuracy = roc_auc_score(c_true.cpu(), c_pred.squeeze().cpu().detach())
        concept_accuracy2 = roc_auc_score(c_true.cpu(), c_pred_d.squeeze().cpu().detach())
        try:
            task_cf_accuracy = roc_auc_score(y_prime.squeeze().cpu(), y_cf.squeeze().cpu().detach(), average='micro')
        except:
            task_cf_accuracy = 0
        
        val_acc = np.float32((task_accuracy + concept_accuracy + task_cf_accuracy) / 3)
        self.log("val_acc", val_acc)
        return loss


class StandardDCR(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 temperature: float = 10, logic: Logic = ProductTNorm(), explanation_mode: str = 'local',
                 task_weight: float = 0.1):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight)
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature)
        self.cross_entropy = self.bce

    def forward(self, X, explain=False, explanation_mode='global'):
        embeddings = self.encoder(X)
        c_preds = self.relation_classifiers(embeddings)
        c_preds = torch.sigmoid(c_preds)
        y_preds = self.reasoner(embeddings, c_preds)
        explanation = None
        if explain:
            explanation = self.reasoner.explain(embeddings, c_preds, explanation_mode,
                                                self.concept_names, self.task_names)
        return c_preds, y_preds, explanation

class CounterfactualDCR_V3(CounterfactualCBM_V3):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int, concept_set: torch.Tensor, concept_labels: torch.Tensor,
                 shield: torch.Tensor = None, train_intervention: bool = False,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 temperature: float = 10, logic: Logic = ProductTNorm(), explanation_mode: str = 'local',
                 task_weight: float = 0.1, bool_concepts: bool = True, bool_cf: bool =False, resample: int = 0):
        super().__init__(input_features, n_concepts, n_classes, emb_size, concept_set, concept_labels, shield, train_intervention,
                         learning_rate, concept_names, task_names, task_weight, bool_concepts, False, False, bool_cf, resample)
        
        self.temperature = temperature
        self.logic = logic
        self.explanation_mode = explanation_mode
        self.reasoner = ConceptReasoningLayer(emb_size, n_concepts=n_concepts, logic=logic,
                                              n_classes=n_classes, temperature=temperature)
        self.cross_entropy = torch.nn.NLLLoss()

    def predict_counterfactuals(self, X, y_prime=None, resample=1, return_cf=False, auto_intervention=0, full=False):
        h = self.encoder(X)
        z2_mu = self.concept_mean_predictor(h)
        z2_log_var = self.concept_var_predictor(h)
        z2_sigma = torch.exp(z2_log_var / 2) + EPS
        qz2_x = torch.distributions.Normal(z2_mu, z2_sigma)
        z2 = qz2_x.rsample()

        c_pred = torch.sigmoid(self.concept_predictor(z2)*4)

        if auto_intervention > 0:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_pred.shape[1])[:int(math.ceil(auto_intervention*c_pred.shape[1]))]
            # flip
            c_pred[:, index] = 1 - c_pred[:, index]

        y_pred = self._predict_task(c_pred, h)

        if y_prime is None:
            y_prime = self.randomize_class((y_pred).float())

        # q(z3|z2, c, y, y')
        z2_c_y_y_prime = torch.cat((z2, c_pred, y_pred, y_prime), dim=1)
        z3_mu = self.concept_mean_qz3_predictor(z2_c_y_y_prime)
        z3_log_var = self.concept_var_qz3_predictor(z2_c_y_y_prime)
        z3_sigma = torch.exp(z3_log_var / 2) + EPS
        qz3_z2_c_y_y_prime = torch.distributions.Normal(z3_mu, z3_sigma)
        z3 = qz3_z2_c_y_y_prime.rsample(torch.Size([resample]))

        c_prime_pred = torch.sigmoid(self.concept_predictor(z3)*4)


        y_cf_pred = torch.empty((resample, c_pred.shape[0], self.n_classes))
        for i in range(resample):
            y_cf_pred_tmp = self._predict_task(c_prime_pred[i], h)
            y_cf_pred[i] = y_cf_pred_tmp

        c_prime_full = c_prime_pred.clone()
        y_cf_pred_full = y_cf_pred.clone()

        c_prime_pred, y_cf_pred = self.filter_counterfactuals(c_prime_pred, c_pred, y_cf_pred, y_prime, z3, qz3_z2_c_y_y_prime)

        weights = torch.ones(c_prime_pred.shape[0])
        
        y_cf_pred = self._predict_task(c_prime_pred, h)

        if auto_intervention > 0:
            if full:
                return y_cf_pred, c_prime_pred, y_pred, c_pred
            return y_cf_pred, c_prime_pred, y_pred, c_pred

        if return_cf:
            return y_cf_pred, c_prime_pred

        return y_cf_pred


    def _predict_task(self, c, embeddings=None):
        return torch.log(self.reasoner(embeddings, c))

    def extract_counterfactual_explanation(self, y_preds, class_id, c_preds, c_cf, embeddings):
        _, _, filter_attn = self.reasoner(embeddings.detach(), c_preds.detach(), return_attn=True)

        class_mask = y_preds[:, class_id] > self.classification_threshold
        c_preds_class = (c_preds[class_mask] > self.classification_threshold).int()
        filter_attn_class = (filter_attn[class_mask, :, class_id] > self.classification_threshold).int() * 2 - 1
        c_preds_explanation = c_preds_class * filter_attn_class

        c_preds_class_cf = (c_cf[class_mask] > self.classification_threshold).int()
        c_preds_explanation_cf = c_preds_class_cf * filter_attn_class
        return c_preds_explanation, c_preds_explanation_cf
    
class ConceptVCNet(StandardCBM):
    def __init__(self, input_features: int, n_concepts: int, n_classes: int, emb_size: int,
                 shield: torch.Tensor = None, train_intervention: bool = False,
                 learning_rate: float = 0.01, concept_names: list = None, task_names: list = None,
                 task_weight: float = 0.1, bool_concepts: bool = True, deep: bool = True):
        super().__init__(input_features, n_concepts, n_classes, emb_size, learning_rate, concept_names, task_names, task_weight, bool_concepts, deep)

        self.concept_cf_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts + n_classes, n_concepts), torch.nn.LeakyReLU(), torch.nn.Linear(n_concepts, n_concepts))
        self.concept_predictor = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, n_concepts))
        self.concept_mean_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts + n_classes, n_concepts), torch.nn.LeakyReLU(), torch.nn.Linear(n_concepts, n_concepts))
        self.concept_var_predictor = torch.nn.Sequential(torch.nn.Linear(n_concepts + n_classes, n_concepts), torch.nn.LeakyReLU(), torch.nn.Linear(n_concepts, n_concepts))
        
        self.classification_loss = torch.nn.CrossEntropyLoss()
        self.concept_loss = torch.nn.BCELoss()
        self.classification_threshold = 0.5 if bool_concepts else 0

    def _predict_concepts(self, embeddings, predictor):
        c_preds = predictor(embeddings)
        c_preds = torch.sigmoid(c_preds)
        return c_preds

    def _predict_task(self, c, embeddings=None):
        y_preds = self.reasoner(c)
        return y_preds
    
    def counterfactual_times(
        self,
        test_dl,
        accelerator,
        return_cf=True
    ):
        total = 0
        found = 0
        total_time = 0
        y_prime_pred = torch.empty((0, self.n_classes))
        y_prime_list = torch.empty((0, self.n_classes))
        c_preds_list = torch.empty((0, self.n_concepts))
        c_cf_list = torch.empty((0, self.n_concepts))
        correct = 0
        for x, c, y in test_dl:
            x = x
            y = y
            start = time.time() 
            y_prime = self.randomize_class(y, include=False)
            y_cf_logits, c_cf_pred, _, c_preds = self.predict_counterfactuals(
                x, y_cf_target=y_prime
            )
            time_taken = time.time() - start
            total_time += time_taken

            y_prime_pred = torch.cat([y_prime_pred, y_cf_logits], dim=0)
            y_prime_list = torch.cat([y_prime_list, y_prime], dim=0)
            c_preds_list = torch.cat([c_preds_list, c_preds], dim=0)
            c_cf_list = torch.cat([c_cf_list, c_cf_pred], dim=0)
            
            pred = torch.argmax(y_cf_logits, dim=-1)
            y_p = torch.argmax(y_prime, dim=-1)

            filter = pred != y_p
            total += y.shape[0]
            correct += (pred == y_p).sum()
        print(y_prime_pred.shape, y_prime_list.shape)
        print(correct/total)
        roc_cf = roc_auc_score(y_prime_list.cpu().detach().numpy(), y_prime_pred.cpu().detach().numpy())
        if return_cf:
            return total_time, roc_cf, c_cf_list, y_prime_list, c_preds_list, y_prime_pred
        return total_time, roc_cf
    
    def predict_counterfactuals(self, X, y_cf_target=None, auto_intervention=0.0):
        # standard forward pass
        embeddings = self.encoder(X)
        # predict concepts
        c_preds = self._predict_concepts(embeddings, self.concept_predictor)

        if auto_intervention > 0:
            # flip auto_intervention percentage of c_pred
            # index to flip
            index = torch.randperm(c_preds.shape[1])[:int(math.ceil(auto_intervention*c_preds.shape[1]))]
            # flip
            c_preds[:, index] = 1 - c_preds[:, index]

        # predict task
        y_preds = self._predict_task(c_preds, embeddings)
        y_softmax = torch.softmax(y_preds, dim=-1)
        if y_cf_target is None:
            y_cf_target = self.randomize_class((y_preds).float())

        # counterfactual generation
        # get parameters of normal distribution
        mu_cf = self.concept_mean_predictor(torch.cat([c_preds, y_softmax], dim=-1))
        log_var_cf = self.concept_var_predictor(torch.cat([c_preds, y_softmax], dim=-1))

        # sample z from q
        sigma_cf = torch.exp(log_var_cf / 2) + EPS
        q_cf = torch.distributions.Normal(mu_cf, sigma_cf)
        z_cf = q_cf.rsample()

        # predict concept counterfactual
        y_desired = y_cf_target

        zy_cf = torch.cat([z_cf, y_desired], dim=1)
        c_cf = self._predict_concepts(zy_cf, self.concept_cf_predictor)

        # counterfactual task predictions
        y_cf = self._predict_task(c_cf, embeddings)

        return y_cf, c_cf, y_preds, c_preds

    def forward(self, X, c_int=None, y_int=None, explain=False, explanation_mode='local', auto_intervention=True):
        # standard forward pass
        embeddings = self.encoder(X)

        # predict concepts
        c_preds = self._predict_concepts(embeddings, self.concept_predictor)
        # predict task
        y_preds = self._predict_task(c_preds, embeddings)
        y_current = torch.softmax(y_preds, dim=-1)
        
        mu_cf = self.concept_mean_predictor(torch.cat([c_preds, y_current], dim=-1))
        log_var_cf = self.concept_var_predictor(torch.cat([c_preds, y_current], dim=-1))

        # sample z from q
        sigma_cf = torch.exp(log_var_cf / 2) + EPS
        q_cf = torch.distributions.Normal(mu_cf, sigma_cf)
        z_cf = q_cf.rsample()

        zy_cf = torch.cat([z_cf, y_current], dim=1)
        c_cf = self._predict_concepts(zy_cf, self.concept_cf_predictor)

        return c_preds, y_preds, c_cf, q_cf

    def training_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, c_cf, q_cf = self.forward(X)

        # losses
        q_multivariate_normal = torch.distributions.Normal(torch.zeros_like(q_cf.mean), torch.ones_like(q_cf.mean))
        kl_loss_cf = torch.distributions.kl_divergence(q_cf, q_multivariate_normal).mean()
        reconstruction_loss = self.bce(c_cf, c_preds)
        concept_loss = self.concept_loss(c_preds, c_true.float())
        task_loss = self.classification_loss(y_preds,  y_true.float().argmax(dim=-1))
        loss = concept_loss + 0.5*task_loss + 0.2*reconstruction_loss + 0.8*kl_loss_cf 

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_preds.squeeze().cpu().detach())
        concept_accuracy = roc_auc_score(c_true.cpu(), c_preds.squeeze().cpu().detach())

        print(f'Epoch {self.current_epoch}: task: {task_accuracy:.4f} concept: {concept_accuracy:.4f}')
        return loss

    def validation_step(self, I, batch_idx):
        X, c_true, y_true = self._unpack_input(I)

        c_preds, y_preds, c_cf, q_cf = self.forward(X)

        q_multivariate_normal = torch.distributions.Normal(torch.zeros_like(q_cf.mean), torch.ones_like(q_cf.mean))
        kl_loss_cf = torch.distributions.kl_divergence(q_cf, q_multivariate_normal).mean()
        reconstruction_loss = self.bce(c_cf, c_preds.float())
        concept_loss = self.concept_loss(c_preds, c_true.float())
        task_loss = self.classification_loss(y_preds,  y_true.float().argmax(dim=-1))
        # loss = concept_loss + 0.5*task_loss #+ 0.5*task_cf_loss + 0.1*distance_loss + 0.01*kl_loss + 0.01*kl_loss_cf + 0.2*int_concept_loss + 0.2*int_task_loss
        loss = concept_loss + 0.5*task_loss + 0.5*reconstruction_loss + 0.8*kl_loss_cf 
        print(reconstruction_loss, kl_loss_cf)

        task_accuracy = roc_auc_score(y_true.squeeze().cpu(), y_preds.squeeze().cpu().detach())
        concept_accuracy = roc_auc_score(c_true.cpu(), c_preds.squeeze().cpu().detach())

        self.log("val_acc", -loss)
        return loss

class VAE(NeuralNet):
    def __init__(self, input_dims, n_classes, encoded_size=5, learning_rate=0.01):
        super().__init__(input_dims, n_classes, encoded_size, learning_rate)
        self.encoder_mean = torch.nn.Sequential(torch.nn.Linear(input_dims + n_classes, encoded_size), torch.nn.LeakyReLU(), torch.nn.Linear(encoded_size, encoded_size))
        self.encoder_var = torch.nn.Sequential(torch.nn.Linear(input_dims + n_classes, encoded_size), torch.nn.LeakyReLU(), torch.nn.Linear(encoded_size, encoded_size))
        self.decoder_mean = torch.nn.Sequential(torch.nn.Linear(encoded_size + n_classes, encoded_size), torch.nn.LeakyReLU(), torch.nn.Linear(encoded_size, input_dims))
        self.encoder_mean_y = torch.nn.Sequential(torch.nn.Linear(n_classes, encoded_size), torch.nn.LeakyReLU(), torch.nn.Linear(encoded_size, encoded_size))
        self.encoder_var_y = torch.nn.Sequential(torch.nn.Linear(n_classes, encoded_size), torch.nn.LeakyReLU(), torch.nn.Linear(encoded_size, encoded_size))

    def encoder(self, x):
        mean = self.encoder_mean(x)
        var = self.encoder_var(x)
        sigma = torch.exp(var / 2) + EPS
        return mean, sigma
    
    def prior(self, y):
        mean = self.encoder_mean_y(y)
        var = self.encoder_var_y(y)
        sigma = torch.exp(var / 2) + EPS
        return mean, sigma

    def decoder(self, z):
        mean = self.decoder_mean(z)
        return mean

    def forward(self, x, c):
        """
        x: input instance
        c: target y
        """
        # c = torch.tensor(c).float()
        res = {}
        mc_samples = 1 #30
        em, ev = self.encoder(torch.cat((x, c), 1))
        res['em'] = em
        res['ev'] = ev
        res['z'] = []
        res['x_pred'] = []
        res['mc_samples'] = mc_samples
        q_cf = torch.distributions.Normal(em, ev)
        p_em, p_ev = self.prior(c)
        res['p_em'] = p_em
        res['p_ev'] = p_ev
        for i in range(mc_samples):
            z = q_cf.rsample()
            x_pred = torch.sigmoid(self.decoder(torch.cat((z, c), 1)))
            res['z'].append(z)
            res['x_pred'].append(x_pred)
        return res

    def compute_elbo(self, x, c, model):
        c= c.clone().detach().float()
        # c=c.view(c.shape[0], 1)
        em, ev = self.encoder(torch.cat((x,c),1))
        q_cf = torch.distributions.Normal(em, ev)
        em, ev = self.prior(c)
        q_multivariate_normal = torch.distributions.Normal(em, ev)
        kl_divergence = torch.distributions.kl_divergence(q_cf, q_multivariate_normal).mean()

        z = q_cf.rsample()
        dm= torch.sigmoid(self.decoder(torch.cat((z,c),1)))
        log_px_z = torch.tensor(0.0)

        x_pred= dm
        return torch.mean(log_px_z), kl_divergence, x, x_pred, model.forward(x_pred)

def hinge_loss(input, target, margin=1):
    """
    reference:
    - https://github.com/interpretml/DiCE/blob/a772c8d4fcd88d1cab7f2e02b0bcc045dc0e2eab/dice_ml/explainer_interfaces/dice_pytorch.py#L196-L202
    - https://en.wikipedia.org/wiki/Hinge_loss
    """
    # input = torch.log((abs(input - 1e-6) / (1 - abs(input - 1e-6))))
    # all_ones = torch.ones_like(target)
    # target = 2 * target - all_ones
    # loss = all_ones - torch.mul(target, input)
    # loss = F.relu(loss)
    # return torch.norm(loss)
    mml = torch.nn.MultiMarginLoss(margin=margin)
    input = torch.sigmoid(input)
    #input = torch.softmax(input, dim=-1)
    target = target.argmax(dim=-1)
    loss = mml(input, target)

    return loss

class CCHVAE(NeuralNet):
    """
    Refer to https://github.com/carla-recourse/CARLA/blob/main/carla/recourse_methods/catalog/cchvae/model.py
    """
    def __init__(self, input_features: int, n_classes: int, emb_size: int, model,
                 learning_rate: float = 0.01):
        """
        config: basic configs
        model: the black-box model to be explained
        """
        super().__init__(input_features, n_classes, emb_size, learning_rate)

        self.enc_dims = emb_size
        self.dec_dims = emb_size
        self.concept_size = input_features
        self.model  = model
        self.encoder_mean = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.encoder_var = torch.nn.Sequential(torch.nn.Linear(input_features, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, emb_size))
        self.decoder_mean = torch.nn.Sequential(torch.nn.Linear(emb_size, emb_size), torch.nn.LeakyReLU(), torch.nn.Linear(emb_size, input_features))
        self.lr = learning_rate
        self.continous_cols = input_features
        self.automatic_optimization = False

    def set_model(self, model):
        self.model = model 
        for param in self.model.parameters():
            param.requires_grad = False

    def predict(self, x):
        return self.model.predict(x)
    
    def encoder(self, x):
        mean = self.encoder_mean(x)
        var = self.encoder_var(x)
        sigma = torch.exp(var / 2) + EPS
        return mean, sigma

    def decoder(self, z):
        reconstruction = self.decoder_mean(z)
        reconstruction = torch.sigmoid(reconstruction)
        return reconstruction
    
    def randomize_class(self, a, include=True):
        # Get the number of classes and the number of samples
        num_classes = a.size(1)
        num_samples = a.size(0)

        # Generate random indices for each row to place 1s, excluding the original positions
        random_indices = torch.randint(0, num_classes, (num_samples,))

        # Ensure that the generated indices are different from the original positions
        # TODO we inclue also same label to make sure that every class is represented 
        if not include:
          original_indices = a.argmax(dim=1)
          random_indices = torch.where(random_indices == original_indices, (random_indices + 1) % num_classes, random_indices)

        # Create a second tensor with 1s at the random indices
        b = torch.zeros_like(a)
        b[torch.arange(num_samples), random_indices] = 1
        return b

    def _hyper_sphere_coordindates(
        self, x, high: int, low: int, n_search_samples: int
    ):
        """
        :param n_search_samples: int > 0
        :param x: input point array
        :param high: float>= 0, h>l; upper bound
        :param low: float>= 0, l<h; lower bound
        :return: candidate counterfactuals & distances
        """
        delta_instance = torch.randn(n_search_samples, x.size(2))
        dist = (
            torch.rand(n_search_samples) * (high - low) + low
        )  # length range [l, h)
        norm_p = torch.norm(delta_instance, p=1, dim=1)
        d_norm = torch.divide(dist, norm_p).reshape(-1, 1)  # rescale/normalize factor
        delta_instance = torch.multiply(delta_instance, d_norm)
        candidate_counterfactuals = x + delta_instance
        return candidate_counterfactuals, dist
    
    def counterfactual_times(
        self,
        test_dl,
        accelerator,
        rerun=False,
        n_times = 5,
        return_cf = False
    ):
        total = 0
        found = 0
        total_time = 0
        y_prime_pred = torch.empty((0, self.n_classes))
        y_prime_list = torch.empty((0, self.n_classes))
        c_cf_list = torch.empty((0, self.concept_size))
        c_pred_list = torch.empty((0, self.concept_size))
        correct = 0
        for c, _, y in test_dl:
            c = c
            y = y
            start = time.time() 
            y_prime = self.randomize_class(y, include=False)
            y_cf_logits = self.generate_cf(
                c, y_prime=y_prime
            )
            y_cf_logits, c_cf_pred, y_prime, c = y_cf_logits
            c_pred_list = torch.cat([c_pred_list, c], dim=0)
            c_cf_list = torch.cat([c_cf_list, c_cf_pred], dim=0)
            time_taken = time.time() - start
            total_time += time_taken

            y_prime_pred = torch.cat([y_prime_pred, y_cf_logits], dim=0)
            y_prime_list = torch.cat([y_prime_list, y_prime], dim=0)
            
            pred = torch.argmax(y_cf_logits, dim=-1)
            y_p = torch.argmax(y_prime, dim=-1)

            filter = pred != y_p
            total += y.shape[0]
            correct += (pred == y_p).sum()
        roc_cf = roc_auc_score(y_prime_list.cpu().detach().numpy(), y_prime_pred.cpu().detach().numpy())
        if return_cf:
            return total_time, roc_cf, c_cf_list, c_pred_list, y_prime_pred, y_prime_list
        return total_time, roc_cf

    def generate_cf(self, x, y_prime, auto_intervention=0, c_true=None):
        # params
        n_search_samples = 300; count = 0; max_iter = 1000; step=0.1
        low = 0; high = step
        if c_true is None:
            c_true = x.clone()

        if auto_intervention > 0:
            index = torch.randperm(x.shape[1])[:int(math.ceil(auto_intervention*x.shape[1]))]
            # flip
            x[:, index] = 1 - x[:, index]
            y_preds = self.model.forward(x)

        # vectorize z
        mu, sigma = self.encoder(x)
        q_cf = torch.distributions.Normal(mu, sigma)
        z = q_cf.rsample()

        z_rep = torch.repeat_interleave(
            z.unsqueeze(1), n_search_samples, dim=1
        )

        y_prime = torch.repeat_interleave(
            y_prime.unsqueeze(1), n_search_samples, dim=1
        )
        x = torch.repeat_interleave(
            x.unsqueeze(1), n_search_samples, dim=1
        )
        c_true = torch.repeat_interleave(
            c_true.unsqueeze(1), n_search_samples, dim=1
        )
        
        x_ce_list = torch.empty(0, x.shape[2])
        y_ce = torch.empty(0, y_prime.shape[2])
        y_prime_list = torch.empty(0, y_prime.shape[2])
        x_list = torch.empty(0, x.shape[2])
        c_list = torch.empty(0, x.shape[2])
        remained = x.shape[0]
        progress_bar = tqdm(total=max_iter, desc="Processing")
        with torch.no_grad():
            while count <= max_iter and remained > 0:
                count = count + 1
                progress_bar.update(1)
                # STEP 1 -- SAMPLE POINTS on hyper sphere around instance
                
                latent_neighbourhood, _ = self._hyper_sphere_coordindates(z_rep, high, low, n_search_samples)
                x_ce = self.decoder(latent_neighbourhood)

                # STEP 2 -- COMPUTE l1 norms
                distances = torch.abs((x_ce - x)).sum(dim=2)

                # counterfactual labels
                y_candidate = self.model(x_ce)
                indeces = y_candidate.argmax(dim=-1) != y_prime.argmax(dim=-1)
                distances[indeces] = 10000

                found = indeces.sum(dim=-1) < n_search_samples
                if found.sum() > 0:
                    # certain candidates generated
                    distances_found = distances[found]
                    min_index = torch.argsort(distances_found, dim=-1)[:, 0]
                    cf_found = x_ce[found]
                    y_candidate_tmp = y_candidate[found]
                    x_ce_list = torch.cat((x_ce_list, cf_found[torch.tensor(list(range(cf_found.shape[0]))), min_index, :]), dim=0)
                    y_ce = torch.cat((y_ce, y_candidate_tmp[torch.tensor(list(range(cf_found.shape[0]))), min_index, :]), dim=0)
                    y_prime_list = torch.cat((y_prime_list, y_prime[found][torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
                    x_list = torch.cat((x_list, x[found][torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
                    c_list = torch.cat((c_list, c_true[found][torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
                remained -= found.sum()
                z_rep = z_rep[~found]
                x = x[~found]
                y_prime = y_prime[~found]
                c_true = c_true[~found]
                low = high
                high = low + step
                print(remained)
            if remained > 0:
                not_found = ~found
                print('remained ', not_found.sum())
                distances_found = distances[not_found]
                min_index = torch.argsort(distances_found, dim=-1)[:, 0]
                cf_found = x_ce[not_found]
                y_candidate = y_candidate[not_found]
                x_ce_list = torch.cat((x_ce_list, cf_found[torch.tensor(list(range(cf_found.shape[0]))), min_index, :]), dim=0)
                y_ce = torch.cat((y_ce, y_candidate[torch.tensor(list(range(cf_found.shape[0]))), min_index, :]), dim=0)
                y_prime_list = torch.cat((y_prime_list, y_prime[torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
                x_list = torch.cat((x_list, x[torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
                c_list = torch.cat((c_list, c_true[torch.tensor(list(range(cf_found.shape[0]))), 0, :]), dim=0)
            print(torch.unique(y_prime_list.argmax(dim=-1)))
            print(x_ce_list.shape, y_ce.shape, y_prime_list.shape, x_list.shape)

        y_preds = self.model.forward(x_list)
        if auto_intervention > 0:
            return y_ce, x_ce_list, y_prime_list, x_list, y_preds, c_list
        return y_ce, x_ce_list, y_prime_list, x_list
    
    def forward(self, x):
        em, ev = self.encoder(x)
        q_cf = torch.distributions.Normal(em, ev)
        z = q_cf.rsample()
        x_pred = self.decoder(z)
        return x_pred, q_cf

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, _, _ = batch
        x_pred, q = self.forward(x)

        mse_loss = F.mse_loss(x_pred, x)
        p_multivariate_normal = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.mean))
        kl_divergence = torch.distributions.kl_divergence(q, p_multivariate_normal).mean()
        loss = mse_loss + 0.5*kl_divergence
        print(mse_loss, kl_divergence)

        self.manual_backward(loss, retain_graph=True)
        opt.step()
        self.log('train/loss', -loss)

        return loss


    def validation_step(self, batch, batch_idx):
        x, _, y = batch
        x_pred, q = self.forward(x)

        mse_loss = F.mse_loss(x_pred, x)
        p_multivariate_normal = torch.distributions.Normal(torch.zeros_like(q.mean), torch.ones_like(q.mean))
        kl_divergence = torch.distributions.kl_divergence(q, p_multivariate_normal).mean()
        loss = mse_loss + kl_divergence

        self.log('val_acc', -loss)
        return loss
    
    def validation_epoch_end(self, val_outs):
        return

# Cell
class VAE_CF(NeuralNet):
    def __init__(self, input_features: int, n_classes: int, emb_size: int, model,
                 learning_rate: float = 0.01):
        """
        config: basic configs
        model: the black-box model to be explained
        """
        super().__init__(input_features, n_classes, emb_size, learning_rate)

        self.enc_dims = emb_size
        self.dec_dims = emb_size
        self.concept_size = input_features
        self.model  = model
        self.vae = VAE(input_dims=input_features, n_classes=n_classes, encoded_size=self.enc_dims, learning_rate=learning_rate)
        self.lr = learning_rate
        self.continous_cols = input_features
        self.automatic_optimization = False
        self.margin = 1
        # validity_reg set to 42.0
        # according to https://interpret.ml/DiCE/notebooks/DiCE_getting_started_feasible.html#Generate-counterfactuals-using-a-VAE-model
        self.validity_reg = 15.0 #TOCHANGE 1.0

    def set_model(self, model):
        self.model = model 
        for param in self.model.parameters():
            param.requires_grad = False

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.lr)

    def predict(self, x):
        y_preds = self.model.forward(x)
        return y_preds
    
    def randomize_class(self, a, include=True):
        # Get the number of classes and the number of samples
        num_classes = a.size(1)
        num_samples = a.size(0)

        # Generate random indices for each row to place 1s, excluding the original positions
        random_indices = torch.randint(0, num_classes, (num_samples,))

        # Ensure that the generated indices are different from the original positions
        # TODO we inclue also same label to make sure that every class is represented 
        if not include:
          original_indices = a.argmax(dim=1)
          random_indices = torch.where(random_indices == original_indices, (random_indices + 1) % num_classes, random_indices)

        # Create a second tensor with 1s at the random indices
        b = torch.zeros_like(a)
        b[torch.arange(num_samples), random_indices] = 1
        return b
    
    def counterfactual_times(
        self,
        test_dl,
        accelerator,
        rerun=False,
        n_times = 5,
        return_cf = False
    ):
        total = 0
        found = 0
        total_time = 0
        y_prime_pred = torch.empty((0, self.n_classes))
        y_prime_list = torch.empty((0, self.n_classes))
        c_cf_list = torch.empty((0, self.concept_size))
        c_pred_list = torch.empty((0, self.concept_size))
        correct = 0
        for c, _, y in test_dl:
            c = c
            y = y
            start = time.time() 
            y_prime = self.randomize_class(y, include=False)
            y_cf_logits = self.generate_cf(
                c, y_prime=y_prime
            )
            y_cf_logits, c_cf_pred = y_cf_logits
            time_taken = time.time() - start
            total_time += time_taken

            print(y_prime_pred.shape, y_prime.shape, y_cf_logits.shape, y.shape)

            y_prime_pred = torch.cat([y_prime_pred, y_cf_logits], dim=0)
            y_prime_list = torch.cat([y_prime_list, y_prime], dim=0)

            print(y_prime_list.argmax(dim=-1).unique())
            
            pred = torch.argmax(y_cf_logits, dim=-1)
            y_p = torch.argmax(y_prime, dim=-1)

            filter = pred != y_p
            total += y.shape[0]
            correct += (pred == y_p).sum()
            c_cf_list = torch.cat([c_cf_list, c_cf_pred], dim=0)
            c_pred_list = torch.cat([c_pred_list, c], dim=0)

        roc_cf = roc_auc_score(y_prime_list.cpu().detach().numpy(), y_prime_pred.cpu().detach().numpy())
        if return_cf:
            return total_time, roc_cf, c_cf_list, c_pred_list, y_prime_pred, y_prime_list
        return total_time, roc_cf

    def compute_loss(self, out, x, y):
        em = out['em']
        ev = out['ev']
        z = out['z']
        dm = out['x_pred']
        mc_samples = out['mc_samples']
        p_em = out['p_em']
        p_ev = out['p_ev']
        #KL Divergence
        q_cf = torch.distributions.Normal(em, ev)
        p_cf = torch.distributions.Normal(p_em, p_ev)
        # q_multivariate_normal = torch.distributions.Normal(torch.zeros_like(q_cf.mean), torch.ones_like(q_cf.mean))
        kl_divergence = torch.distributions.kl_divergence(q_cf, p_cf).mean()

        #Reconstruction Term
        #Proximity: L1 Loss
        x_pred = dm[0]
        recon_err = torch.sum(torch.abs(x - x_pred), axis=1).mean()

        #Validity
        c_y = self.model(x_pred)
        validity_loss = torch.zeros(1, device=self.device)
        validity_loss += hinge_loss(input=c_y, target=y.float(), margin=self.margin)
        
        for i in range(1, mc_samples):
            x_pred = dm[i]

            recon_err += torch.sum(torch.abs(x - x_pred), axis=1).mean()

            #Validity
            c_y = self.model(x_pred)
            validity_loss += hinge_loss(c_y, y.float(), margin=self.margin)
        
        recon_err = recon_err / mc_samples
        validity_loss = self.validity_reg * validity_loss / mc_samples
        print(recon_err, validity_loss, kl_divergence)
        # return 3*recon_err + kl_divergence + validity_loss #dsprites
        # return 1.3*recon_err + kl_divergence + validity_loss #mnist
        return recon_err + 2*kl_divergence + validity_loss #cub


    def training_step(self, batch, batch_idx):
        # batch
        opt = self.optimizers()
        opt.zero_grad()
        x, _, _ = batch
        # prediction
        y_hat = self.model.forward(x)
        # target
        y = self.randomize_class(y_hat)

        out = self.vae(x, y)
        loss = self.compute_loss(out, x, y)
        self.manual_backward(loss, retain_graph=True)
        opt.step()
        self.log('train/loss', -loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        x, _, _ = batch
        # prediction
        y_hat = self.model.forward(x)
        # target
        y = self.randomize_class(y_hat)


        out = self.vae(x, y)
        loss = self.compute_loss(out, x, y)

        _, _, _, x_pred, cf_label = self.vae.compute_elbo(x, y, self.model)

        cf_proximity = torch.abs(x - x_pred).sum(dim=1).mean()
        try:
            cf_accuracy = roc_auc_score(y.cpu(), cf_label.cpu())    
        except:
            cf_accuracy = 0
        self.log('val/val_loss', loss)
        self.log('val/proximity', cf_proximity)
        self.log('val_acc', -loss)
        print(cf_accuracy, loss, cf_proximity)

        return loss

    def validation_epoch_end(self, val_outs):
        return

    def generate_cf(self, x, y_prime, auto_intervention=0, c_true=None):
        #y_hat = self.model.forward(x)
        if auto_intervention > 0:
            index = torch.randperm(x.shape[1])[:int(math.ceil(auto_intervention*x.shape[1]))]
            # flip
            x[:, index] = 1 - x[:, index]
            y_preds = self.model.forward(x)
        out = self.vae(x, y_prime)
        x_pred = out['x_pred'][0]
        # recon_err, kl_err, x_true, x_pred, cf_label = self.vae.compute_elbo(x, y_prime, self.model)
        cf_pred = self.model(x_pred)
        if auto_intervention > 0:
            return cf_pred, x_pred, y_preds, x
        return cf_pred, x_pred

