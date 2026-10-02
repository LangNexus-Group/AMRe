import pdb
import random
import warnings

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict

from torch.utils.data import DataLoader

# For MLA
class GSPlugin():
    def __init__(self, gs_flag = True):

        super().__init__()

        dtype = torch.cuda.FloatTensor  # run on GPU
        with torch.no_grad():
            # self.Pl = torch.autograd.Variable(torch.eye(768).type(dtype))
            self.Pl = torch.autograd.Variable(torch.eye(512).type(dtype))
        self.exp_count = 0

    # @torch.no_grad()
    def before_update(self, model, before_batch_input, batch_index, len_dataloader, train_exp_counter):

        lamda = batch_index / len_dataloader + 1
        alpha = 1.0 * 0.1 ** lamda
        # x_mean = torch.mean(strategy.mb_x, 0, True)
        if train_exp_counter != 0:
            for n, w in model.named_parameters():

                if n == "module.weight":

                    r = torch.mean(before_batch_input, 0, True)
                    k = torch.mm(self.Pl, torch.t(r))
                    self.Pl = torch.sub(self.Pl, torch.mm(k, torch.t(k)) / (alpha + torch.mm(k, r)))

                    pnorm2 = torch.norm(self.Pl.data, p='fro')

                    self.Pl.data = self.Pl.data / pnorm2
                    w.grad.data = torch.mm(w.grad.data, torch.t(self.Pl.data))

def calculate_entropy(output):
    probabilities = F.softmax(output, dim=1)
    # probabilities = F.softmax(output, dim=1)
    log_probabilities = torch.log(probabilities)
    entropy = -torch.sum(probabilities * log_probabilities,dim=1)
    print(entropy)
    return entropy

def calculate_gating_weights(encoder_output_1, encoder_output_2):
    
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    
    max_entropy = max(entropy_1, entropy_2)
    
    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    
    sum_weights = gating_weight_1 + gating_weight_2
    
    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    
    return gating_weight_1, gating_weight_2

def calculate_gating_weights3(encoder_output_1, encoder_output_2, encoder_output_3):
    entropy_1 = calculate_entropy(encoder_output_1)
    entropy_2 = calculate_entropy(encoder_output_2)
    entropy_3 = calculate_entropy(encoder_output_3)
    
    max_entropy = max(entropy_1, entropy_2, entropy_3)
    
    gating_weight_1 = torch.exp(max_entropy - entropy_1)
    gating_weight_2 = torch.exp(max_entropy - entropy_2)
    gating_weight_3 = torch.exp(max_entropy - entropy_3)
    
    sum_weights = gating_weight_1 + gating_weight_2 + gating_weight_3
    
    gating_weight_1 /= sum_weights
    gating_weight_2 /= sum_weights
    gating_weight_3 /= sum_weights
    
    return gating_weight_1, gating_weight_2, gating_weight_3


# For QMF
class History(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))
        self.confidence = np.zeros((n_data))
        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness, confidence):
        #probs = torch.nn.functional.softmax(output, dim=1)
        #confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()
        data_idx = [idx[0] for idx in data_idx]

        self.correctness[data_idx] += correctness.cpu().numpy()
        self.confidence[data_idx] = confidence.cpu().detach().numpy()

    # max correctness update
    def max_correctness_update(self, epoch):
        if epoch > 1:
            self.max_correctness += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, data):
        data_min = self.correctness.min()
        #data_max = float(self.max_correctness)
        data_max = float(self.correctness.max())

        return (data - data_min) / (data_max - data_min)

    # get target & margin
    def get_target_margin(self, data_idx1, data_idx2):
        data_idx1 = data_idx1.cpu().numpy()
        cum_correctness1 = self.correctness[data_idx1]
        cum_correctness2 = self.correctness[data_idx2]
        # normalize correctness values
        cum_correctness1 = self.correctness_normalize(cum_correctness1)
        cum_correctness2 = self.correctness_normalize(cum_correctness2)
        # make target pair
        n_pair = len(data_idx1)
        target1 = cum_correctness1[:n_pair]
        target2 = cum_correctness2[:n_pair]
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.from_numpy(target).float().cuda()
        # calc margin
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()

        return target, margin
