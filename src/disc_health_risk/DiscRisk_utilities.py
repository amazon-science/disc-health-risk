# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import random

from matplotlib.colors import LinearSegmentedColormap
import torch
import pickle as pkl
import numpy as np
import enum
import os
import DiscRisk_constants as CONSTS
import sklearn
import json
from random import choices
from string import ascii_lowercase

cuda_is_available = torch.cuda.is_available()

##############################################################################################################
# Functions
##############################################################################################################

def cuda_to_numpy(x):
    if torch.is_tensor(x):
        if x.requires_grad == True:
            x = x.detach()
        if cuda_is_available:
            x = x.cpu()
        x = x.numpy()
    return x


def numpy_to_cuda(x):
    if not torch.is_tensor(x):
        x = torch.from_numpy(x)
    if cuda_is_available:
        x = x.cuda()
    return x


def construct_model_folder_name(cfg):

    random_id = "".join(choices(ascii_lowercase, k=3))

    # name of best CNN model to be saved to disk
    gender_str = genderStringFromId(cfg.GENDER_ID)
    num_bioms = min(6, len(cfg.NAMES_BIOMS))
    num_conds = min(6, len(cfg.NAMES_CONDS))
    folder_name = "../../models/{}/".format(gender_str)
    folder_name += "modid{}_{}_bioms".format(cfg.BIOM_MODEL_IDX, random_id)
    for biom_id in range(num_bioms):
        folder_name += "_{}".format(cfg.NAMES_BIOMS[biom_id])
    folder_name += "_conds"
    for cond_id in range(num_conds):
        folder_name += "_{}".format(cfg.NAMES_CONDS[cond_id])
    folder_name += "_feats"
    for num_feats in cfg.NN_ARCHITECTURE:
        folder_name += "_{}".format(num_feats)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def getBiomarkerCodesFromBiomarkerModelId(biom_model_idx):

    ###
    ### 1D ------------------------------------------------------------------------
    ###
    if biom_model_idx==0:
        names_BIOMs = ['BMXHT']
    elif biom_model_idx == 1:
        names_BIOMs = ['BMXWT']
    elif biom_model_idx == 2:
        names_BIOMs = ['BMXBMI']
    elif biom_model_idx == 3:
        names_BIOMs = ['DXDTOPF']
    elif biom_model_idx == 4:
        names_BIOMs = ['DXDTRPF']
    elif biom_model_idx == 5:
        names_BIOMs = ['BMXWAIST']
    elif biom_model_idx == 6:
        names_BIOMs = ['RIDAGEYR']
    elif biom_model_idx == 7:
        names_BIOMs = ['SLD010H']
    elif biom_model_idx == 8:
        names_BIOMs = ['PAQ635']
    elif biom_model_idx == 9:
        names_BIOMs = ['PAQ710']
    elif biom_model_idx == 10:
        names_BIOMs = ['PAQ635']
    elif biom_model_idx == 11:
        names_BIOMs = ['DBQ700']
    elif biom_model_idx == 12:
        names_BIOMs = ['CBQ505']
    elif biom_model_idx == 13:
        names_BIOMs = ['DPQ020']
    ###
    ### 2D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 14:
        names_BIOMs = ['BMXBMI','DPQ020']
    elif biom_model_idx == 15:
        names_BIOMs = ['BMXBMI','PAQ635']
    elif biom_model_idx == 16:
        names_BIOMs = ['BMXBMI','DXDTOPF']
    elif biom_model_idx == 17:
        names_BIOMs = ['BMXBMI','BMXWAIST']
    elif biom_model_idx == 18:
        names_BIOMs = ['BMXBMI','RIDAGEYR']
    elif biom_model_idx == 19:
        names_BIOMs = ['DXDTRPF','RIDAGEYR']
    elif biom_model_idx == 20:
        names_BIOMs = ['BMXWAIST','RIDAGEYR']
    elif biom_model_idx == 21:
        names_BIOMs = ['BMXWAIST','BMXHIP']
    elif biom_model_idx == 22:
        names_BIOMs = ['BMXWT','PAQ560']
    elif biom_model_idx == 23:
        names_BIOMs = ['BMXWT','BMXWAIST']
    elif biom_model_idx == 24:
        names_BIOMs = ['BMXWT','BMXHT']
    elif biom_model_idx == 25:
        names_BIOMs = ['BMXWAIST','BMXHT']
    ###
    ### 3D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 26:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXHT']
    elif biom_model_idx == 27:
        names_BIOMs = ['BMXWT', 'BMXHT', 'BMXWAIST']    # testing swapping order of biomarkers
    elif biom_model_idx == 28:
        names_BIOMs = ['BMXWAIST','BMXHT','RIDAGEYR']
    elif biom_model_idx == 29:
        names_BIOMs = ['BMXBMI','BMXWAIST','RIDAGEYR']
    elif biom_model_idx == 30:
        names_BIOMs = ['BMXWT','BMXWAIST','RIDAGEYR']
    elif biom_model_idx == 31:
        names_BIOMs = ['BMXWAIST','BMXHIP','RIDAGEYR']
    ###
    ### 4D  ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 32:
        names_BIOMs = ['BMXWT','BMXHT','DXDTRPF','RIDAGEYR']
    elif biom_model_idx == 33:
        names_BIOMs = ['BMXBMI','BMXWAIST','DXDTOPF','RIDAGEYR']
    elif biom_model_idx == 34:
        names_BIOMs = ['BMXWAIST','BMXTHICR','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 35:
        names_BIOMs = ['BMXWT','BMXWAIST','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 36:
        names_BIOMs = ['BMXWAIST','BMXHIP','RIDAGEYR','RIDRETH1']
    ###
    ### 5D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 38:
        names_BIOMs = ['BMXWT','BMXHT','DXDTRPF','RIDAGEYR','SLD010H']
    elif biom_model_idx == 39:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXHIP','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 40:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXTHICR','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 41:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXHT','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 46:
        names_BIOMs = ['BMXBMI','BMXWAIST','BMXHIP','RIDAGEYR','RIDRETH1']
    ###
    ### 6D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 42:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXTHICR','BMXHT','RIDAGEYR','RIDRETH1']
    elif biom_model_idx == 44:
        names_BIOMs = ['BMXWT', 'BMXWAIST', 'BMXTHICR', 'DXDTOPF','RIDAGEYR', 'RIDRETH1']
    elif biom_model_idx == 45:
        names_BIOMs = ['BMXWT', 'BMXWAIST', 'BMXTHICR', 'DXDTRPF','RIDAGEYR', 'RIDRETH1']
    elif biom_model_idx == 47:
        names_BIOMs = ['BMXWT', 'BMXWAIST', 'BMXHIP', 'PAQ635', 'RIDAGEYR', 'RIDRETH1']
    ###
    ### 7D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 37:
        names_BIOMs = ['BMXBMI', 'BMXWAIST', 'DXDTOPF', 'DXDTRPF', 'BMXWT', 'BMXHT','BMXTHICR']  # good for overfitting experiments
    elif biom_model_idx == 48:
        names_BIOMs = ['BMXWT','BMXWAIST','BMXHIP','PAQ635','BMXHT','RIDAGEYR','RIDRETH1']
    ###
    ### 9D ------------------------------------------------------------------------
    ###
    elif biom_model_idx == 43:
        names_BIOMs = ['BMXBMI','BMXWAIST','DXDTOPF','BMXTHICR','BMXWT','BMXHT','DXDTRPF','RIDAGEYR','RIDRETH1']

    return names_BIOMs

def genderStringFromId(gender_id):
    if gender_id == 1:
        gender_str = 'men'
    elif gender_id == 2:
        gender_str = 'women'
    elif gender_id == -1:
        gender_str = 'both'
    else:
        gender_str = 'error'
    return gender_str

def loadRawDataFromPickleFile(file_name):
    fid = open(file_name, 'rb')
    data_raw = pkl.load(fid)
    fid.close()
    return data_raw

def checkFieldsExistInData(data_raw_yslot, names_BIOMs, name_CONDs, name_SMQ, REMOVE_SMOKERS, year_slot_name):

    # check all biomarkers
    for name_BIOM in names_BIOMs:
        if 'RIDRETH1' in name_BIOM:
            name_BIOM = 'RIDRETH1'  # Could be RIDRETH1_1, RIDRETH1_2, RIDRETH1_3, RIDRETH1_4, RIDRETH1_5
        if name_BIOM == 'BMXBMI' and 'BMXWT' in data_raw_yslot.dtype.names and 'BMXHT' in data_raw_yslot.dtype.names:
            continue
        if not name_BIOM in data_raw_yslot.dtype.names:
            print(' * Warning biomarker %s not in data, skipping year slot %s' % (name_BIOM, year_slot_name))
            return False
    # check all health conditions
    for name_COND in name_CONDs:
        if not name_COND in data_raw_yslot.dtype.names:
            print(' * Warning condition %s not in data, skipping year slot %s' % (name_CONDs, year_slot_name))
            return False
    # check smokers
    if REMOVE_SMOKERS and (not name_SMQ in data_raw_yslot.dtype.names):
        print(' * Warning smokers field %s not in data, skipping year slot %s' % (name_SMQ, year_slot_name))
        return False

    return True

def ownColormap_3colors(col1,col2,col3):
    name = '3-color-colormap'
    keypoint_pos = [0.0, 0.5, 1.0]
    cdict = {'red': [(keypoint_pos[0], col1[0], col1[0]),
                     (keypoint_pos[1], col2[0], col2[0]),
                     (keypoint_pos[2], col3[0], col3[0])],
             'green': [(keypoint_pos[0], col1[1], col1[1]),
                       (keypoint_pos[1], col2[1], col2[1]),
                       (keypoint_pos[2], col3[1], col3[1])],
             'blue': [(keypoint_pos[0], col1[2], col1[2]),
                      (keypoint_pos[1], col2[2], col2[2]),
                      (keypoint_pos[2], col3[2], col3[2])]}
    return LinearSegmentedColormap(name, cdict, N=256, gamma=1.0)

def compute_bmi(wt, ht):

    bmi = np.divide(wt, np.multiply(ht, ht))
    return bmi

def appendToData(data_raw_yslot, data_global, names_data_fields):

    num_addit_data_for_yslot = data_raw_yslot.shape[0]
    [num_data_current, num_data_fields] = data_global.shape
    num_data_total = num_data_current + num_addit_data_for_yslot

    data_global_out = np.zeros((num_data_total, num_data_fields))
    data_global_out[0:num_data_current, :] = data_global

    for biom_id in range(num_data_fields):
        name_data_field = names_data_fields[biom_id]

        ethnicity_index = -1
        if 'RIDRETH1' in name_data_field:
            parts = name_data_field.split('_')
            if len(parts) == 2:
                ethnicity_index = int(parts[1])
            name_data_field = parts[0]

        if name_data_field == 'BMXBMI' and name_data_field not in data_raw_yslot.dtype.names:
            data_raw_wt = data_raw_yslot['BMXWT']  # kg
            data_raw_ht = data_raw_yslot['BMXHT']  # cm
            data_raw_ht *= 0.01  # meters
            data_raw_this = compute_bmi(data_raw_wt, data_raw_ht)
        else:
            if name_data_field in data_raw_yslot.dtype.names:
                data_raw_this = data_raw_yslot[name_data_field]
                if ethnicity_index >= 0:
                    old_dtype = data_raw_this.dtype
                    data_raw_this = (data_raw_this == ethnicity_index).astype(dtype=old_dtype)
            else:
                data_raw_this = -np.ones(num_data_total-num_data_current, dtype=data_global_out.dtype)
        data_global_out[num_data_current:num_data_total, biom_id] = data_raw_this

    return data_global_out

# def appendToDataConds(data_raw_yslot, data_CONDs, names_CONDs):
#
#     num_addit_data_for_yslot = data_raw_yslot.shape[0]
#     [num_data_current, num_conditions] = names_CONDs.shape
#     num_data_total = num_data_current + num_addit_data_for_yslot
#
#     data_CONDs_out = np.zeros((num_data_total, num_conditions))
#     data_CONDs_out[0:num_data_current,:] = data_CONDs
#
#     for cond_id in range(num_conditions):
#         data_CONDs_out[num_data_current:num_data_total,cond_id] = data_raw_yslot[names_CONDs[cond_id]]
#
#     return data_CONDs_out

def setPointColors(labels_in, col_pos, col_neg, col_unc):

    num_data = len(labels_in)

    labels_cols = np.zeros((num_data, len(col_pos)))  # output colours

    for data_id in range(num_data):

        w = labels_in[data_id]  # 0 for negative to condition and 1 for positive
        col = w * col_pos + (1 - w) * col_neg   # mixing red and green

        d = abs(w-0.5)/0.5      # distance from the middle, 0.5
        labels_cols[data_id, :] = d * col + (1 - d) * col_unc   # mixing in uncertain colour

    return labels_cols


def testNNEnsemble(NN_ensemble, bioms_norm_to, nn_net_index):
    NN_ensemb_size = len(NN_ensemble)
    with torch.no_grad():
        if nn_net_index == -1:  # use entire ensemble of pretrained networks
            net = NN_ensemble[0]
            if cuda_is_available:
                net.to('cuda:0')
            activations_individ_net = net(bioms_norm_to)  # apply first network in ensemble
            softmax_ensemble = cuda_to_numpy(torch.nn.functional.softmax(activations_individ_net.detach(), 1))
            for NN_idx in range(1, NN_ensemb_size):
                net = NN_ensemble[NN_idx]
                if cuda_is_available:
                    net.to('cuda:0')
                activations_individ_net = net(bioms_norm_to)  # apply network to input biomarkers to obtain activations of output layer
                softmax_ensemble += cuda_to_numpy(torch.nn.functional.softmax(activations_individ_net.detach(), 1))
            softmax_ensemble = np.divide(softmax_ensemble, np.sum(softmax_ensemble, axis=1, keepdims=True))
            #softmax_ensemble /= NN_ensemb_size  # averaging activations or softmax?
        else:
            net = NN_ensemble[nn_net_index]
            if cuda_is_available:
                net.to('cuda:0')
            activations_individ_net = net(bioms_norm_to)  # apply selected network only
            softmax_ensemble = cuda_to_numpy(torch.nn.functional.softmax(activations_individ_net.detach(), 1))

    return softmax_ensemble


def calculate_inverse_frequency_class_weights(labels):
    num_labels = labels.shape[0]
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        labels_single = labels[:, -1]
    else:
        labels_single = labels
    count_positive = torch.sum(labels_single)
    count_negative = num_labels - count_positive
    if count_positive > 0 and count_negative > 0:
        inv_positive_freq = torch.div(num_labels, count_positive)
        inv_negative_freq = torch.div(num_labels, count_negative)
        weights = torch.multiply(labels_single, inv_positive_freq) \
                + torch.multiply(1.0-labels_single, inv_negative_freq)
    else:
        weights = torch.ones_like(labels_single)
    return weights


def unnorm2norm(data_unnorm, mu=None, std=None, valid_mask=None, valid_cols=None):

    if mu is None:
        if valid_mask is None:
            if data_unnorm.shape[0] == 0:
                mu = 0
            elif valid_cols is None:
                mu = np.mean(data_unnorm, axis=0, keepdims=True)
            else:
                mu = np.zeros((1, data_unnorm.shape[1]), dtype=data_unnorm.dtype)
                for d in range(data_unnorm.shape[1]):
                    if valid_cols[d]:
                        mu[:, d] = np.mean(data_unnorm, axis=0, keepdims=True)
        else:
            if np.count_nonzero(valid_mask) == 0:
                mu = 0
            else:
                mu = np.zeros((1, data_unnorm.shape[1]), dtype=data_unnorm.dtype)
                for d in range(data_unnorm.shape[1]):
                    if valid_cols is None or valid_cols[d]:
                        mu[:, d] = np.mean(data_unnorm[valid_mask[:, d], d], axis=0, keepdims=True)

    if std is None:
        if valid_mask is None:
            if data_unnorm.shape[0] == 0:
                std = 1.0
            elif valid_cols is None:
                std = np.std(data_unnorm, axis=0, keepdims=True)
            else:
                std = np.ones((1, data_unnorm.shape[1]), dtype=data_unnorm.dtype)
                for d in range(data_unnorm.shape[1]):
                    if valid_cols[d]:
                        std[:, d] = np.std(data_unnorm[valid_mask[:, d], d], axis=0, keepdims=True)
        else:
            if np.count_nonzero(valid_mask) == 0:
                std = 1.0
            else:
                std = np.ones((1, data_unnorm.shape[1]), dtype=data_unnorm.dtype)
                for d in range(data_unnorm.shape[1]):
                    if valid_cols is None or valid_cols[d]:
                        std[:, d] = np.std(data_unnorm[valid_mask[:, d], d], axis=0, keepdims=True)
        std = np.maximum(std, 1e-8) # for robustnes
    if valid_mask is None:
        data_norm = np.divide((data_unnorm - mu), std)
    else:
        data_norm = np.divide((data_unnorm - mu), std)
        invalid_mask = np.logical_not(valid_mask)
        data_norm[invalid_mask] = data_unnorm[invalid_mask]
    return data_norm, mu, std

def norm2unnorm(data_norm, mu, std, valid_mask=None):
    if valid_mask is None:
        return mu + (std * data_norm)
    else:
        result = mu + (std * data_norm)
        invalid_mask = np.logical_not(valid_mask)
        result[invalid_mask] = data_norm[invalid_mask]
        return result

def unnorm2pix(pos_unnorm, low_unnorm, high_unnorm, high_pix):
    den_unnorn = (high_unnorm - low_unnorm)
    pos_pix = high_pix * (pos_unnorm - low_unnorm) / den_unnorn
    return pos_pix

def pix2unnorm(pos_pix, low_unnorm, high_unnorm, high_pix):
    pos_unnorm = low_unnorm + (high_unnorm - low_unnorm) * pos_pix / high_pix
    return pos_unnorm

def getNormRangeForBiom(lows_un, highs_un, mus_un, stds_un, side_test_grid, vis_biom_idx):
    low_norm = (lows_un[vis_biom_idx] - mus_un[vis_biom_idx]) / stds_un[vis_biom_idx]
    high_norm = (highs_un[vis_biom_idx] - mus_un[vis_biom_idx]) / stds_un[vis_biom_idx]
    stp_norm = (high_norm - low_norm) / side_test_grid
    return np.arange(low_norm, high_norm, stp_norm)

def saveFigureToFile(plt,names_BIOMs,names_CONDs,gender_str,suffix):
    file_name = 'plots/MHRP_bioms_%dD' % len(names_BIOMs)
    for name_BIOM in names_BIOMs:
        file_name += '_%s' %name_BIOM
    # file_name += '_conds_%d' % len(names_CONDs)
    # for name_COND in names_CONDs:
    #     file_name += '_%s' %name_COND
    file_name += '_%s_%s.png' %(gender_str,suffix)
    plt.savefig(file_name)

def get_optimizer(net, optimizer_str, learning_rate, momentum=None):

    optimizer_str = optimizer_str.lower()
    if optimizer_str == 'adam':               # set the optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer_str == 'adamw':
        optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    elif optimizer_str == 'adamax':
        optimizer = torch.optim.Adamax(net.parameters(), lr=learning_rate)
    elif optimizer_str == 'adagrad':
        optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    elif optimizer_str == 'sgd':
        if momentum is None:
            raise Exception('Need momentum')
        optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_str == 'rmsprop':
        if momentum is None:
            raise Exception('Need momentum')
        optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise NotImplementedError('NN_OPTIM: {} is not iplemented'.format(optimizer_str))
    return optimizer

def get_loss_func(loss_func_str='CrossEntropyLoss', weight=None):

    # TODO: try additional classification losses
    loss_func_str = loss_func_str.lower()
    if loss_func_str == 'crossentropyloss':
        loss_func = torch.nn.CrossEntropyLoss(weight=weight, reduction='none')
    elif loss_func_str == 'l1loss' or loss_func_str == 'maeloss':
        loss_func = torch.nn.L1Loss(reduction='none')
    elif loss_func_str == 'l2loss' or loss_func_str == 'mseloss':
        loss_func = torch.nn.MSELoss(reduction='none')
    elif loss_func_str == 'f1loss':
        loss_func = f1_loss
    else:
        raise NotImplementedError('NN_LOSS: {} is not implemented'.format(loss_func_str))
    return loss_func

def compute_rds(softmax_test, labels_gt_test_numpy, sample_weights=None):

    rds_array = compute_rds_general(softmax_test, labels_gt_test_numpy, sample_weights=sample_weights)
    idx = np.argmin(np.abs(rds_array[:, 1] - 0.10))
    risk_disc_score = rds_array[idx, 4]

    """
    risk_disc_score = -1
    # RDS
    # calculating population percentages corresponding to all softmax cutoffs
    num_test = softmax_test.shape[0]

    if sample_weights is None:
        sample_weights = np.ones(num_test)
    sample_weights = sample_weights / np.sum(sample_weights)

    num_sftmx_steps = 100                           # 100
    pop_perc_cutoffs = [10, 30, 70, 90]             # [10, 30, 70, 90]
    num_perc_cutoffs = len(pop_perc_cutoffs)
    popul_percentg = []
    sftmx_min, sftmx_max = np.min(softmax_test[:, 1]), np.max(softmax_test[:, 1])
    stmx_step = (sftmx_max-sftmx_min) / num_sftmx_steps
    try:
        sftmx_rng = np.arange(sftmx_min, sftmx_max, stmx_step)
    except Exception as detail:
        print("sftmx_min={}, sftmx_max={}, stmx_step={}".format(sftmx_min, sftmx_max, stmx_step))
        print(detail)
        return 0

    # CDF
    popul_percentg = np.zeros((len(sftmx_rng), 2))
    for i in range(len(sftmx_rng)):
        softmax_thresh = sftmx_rng[i]
        pts_idxs = softmax_test[:, 1] < softmax_thresh               # all people with softmax < current threshold
        perc = np.multiply(pts_idxs, sample_weights)
        perc = np.sum(perc)
        perc = perc * 100.0
        #num_people_in_region = np.count_nonzero(pts_idxs)
        #perc = 100 * num_people_in_region / num_test
        popul_percentg[i, :] = [softmax_thresh, perc]
    popul_percentg = np.array(popul_percentg)

    # calculating softmax cutoffs corresponding to selected population percentage cutoffs
    sftmx_cutoffs = np.zeros(num_perc_cutoffs)
    for i in range(num_perc_cutoffs):
        diff_vec = abs(popul_percentg[:, 1] - pop_perc_cutoffs[i])
        sftmx_cutoffs[i] = popul_percentg[np.argmin(diff_vec), 0]

    # padding by adding 0 and 100 at beginning and end of softmax and pop percentage arrays
    num_cutoffs = len(sftmx_cutoffs)
    pop_perc_cutoffs_padded, cutoff_sftmx_padded = np.zeros(num_cutoffs+2), np.zeros(num_cutoffs+2)
    for i in range(num_cutoffs):
        cutoff_sftmx_padded[1+i], pop_perc_cutoffs_padded[1+i] = sftmx_cutoffs[i], pop_perc_cutoffs[i]
    cutoff_sftmx_padded[num_cutoffs+1], pop_perc_cutoffs_padded[num_cutoffs+1] = 1, 100

    # computing risk profile on sparse data
    num_cutoffs_padded = len(range(len(cutoff_sftmx_padded)-1))
    risk_profile = np.zeros((num_cutoffs_padded,5))
    for i in range(num_cutoffs_padded):
        sftmx_cutoff_low, sftmx_cutoff_high = cutoff_sftmx_padded[i], cutoff_sftmx_padded[i + 1]
        pts_idxs = np.logical_and(sftmx_cutoff_low < softmax_test[:, 1], softmax_test[:, 1] < sftmx_cutoff_high)
        sample_weights_sub = sample_weights[pts_idxs].copy()
        sample_weights_sub = sample_weights_sub / np.sum(sample_weights_sub)
        #num_people_in_region = np.count_nonzero(pts_idxs)
        #if num_people_in_region==0:
        if np.sum(sample_weights_sub) == 0.0:
            print('\n**** WARNING 0 participants in region! Difficult to estimate good RDS -> EXITING ****')
            return risk_disc_score
        cond_prevalence_in_region = np.multiply(labels_gt_test_numpy[pts_idxs], sample_weights_sub)
        cond_prevalence_in_region = np.sum(cond_prevalence_in_region)
        cond_prevalence_in_region = cond_prevalence_in_region * 100.0
        #num_positives_in_region = np.sum(labels_gt_test_numpy[pts_idxs])
        pop_cutoff_low, pop_cutoff_high = pop_perc_cutoffs_padded[i], pop_perc_cutoffs_padded[i+1]
        #cond_prevalence_in_region = 100 * num_positives_in_region / num_people_in_region
        risk_profile[i,:] = [sftmx_cutoff_low, sftmx_cutoff_high, pop_cutoff_low, pop_cutoff_high, cond_prevalence_in_region]
    risk_disc_score = np.max(risk_profile[:, 4]) - np.min(risk_profile[:, 4])
    """

    return risk_disc_score


def compute_rds_general(softmax_test, labels_gt_test_numpy, sample_weights=None, fname_save=None):

    # calculating population percentages corresponding to all softmax cutoffs
    num_test = softmax_test.shape[0]
    prevalence_gt = np.mean(labels_gt_test_numpy.astype(np.float32))

    if sample_weights is None:
        sample_weights = np.ones(num_test)
    sample_weights = sample_weights / np.sum(sample_weights)

    sorted_indices = np.argsort(softmax_test[:, 1])
    labels_gt_test_numpy_sorted = labels_gt_test_numpy[sorted_indices].astype(np.float32)
    sample_weights_sorted = sample_weights[sorted_indices]
    total_sample_weights = np.sum(sample_weights)
    prevalence_los = np.zeros((num_test,), dtype=np.float32)
    prevalence_his = np.zeros((num_test,), dtype=np.float32)
    cutoff_los = np.zeros((num_test,), dtype=np.float32)
    cutoff_his = np.zeros((num_test,), dtype=np.float32)
    for i_lo in range(num_test):
        i_hi = num_test - i_lo - 1
        labels_lo = labels_gt_test_numpy_sorted[0:i_lo+1]
        labels_hi = labels_gt_test_numpy_sorted[i_hi:]
        weights_lo = sample_weights_sorted[0:i_lo+1]
        weights_hi = sample_weights_sorted[i_hi:]
        total_weights_lo = np.sum(weights_lo).clip(min=1e-8)
        total_weights_hi = np.sum(weights_hi).clip(min=1e-8)
        prevalence_lo = np.sum(np.multiply(labels_lo, weights_lo)) / total_weights_lo
        prevalence_hi = np.sum(np.multiply(labels_hi, weights_hi)) / total_weights_hi
        cutoff_lo = total_weights_lo / total_sample_weights
        cutoff_hi = total_weights_hi / total_sample_weights
        prevalence_los[i_lo] = prevalence_lo
        prevalence_his[i_lo] = prevalence_hi  # yes, i_lo is correct
        cutoff_los[i_lo] = cutoff_lo
        cutoff_his[i_lo] = cutoff_hi

    rds_array = np.zeros((num_test, 5), dtype=np.float32)  # cutoff percentile, prevalence_lo, prevalence_hi, rds
    for i in range(num_test):
        cutoff_hi = cutoff_his[i]
        frac_of_prevalence_gt = cutoff_hi / prevalence_gt
        prevalence_hi = prevalence_his[i]
        idx_lo = np.argmin(np.abs(cutoff_hi - cutoff_los))
        prevalence_lo = prevalence_los[idx_lo]
        rds_this = prevalence_hi - prevalence_lo
        rds_array[i, :] = np.array([frac_of_prevalence_gt, cutoff_hi, prevalence_lo, prevalence_hi, rds_this])

    if fname_save is not None:
        dir_save = os.path.dirname(fname_save)
        if not os.path.exists(dir_save):
            os.makedirs(dir_save)
        fields = ['frac_gt_prev', 'cutoff', 'prev_low', 'prev_high', 'rds']
        with open(fname_save, 'w') as csvfid:
            writer = csv.DictWriter(csvfid, fieldnames=fields)
            writer.writeheader()
            for i in range(num_test):
                row = dict()
                for fidx, field in enumerate(fields):
                    row[field] = rds_array[i, fidx]
                writer.writerow(row)

    return rds_array


def compute_roc_curve_and_auc(pred_probs, gt_conds, sample_weight=None):

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(gt_conds, pred_probs, sample_weight=sample_weight)
    auc = sklearn.metrics.roc_auc_score(gt_conds, pred_probs, sample_weight=sample_weight)

    return fpr, tpr, thresholds, auc


def randomly_omit_columns(valid_mask_torch, frac_omit_one=0.333333, frac_omit_mask=0.333333):

    num_rows = valid_mask_torch.shape[0]
    num_features = valid_mask_torch.shape[1]

    shuffled_rows = list(range(num_rows))
    random.shuffle(shuffled_rows)
    valid_mask_exemplars_torch = valid_mask_torch.detach().clone()
    valid_mask_exemplars_torch = valid_mask_exemplars_torch[shuffled_rows, :]

    # omit one
    rand_cols = np.random.randint(0, num_features, size=num_rows)
    valid_mask_omit_one = valid_mask_torch.detach().clone()
    valid_mask_omit_one[range(num_rows), rand_cols] = 0.0

    valid_mask_return_torch = valid_mask_torch.detach().clone()

    weight_omit_one = numpy_to_cuda(torch.zeros((num_rows, 1), dtype=valid_mask_torch.dtype))
    weight_omit_mask = numpy_to_cuda(torch.zeros((num_rows, 1), dtype=valid_mask_torch.dtype))
    rand_val = np.random.rand(num_rows)
    weight_omit_one[rand_val <= frac_omit_one] = 1.0
    weight_omit_mask[rand_val >= 1.0-frac_omit_mask] = 1.0
    weight_orig = 1.0 - weight_omit_one - weight_omit_mask

    weight_total = weight_orig + weight_omit_mask + weight_omit_one
    assert(torch.all(weight_total == 1.0))

    valid_mask_return_torch = torch.multiply(valid_mask_return_torch, weight_orig) \
                            + torch.multiply(valid_mask_exemplars_torch, weight_omit_mask) \
                            + torch.multiply(valid_mask_omit_one, weight_omit_one)

    return valid_mask_return_torch


def has_hypertension(data_raw_yslot):
    # Looking for **undiagnosed** hypertension
    #
    # High blood pressure (hypertension) (BPQ020): ever told you had high blood pressure
    #     BPXSY1: systolic (mm Hg), first reading
    #     BPXDI1: diastolic (mm Hg), first reading
    #     BPXSY2: second reading
    #     BPXDI2: second reading
    #     BPXSY3: third reading
    #     BPXDI3: third reading
    #     BPXSY4: fourth reading
    #     BPXDI4: fourth reading
    #     According to https://www.nhlbi.nih.gov/health/high-blood-pressure:
    #         Normal: systolic: < 120 mm Hg, diastolic: < 80 mm Hg
    #         Elevated: systolic: 120-129 mm Hg, diastolic: < 80 mm Hg
    #         High blood pressure: systolic: >= 130 mm Hg, diastolic: >= 80 mm Hg
    #     According to https://www.who.int/news-room/fact-sheets/detail/hypertension:
    #         High blood pressure: systolic >= 140 mmHg and/or diastolic >= 90 mm Hg on two different days
    #     According to https://my.clevelandclinic.org/health/diseases/4314-hypertension-high-blood-pressure:
    #         Normal: < 130/80 mmHg
    #         Mild hypertension (stage 1): 130-139/OR diastolic between 80-89mmHg
    #         Moderate hypertension (stage 2): 140/90 mmHg or higher
    #         Hypertensive crisis (get emergency care): 180/120 mmHg

    systolic_thresh = 130  # mm Hg
    diastolic_thresh = 80  # mm Hg

    systolic_fields = ['BPXSY1', 'BPXSY2', 'BPXSY3', 'BPXSY4']
    diastolic_fields = ['BPXDI1', 'BPXDI2', 'BPXDI3', 'BPXDI4']

    num_rows = data_raw_yslot.shape[0]
    has_it = np.zeros((num_rows,), dtype=bool)

    count = 0
    accum = 0
    for field in systolic_fields:
        if field in data_raw_yslot.dtype.names:
            valid = data_raw_yslot[field] > 1e-6
            valid = valid.astype(np.float32)
            count = count + valid
            accum = accum + np.multiply(valid, data_raw_yslot[field])
    count = np.clip(count, 1e-6, None)
    accum = np.divide(accum, count)
    high_this = accum >= systolic_thresh
    has_it = np.logical_or(has_it, high_this)

    count = 0
    accum = 0
    for field in diastolic_fields:
        if field in data_raw_yslot.dtype.names:
            valid = data_raw_yslot[field] > 1e-6
            valid = valid.astype(np.float32)
            count = count + valid
            accum = accum + np.multiply(valid, data_raw_yslot[field])
    count = np.clip(count, 1e-6, None)
    accum = np.divide(accum, count)
    high_this = accum >= diastolic_thresh
    has_it = np.logical_or(has_it, high_this)

    return has_it


def has_diabetes(data_raw_yslot):
    # Diabetes (DIQ010): fasting glucose >= 7.0 mmol/L (>=126 mg/dL) and HbA1c >= ≥6.5%
    #                    LBDGLUSI is mmol/L
    #                    LBXGLU is mg/dL
    #                    HbA1c is LBXGH (glycohemoglobin)

    fasting_glucose_mmol_per_L_thresh = 7.0  # mmol/L
    glycohemoglobin_thresh = 6.5
    oral_glucose_tolerance_threshold = 11.1  # mmol/L. from https://www.mayoclinic.org/diseases-conditions/diabetes/diagnosis-treatment/drc-20371451

    num_rows = data_raw_yslot.shape[0]
    has_it = np.zeros((num_rows,), dtype=bool)

    if 'LBDGLUSI' in data_raw_yslot.dtype.names:
        high_this = data_raw_yslot['LBDGLUSI'] >= fasting_glucose_mmol_per_L_thresh
        has_it = np.logical_or(has_it, high_this)
    elif 'LBXGLUSI' in data_raw_yslot.dtype.names:
        high_this = data_raw_yslot['LBXGLUSI'] >= fasting_glucose_mmol_per_L_thresh
        has_it = np.logical_or(has_it, high_this)

    if 'LBXGH' in data_raw_yslot.dtype.names:
        high_this = data_raw_yslot['LBXGH'] >= glycohemoglobin_thresh
        has_it = np.logical_or(has_it, high_this)

    if 'LBDGLTSI' in data_raw_yslot.dtype.names:  # two hour glucose (OGTT) in mmol/L
        high_this = data_raw_yslot['LBDGLTSI'] >= oral_glucose_tolerance_threshold
        has_it = np.logical_or(has_it, high_this)

    return has_it


def has_arthritis(data_raw_yslot):
    # arthritis or gout
    # MCQ160n: gout

    num_rows = data_raw_yslot.shape[0]
    has_it = np.zeros((num_rows,), dtype=bool)

    if 'MCQ160N' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160N'] == 1
        has_it = np.logical_or(has_this, has_it)
    elif 'MCQ160n' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160n'] == 1
        has_it = np.logical_or(has_this, has_it)

    return has_it

def has_coronary_heart_disease_or_related(data_raw_yslot):

    num_rows = data_raw_yslot.shape[0]
    has_it = np.zeros((num_rows,), dtype=bool)

    # ever told you had heart attack
    if 'MCQ160E' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160E'] == 1
        has_it = np.logical_or(has_this, has_it)
    elif 'MCQ160e' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160e'] == 1
        has_it = np.logical_or(has_this, has_it)

    # ever told you had angina/angina pectoris
    if 'MCQ160D' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160D'] == 1
        has_it = np.logical_or(has_this, has_it)
    elif 'MCQ160d' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160d'] == 1
        has_it = np.logical_or(has_this, has_it)

    # ever told had congestive heart failure (the relationship is opposite: CHF from CHD)
    if 'MCQ160B' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160B'] == 1
        has_it = np.logical_or(has_this, has_it)
    elif 'MCQ160b' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160b'] == 1
        has_it = np.logical_or(has_this, has_it)

    # ever told you had a stroke
    if 'MCQ160F' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160F'] == 1
        has_it = np.logical_or(has_this, has_it)
    elif 'MCQ160f' in data_raw_yslot.dtype.names:
        has_this = data_raw_yslot['MCQ160f'] == 1
        has_it = np.logical_or(has_this, has_it)

    return has_it


def save_config_object(cfg, fname_save):

    dir_save = os.path.dirname(fname_save)
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
    with open(fname_save, 'w') as fp:
        fp.write(cfg.dump())


def f1_loss(y_pred, y_true):

    if y_true.dtype != y_pred.dtype:
        y_true = y_true.to(y_pred.dtype)

    tp = torch.sum(y_true*y_pred[:, 1], dim=0)
    tn = torch.sum((1-y_true)*y_pred[:, 0], dim=0)
    fp = torch.sum((1-y_true)*y_pred[:, 1], dim=0)
    fn = torch.sum(y_true*y_pred[:, 0], dim=0)

    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)

    f1 = 2*p*r / (p+r+1e-6)
    return 1 - f1
