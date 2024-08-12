# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from scipy.ndimage import gaussian_filter
import pickle
from DiscRisk_utilities import *
from DiscRisk_models import *


def evaluate_nn_ensemble(NN_ensemble, dloader, apply_sample_weights=False, partition='test',
                         include_and_flag_missing_bioms=False,
                         fname_save_roc_curve=None,
                         fname_save_precision_recall_curve=None,
                         fname_save_raw=None,
                         fname_save_rds_array=None,
                         softmax_diff_thresh=None):

    data = dloader.get_data()

    if partition == 'test':
        is_test = data['is_test']
    elif partition == 'val':
        is_test = data['is_val']
    elif partition == 'train':
        is_test = data['is_train']
    else:
        raise NotImplementedError('partition: {} is not supported'.format(partition))

    num_test = np.count_nonzero(is_test)
    data_BIOMS_gt_norm_np = dloader.get_normalized_bioms(include_and_flag_missing_bioms=include_and_flag_missing_bioms)
    bioms_test_torch = numpy_to_cuda(data_BIOMS_gt_norm_np[is_test, :].astype(np.float32))

    # Inference
    NN_selected_net_in_ensemble = -1                # index of single NN to use, set to -1 for all ensemble
    softmax_test = testNNEnsemble(NN_ensemble, bioms_test_torch, NN_selected_net_in_ensemble)

    if softmax_diff_thresh is not None:
        softmax_test[:, 1] = softmax_test[:, 1] - softmax_diff_thresh

    labels_gt_test_numpy = data['CON_gt'][is_test]
    labels_individual_gt_test_numpy = data['CONDs_gt'][is_test]

    if apply_sample_weights:
        sample_weights = data['SAMPLE_WEIGHT'][is_test]
        sample_weights = sample_weights / np.sum(sample_weights)
    else:
        sample_weights = np.ones(num_test) / num_test

    if fname_save_raw is not None:
        data_bioms_save = dloader.get_data()
        data_bioms_save = data_bioms_save['BIOMs_gt'][is_test, :]
        fields = []
        for colidx in range(data_bioms_save.shape[1]):
            if colidx < len(dloader.names_bioms):
                fields.append(dloader.names_bioms[colidx])
            else:
                colprev = colidx - len(dloader.names_bioms)
                name = dloader.names_bioms[colprev]
                fields.append(name + '_flag')
        output_fields = []
        for cond in dloader.names_conds:
            output_fields.append(cond + '_0')
            output_fields.append(cond + '_1')
        fields.extend(output_fields)
        for cond in dloader.names_conds:
            fields.append(cond + '_GT')
        fields.append('sample_weight')
        dir_save_raw = os.path.dirname(fname_save_raw)
        if not os.path.exists(dir_save_raw):
            os.makedirs(dir_save_raw)
        with open(fname_save_raw, 'w') as fid_raw:
            writer = csv.DictWriter(fid_raw, fieldnames=fields)
            writer.writeheader()
            for rowidx in range(softmax_test.shape[0]):
                row_dict = dict()
                for colidx in range(data_bioms_save.shape[1]):
                    field = fields[colidx]
                    val = float(data_bioms_save[rowidx, colidx])
                    row_dict[field] = val
                for colidx in range(softmax_test.shape[1]):
                    field = output_fields[colidx]
                    val = float(softmax_test[rowidx, colidx])
                    row_dict[field] = val
                for colidx, cond in enumerate(dloader.names_conds):
                    if len(labels_individual_gt_test_numpy.shape) == 1:
                        val = labels_individual_gt_test_numpy[rowidx]
                    else:
                        val = labels_individual_gt_test_numpy[rowidx, colidx]
                    field = cond + '_GT'
                    row_dict[field] = val
                row_dict['sample_weight'] = sample_weights[rowidx]
                writer.writerow(row_dict)

    # Accuracy
    labels_pred_test = np.argmax(softmax_test, axis=1)  # torch.max(softmax_sparse_all, 1)[1]          # predicted labels - applying max to hard-assign class labels to input
    accuracy = (labels_pred_test == labels_gt_test_numpy)
    accuracy = np.multiply(accuracy, sample_weights)
    accuracy = np.sum(accuracy) * 100.0

    if fname_save_roc_curve:
        fpr, tpr, thresholds = metrics.roc_curve(labels_gt_test_numpy, softmax_test[:, 1])
        roc_auc = metrics.auc(fpr, tpr)
        display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        display.plot()
        dir_name = os.path.dirname(fname_save_roc_curve)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(fname_save_roc_curve)
        base_name = os.path.splitext(fname_save_roc_curve)[0]
        fname_save_roc_raw = base_name + '.pkl'
        data = dict()
        data['fpr'] = fpr
        data['tpr'] = tpr
        data['thresholds'] = thresholds
        data['roc_auc'] = roc_auc
        with open(fname_save_roc_raw, 'wb') as fid:
            pickle.dump(data, fid)

    if fname_save_precision_recall_curve:
        precision, recall, thresholds = metrics.precision_recall_curve(labels_gt_test_numpy, softmax_test[:, 1])
        avg_pr = metrics.average_precision_score(labels_gt_test_numpy, softmax_test[:, 1])
        display = metrics.PrecisionRecallDisplay(precision=precision, recall=recall, average_precision=avg_pr)
        display.plot()
        dir_name = os.path.dirname(fname_save_precision_recall_curve)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        plt.savefig(fname_save_precision_recall_curve)
        base_name = os.path.splitext(fname_save_precision_recall_curve)[0]
        fname_save_roc_raw = base_name + '.pkl'
        data = dict()
        data['precision'] = precision
        data['recall'] = recall
        data['thresholds'] = thresholds
        data['avg_pr'] = avg_pr
        with open(fname_save_roc_raw, 'wb') as fid:
            pickle.dump(data, fid)

    confusion_matrix = metrics.confusion_matrix(labels_gt_test_numpy, labels_pred_test, sample_weight=sample_weights)
    tp = confusion_matrix[1, 1]
    tn = confusion_matrix[0, 0]
    fp = confusion_matrix[0, 1]
    fn = confusion_matrix[1, 0]
    tpr = tp / (tp+fn)
    tnr = tn / (fp+tn)
    fpr = fp / (fp+tn)
    fnr = fn / (fn+tp)

    confusion_matrix_str = 'TN={}, FP={}; FN={}, TP={}'.format(tn, fp, fn, tp)

    precision, recall, f1score, support = metrics.precision_recall_fscore_support(labels_gt_test_numpy, labels_pred_test, beta=1.0, sample_weight=sample_weights)

    rds_array = compute_rds_general(softmax_test, labels_gt_test_numpy, sample_weights=sample_weights, fname_save=fname_save_rds_array)

    evaluation_dict = dict()
    evaluation_dict['accuracy'] = accuracy
    evaluation_dict['risk_disc_score'] = compute_rds(softmax_test, labels_gt_test_numpy, sample_weights=sample_weights)
    evaluation_dict['roc_area_under_curve'] = metrics.roc_auc_score(labels_gt_test_numpy, softmax_test[:, 1], sample_weight=sample_weights)
    evaluation_dict['pr_area_under_curve'] = metrics.average_precision_score(labels_gt_test_numpy, softmax_test[:, 1], sample_weight=sample_weights)
    evaluation_dict['confusion_matrix'] = confusion_matrix_str
    evaluation_dict['r2'] = metrics.r2_score(labels_gt_test_numpy, labels_pred_test, sample_weight=sample_weights)
    #evaluation_dict['labels_pred'] = labels_pred_test
    evaluation_dict['t_neg_frac'] = tn
    evaluation_dict['f_pos_frac'] = fp
    evaluation_dict['f_neg_frac'] = fn
    evaluation_dict['t_pos_frac'] = tp
    evaluation_dict['t_neg_rate'] = tnr
    evaluation_dict['f_pos_rate'] = fpr
    evaluation_dict['f_neg_rate'] = fnr
    evaluation_dict['t_pos_rate'] = tpr
    evaluation_dict['precision_neg'] = precision[0]
    evaluation_dict['precision_pos'] = precision[1]
    evaluation_dict['recall_neg'] = recall[0]
    evaluation_dict['recall_pos'] = recall[1]
    evaluation_dict['f1score_neg'] = f1score[0]
    evaluation_dict['f1score_pos'] = f1score[1]
    evaluation_dict['support_neg'] = support[0]
    evaluation_dict['support_pos'] = support[1]
    #evaluation_dict['brier_score'] = metrics.brier_score_loss(labels_gt_test_numpy, softmax_test[:, 1])

    f_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    perc_cutoffs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for f_val in f_vals:
        row_idx = np.argmin(np.abs(rds_array[:, 0] - f_val))
        key = 'rds_f_%02d' % int(f_val*100)
        evaluation_dict[key] = rds_array[row_idx, 4]
    for c_val in perc_cutoffs:
        row_idx = np.argmin(np.abs(rds_array[:, 1] - c_val))
        key = 'rds_p_%02d' % int(c_val*100)
        evaluation_dict[key] = rds_array[row_idx, 4]
    for f_val in f_vals:
        row_idx = np.argmin(np.abs(rds_array[:, 0] - f_val))
        key = 'prev_f_%02d' % int(f_val*100)
        evaluation_dict[key] = rds_array[row_idx, 3]
    for c_val in perc_cutoffs:
        row_idx = np.argmin(np.abs(rds_array[:, 1] - c_val))
        key = 'prev_p_%02d' % int(c_val*100)
        evaluation_dict[key] = rds_array[row_idx, 3]

    return evaluation_dict


if __name__ == '__main__':

    ############################################################################################################
    # Parameters -----------------------------------------------------------------------------------------------
    ############################################################################################################

    biom_model_idx = 0                             # see DiscRisk_utilities for biomarker model codes

    gender_id = 2                                   # 1=male, 2=female

    NN_max_ensemble_size = 10                       # setting the max size of the ensemble - large number for all
    NN_selected_net_in_ensemble = -1                # index of single NN to use, set to -1 for all ensemble
    NN_train_v_all_perc = 0.5                       # 0.5 - in [0,1]

    side_test_grid = 200                            # 500 higher (e.g. 500) for more resolution
    num_sftmx_steps = 100                           # 100
    pop_perc_cutoffs = [10, 30, 70, 90]             # [10, 30, 70, 90]
    num_perc_cutoffs = len(pop_perc_cutoffs)

    var_X, var_Y = 0, 1                             # indeces of biomarkers to be shown on the X and Y axis
    fixed_biom_vals = [0, 0, 0, 0, 0, 0, 0, 0, 0]   # values for all fixed biomarkers (in std units)

    PLOT_FIGURES = False            # False
    PLOT_3D_RISKMAP = False         # False
    SAVE_TEST_AND_RISK = True       # True

    REMOVE_PREGNANT = True          # True
    REMOVE_SMOKERS = False          # False

    names_BIOMs = getBiomarkerCodesFromBiomarkerModelId(biom_model_idx)
    names_CONDs = ['BPQ020', 'MCQ160A', 'DIQ010', 'MCQ160C', 'MCQ220']  # BPQ020 MCQ160A DIQ010 MCQ160C MCQ220 - chosen based on max num responses
                                                                        # hypert arthrit diabet coronhe cancer
    # NN architecture
    num_bioms, num_conditions = len(names_BIOMs), len(names_CONDs)
    # NN_architecture = [num_bioms, 4, 2]                     # SMALL ARCH - two-class output
    # NN_architecture = [num_bioms, 4, 4, 4, 2]               # LARGE ARCH - two-class output
    NN_architecture = [num_bioms, 10, 10, 10, 10, 2]            # two-class output

    fig_w, fig_h = 18, 4                            # 18 4
    plt.rcParams['toolbar'] = 'None'

    ############################################################################################################
    # MAIN -----------------------------------------------------------------------------------------------
    ############################################################################################################

    #region preliminaries
    gender_str = genderStringFromId(gender_id)
    # colormaps
    class_colmap = ownColormap_3colors(col_class_neg, col_class_unc, col_class_pos)
    risk_colmap = ownColormap_3colors(col_risk_neg, col_risk_unc, col_risk_pos)
    # name of best CNN model to be saved to disk
    folder_name = '../../models/%s/' % gender_str
    folder_name += 'modid%d_bioms' %biom_model_idx
    for biom_id in range(num_bioms):
        folder_name += '_%s' % names_BIOMs[biom_id]
    folder_name += '_conds'
    for cond_id in range(num_conditions):
        folder_name += '_%s' % names_CONDs[cond_id]
    folder_name += '_feats'
    for num_feats in NN_architecture:
        folder_name += '_%d' %num_feats
    if not os.path.exists(folder_name):
        print('\n*** ERROR folder %s does not exist' %folder_name)
        exit()
    #endregion preliminaries

    # region load NN ensemble
    NN_ensemble_size, NN_ensemble = 0, []
    for net_idx in range(NN_max_ensemble_size):
        file_name_NN_model = '%s/net_%02d.pkl' %(folder_name,net_idx)
        if not os.path.exists(file_name_NN_model):
            break
        NN_ensemble.append(torch.load(file_name_NN_model))  # load network and append it to list
        NN_ensemble_size+=1
    print('\n%d nets in ensemble' % NN_ensemble_size)
    # endregion

    #region load all labeled data
    num_year_slots = len(year_slot_names)               # number of NHANES year slots
    data_BIOMs_gt = np.zeros((0, num_bioms))       # all gt-labelled biomarker data points
    data_CONDs_gt = np.zeros((0, num_conditions))       # all conditions ground truth
    data_GEN, data_AGE = np.zeros(0), np.zeros(0)       # gender, age
    data_PRG, data_SMQ = np.zeros(0), np.zeros(0)       # pregnant, smokers
    for year_slot_id_tmp in range(num_year_slots):

        # load data for year slot from pickle file
        year_slot_name = year_slot_names[year_slot_id_tmp]
        file_name = r'%s/NHANES_merged_%s.pickle' %(dir_merged_datasets, year_slot_name)
        data_raw_yslot = loadRawDataFromPickleFile(file_name)   # mload raw data from file

        # check if fields exist
        if not checkFieldsExistInData(data_raw_yslot, names_BIOMs, names_CONDs, name_SMQ, REMOVE_SMOKERS, year_slot_name):
            continue

        # append new data for current year slot into global arrays
        data_BIOMs_gt = appendToData(data_raw_yslot, data_BIOMs_gt, names_BIOMs)     # Biomarkers
        data_CONDs_gt = appendToData(data_raw_yslot, data_CONDs_gt, names_CONDs)     # Conditions
        data_GEN = np.concatenate((data_GEN,data_raw_yslot[name_GEN]),axis=0)        # Gender
        data_AGE = np.concatenate((data_AGE,data_raw_yslot[name_AGE]),axis=0)        # Age
        data_PRG = np.concatenate((data_PRG,data_raw_yslot[name_PRG]),axis=0)        # Pregnant
        if REMOVE_SMOKERS:
            data_SMQ = np.concatenate((data_SMQ, data_raw_yslot[name_SMQ]), axis=0)  # Smoking
    #endregion load all labeled training data

    #region filter data
    # filter data by age
    mask_age = np.logical_and(data_AGE >= age_min, data_AGE <= age_max)
    data_BIOMs_gt = data_BIOMs_gt[mask_age, :]
    data_CONDs_gt = data_CONDs_gt[mask_age, :]
    data_GEN = data_GEN[mask_age]
    data_PRG = data_PRG[mask_age]
    if REMOVE_SMOKERS:
        data_SMQ = data_SMQ[mask_age]
    print('\n%d \t after age filtering' % len(data_GEN))

    # filter data by gender
    mask_gen = data_GEN == gender_id   # 1=male, 2=female
    data_BIOMs_gt = data_BIOMs_gt[mask_gen, :]
    data_CONDs_gt = data_CONDs_gt[mask_gen, :]
    data_PRG = data_PRG[mask_gen]
    if REMOVE_SMOKERS:
        data_SMQ = data_SMQ[mask_gen]
    print('%d \t %s' % (data_BIOMs_gt.shape[0], gender_str))

    # remove pregnant women
    if REMOVE_PREGNANT and gender_id==2:
        mask_not_pregnant = np.logical_not(data_PRG == 1) # 1 denotes pregnant, !=1 non pregnant
        if np.sum(mask_not_pregnant)>0:
            data_BIOMs_gt = data_BIOMs_gt[mask_not_pregnant, :]
            data_CONDs_gt = data_CONDs_gt[mask_not_pregnant, :]
            if REMOVE_SMOKERS:
                data_SMQ = data_SMQ[mask_not_pregnant]
        print('%d \t %s not pregnant' % (data_BIOMs_gt.shape[0], gender_str))

    # remove smokers
    if REMOVE_SMOKERS:
        mask_non_smokers = data_SMQ==2                  # smoked less than 100 cigarettes in lifetime
        data_BIOMs_gt = data_BIOMs_gt[mask_non_smokers, :]
        data_CONDs_gt = data_CONDs_gt[mask_non_smokers, :]
        print('%d \t %s not smoking' % (data_BIOMs_gt.shape[0], gender_str))

    # remove invalid conditions
    masks_valid_CONDs_all = data_CONDs_gt[:, 0] != -1
    for cond_id in range(1,num_conditions):
        mask_valid_id = data_CONDs_gt[:, cond_id] != -1
        masks_valid_CONDs_all = np.logical_and(masks_valid_CONDs_all, mask_valid_id)
    data_BIOMs_gt = data_BIOMs_gt[masks_valid_CONDs_all, :]
    data_CONDs_gt = data_CONDs_gt[masks_valid_CONDs_all, :]
    print('%d \t %s valid condition %s' % (data_BIOMs_gt.shape[0], gender_str, names_CONDs))

    # remove invalid biomarkers
    masks_valid_BIOMs_all = data_BIOMs_gt[:, 0] != -1
    for biom_id in range(1, num_bioms):
        mask_valid_id = data_BIOMs_gt[:, biom_id] != -1
        masks_valid_BIOMs_all = np.logical_and(masks_valid_BIOMs_all, mask_valid_id)
    data_BIOMs_gt = data_BIOMs_gt[masks_valid_BIOMs_all, :]
    data_CONDs_gt = data_CONDs_gt[masks_valid_BIOMs_all, :]
    print('%d \t %s valid biomarkers\n' % (data_BIOMs_gt.shape[0], gender_str))


    # map ground truth condition labels to integers
    idx_cond_positive_as_1 = np.logical_and(data_CONDs_gt >= min_val_cond_for_pos, data_CONDs_gt <= max_val_cond_for_pos)

    # merge conditions together through an OR operation
    data_CON_gt = np.any(idx_cond_positive_as_1, axis=1).astype('int') # False->0 (negative to condition) True->1 (positive to condition)

    num_gt_data_all = data_BIOMs_gt.shape[0]
    print('All ground-truth labelled data N %d' % num_gt_data_all)
    num_GT_pos = np.sum(data_CON_gt)  # number of positives
    num_GT_neg = num_gt_data_all - num_GT_pos
    print('  N_pos %d N_neg %d' % (num_GT_pos, num_GT_neg))
    global_prevalence = 100 * num_GT_pos / num_gt_data_all
    print('  global prevalence %1.1f%%' % global_prevalence)
    print('')

    #endregion filter data

    #region split train/test data, normalize and prepare data for Torch

    # getting low and high limits for visualization only
    lows_unnorm, highs_unnorm = np.zeros(num_bioms), np.zeros(num_bioms)
    for bio_id in range(num_bioms):
        vec = data_BIOMs_gt[:, bio_id]
        lows_unnorm[bio_id], highs_unnorm[bio_id] = np.min(vec), np.max(vec)

    # normalize input features
    data_BIOMS_gt_norm_np = data_BIOMs_gt.copy()
    mus_unnorm, stds_unnorm = np.zeros(num_bioms), np.zeros(num_bioms)
    for bio_id in range(num_bioms):
        data_BIOMS_gt_norm_np[:, bio_id], mu, std = unnorm2norm(data_BIOMs_gt[:, bio_id])
        mus_unnorm[bio_id], stds_unnorm[bio_id] = mu, std

    # all sparse labelled data (union train and test)
    indxs_sparse_all = list(range(0, num_gt_data_all, 1))                      # indeces of all labelled data
    bioms_sparse_all_norm_np = data_BIOMS_gt_norm_np[indxs_sparse_all]
    labels_gt_sparse_all_np = data_CON_gt[indxs_sparse_all]
    num_sparse_all_data = len(bioms_sparse_all_norm_np)
    bioms_sparse_all_norm_to = torch.from_numpy(bioms_sparse_all_norm_np).type(torch.FloatTensor)  # input features to Torch

    # training data sparse
    train_sampling_step = 1.0 / NN_train_v_all_perc
    indxs_sparse_train = list(np.floor(np.arange(0, num_gt_data_all, train_sampling_step)).astype('int'))
    bioms_sparse_train_norm_np = data_BIOMS_gt_norm_np[indxs_sparse_train]
    labels_gt_sparse_train_np = data_CON_gt[indxs_sparse_train]
    num_sparse_train_data = len(bioms_sparse_train_norm_np)
    bioms_sparse_train_norm_to = torch.from_numpy(bioms_sparse_train_norm_np).type(torch.FloatTensor)  # input features to Torch
    print('Training sparse data N %d' % num_sparse_train_data)

    # testing sparse data
    indxs_sparse_test = list(set(indxs_sparse_all) - set(indxs_sparse_train))          # perform set subtraction to get test indeces
    bioms_sparse_test_norm_np = data_BIOMS_gt_norm_np[indxs_sparse_test]
    labels_gt_sparse_test_np = data_CON_gt[indxs_sparse_test]
    num_sparse_test_data = len(bioms_sparse_test_norm_np)
    bioms_sparse_test_norm_to = torch.from_numpy(bioms_sparse_test_norm_np).type(torch.FloatTensor)   # input features to Torch
    print('Test sparse data N %d' % num_sparse_test_data)

    #endregion prepare data for Torch

    #region test ensemble on sparse data
    softmax_sparse_test = testNNEnsemble(NN_ensemble, bioms_sparse_test_norm_to, NN_selected_net_in_ensemble)
    labels_pred_sparse_test_to = (softmax_sparse_test>0.5).astype('int')
    num_correct_sparse_test = np.count_nonzero(labels_pred_sparse_test_to == labels_gt_sparse_test_np)      # counting number of correct predictions
    test_sparse_accuracy = (100.0 * num_correct_sparse_test) / num_sparse_test_data                      # measuring training accuracy
    labels_pred_sparse_test_cols = setPointColors(softmax_sparse_test, col_class_pos, col_class_neg, col_class_unc)      # colours of predicted labels, for visualization only
    #endregion

    #region test NN model on a regular dense grid
    print('\nTesting on regular dense grid')

    # for biom_id in range(num_bioms):
    #     vf = fixed_biom_vals[biom_id]
    #     if vf==-100:
    #         print('\t%s is variable' %names_BIOMs[biom_id])
    #     else:
    #         vf_unnorm = norm2unnorm(vf, mus_unnorm[biom_id], stds_unnorm[biom_id])
    #         print('\t%s is fixed %f' %(names_BIOMs[biom_id],vf_unnorm))

    # build regular testing grid FIXME FROM HERE
    rng_norm_x = getNormRangeForBiom(lows_unnorm, highs_unnorm, mus_unnorm, stds_unnorm, side_test_grid, var_X)
    rng_norm_y = rng_norm_x                                # if 1 biomarker only then copy range
    if num_bioms>1:
        rng_norm_y = getNormRangeForBiom(lows_unnorm, highs_unnorm, mus_unnorm, stds_unnorm, side_test_grid, var_Y)
    grid_width, grid_height = len(rng_norm_x), len(rng_norm_y)
    num_dense_pts = grid_width*grid_height

    bioms_vec2D_np = np.zeros((num_dense_pts,2)) # fixme do we need this?
    pos = 0
    for x1 in rng_norm_x:
        for x2 in rng_norm_y:
            bioms_vec2D_np[pos,:] = np.array([x1, x2])
            pos += 1

    # extract needed dimensions for NN testing
    bioms_vecND_norm_np = np.zeros((num_dense_pts, num_bioms))
    for biom_id in range(num_bioms):    # setting fixed values
        bioms_vecND_norm_np[:,biom_id] = fixed_biom_vals[biom_id]

    bioms_vecND_norm_np[:, var_X] = bioms_vec2D_np[:, 0]             # setting variable biomarker 0
    if num_bioms>1:
        bioms_vecND_norm_np[:, var_Y] = bioms_vec2D_np[:, 1]         # setting variable biomarker 1

    # transforming to Torch and applying NN to dense grid
    bioms_gridnD_norm_to = torch.from_numpy(bioms_vecND_norm_np).type(torch.FloatTensor)  # input features (biomarker measurements)
    softmax_list_dense = testNNEnsemble(NN_ensemble, bioms_gridnD_norm_to, NN_selected_net_in_ensemble)
    labels_pred_dense_cols = setPointColors(softmax_list_dense, col_class_pos, col_class_neg, col_class_unc)      # colours of predicted labels, for visualization only

    #endregion

    #region compute risk profile on sparse data
    print('Building risk map')
    # applying network to estimate class labels for ALL sparse points
    softmax_sparse_all = testNNEnsemble(NN_ensemble, bioms_sparse_all_norm_to, NN_selected_net_in_ensemble)

    # calculating population percentages corresponding to all softmax cutoffs
    popul_percentg = []
    sftmx_min, sftmx_max = np.min(softmax_sparse_all), np.max(softmax_sparse_all)
    stmx_step = (sftmx_max-sftmx_min) / num_sftmx_steps
    sftmx_rng = np.arange(sftmx_min, sftmx_max, stmx_step)
    popul_percentg = np.zeros((len(sftmx_rng),2))
    for i in range(len(sftmx_rng)):
        softmax_thresh = sftmx_rng[i]
        pts_idxs = softmax_sparse_all < softmax_thresh               # all people with softmax < current threshold
        num_people_in_region = np.count_nonzero(pts_idxs)
        perc = 100 * num_people_in_region / num_sparse_all_data
        popul_percentg[i,:] = [softmax_thresh, perc]
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
        pts_idxs = np.logical_and(sftmx_cutoff_low < softmax_sparse_all, softmax_sparse_all < sftmx_cutoff_high)
        num_positives_in_region = np.sum(labels_gt_sparse_all_np[pts_idxs])
        num_people_in_region = np.count_nonzero(pts_idxs)
        pop_cutoff_low, pop_cutoff_high = pop_perc_cutoffs_padded[i], pop_perc_cutoffs_padded[i+1]
        if num_people_in_region==0:
            print('\n**** WARNING 0 participants in region! Difficult to estimate good RDS -> EXITING ****')
            exit()
        cond_prevalence_in_region = 100 * num_positives_in_region / num_people_in_region
        risk_profile[i,:] = [sftmx_cutoff_low, sftmx_cutoff_high, pop_cutoff_low, pop_cutoff_high, cond_prevalence_in_region]
    risk_disc_score = np.max(risk_profile[:, 4]) - np.min(risk_profile[:, 4])

    #endregion

    #region render risk map

    # create softmax grid and risk map
    softmax_map2D = np.zeros((grid_height, grid_width)) # test grid image
    pos = 0
    for x in range(grid_width):
        for y in range(grid_height):
            softmax_map2D[y, x] = softmax_list_dense[pos]
            pos += 1
    softmax_map2D = np.flip(softmax_map2D, 0)

    risk_map2D = np.zeros((grid_height, grid_width))
    for x in range(grid_width):
        for y in range(grid_height):
            sftm_val = softmax_map2D[y, x]
            for i in range(risk_profile.shape[0]):
                [sftmx_cutoff_low, sftmx_cutoff_high, _, _, risk] = risk_profile[i, :]
                if sftmx_cutoff_low <= sftm_val < sftmx_cutoff_high:
                    risk_map2D[y, x] = risk / 100
                    break
    risk_map2D = gaussian_filter(risk_map2D, sigma=1)   # smoothing hard transitions
    #endregion compute risk map

    #region report results to console
    print('\nEnsemble size %d' % NN_ensemble_size)
    print('Conditions ' , end='')
    for cond_id in range(num_conditions):
        print('%s ' % names_CONDs[cond_id], end='')
    print('\n\nGender %s' % gender_str)
    print('Biomarker model %d '%biom_model_idx)
    print('Biomarkers ' , end='')
    for biom_id in range(num_bioms):
        print('%s ' % short_biom_descriptions[names_BIOMs[biom_id]], end='')
    print('\nRDS \t TEA \t n_tr \t n_te')
    print('%1.1f \t %1.1f \t %d \t %d' % (risk_disc_score,test_sparse_accuracy, num_sparse_train_data, num_sparse_test_data))
    #endregion

    if not PLOT_FIGURES:
        exit()

    #region Figure 1 test data points
    labels_sparse_test_cols = setPointColors(labels_gt_sparse_test_np, col_class_pos, col_class_neg, col_class_unc)     # setting colours for ground truth labels - for visualization only

    vis_data_sparse_test_pts_x = data_BIOMs_gt[indxs_sparse_test, var_X]
    vis_data_sparse_test_pts_y = (np.random.rand(num_sparse_test_data) - 0.5)   # initializing Ys as random for case when num_biomarkers=1
    if num_bioms>1:
        vis_data_sparse_test_pts_y = data_BIOMs_gt[indxs_sparse_test, var_Y]

    fig1, axs1 = plt.subplots(1, 2, figsize=(fig_w/2, fig_h))
    axs1[0].scatter(vis_data_sparse_test_pts_x, vis_data_sparse_test_pts_y, c=labels_sparse_test_cols, s=2, linewidth=0)
    axs1[0].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[var_X]])
    axs1[0].set_xlim([lows_unnorm[var_X], highs_unnorm[var_X]])
    axs1[0].set_title('Testing (%s, age %d-%d, n_conds %d)' % (gender_str, age_min, age_max, num_conditions))
    if num_bioms == 1:
        axs1[0].set_yticks([])
        axs1[0].set_ylim([np.min(vis_data_sparse_test_pts_y), np.max(vis_data_sparse_test_pts_y)])
    else:
        axs1[0].set_ylabel('%s' % long_biom_descriptions[names_BIOMs[var_Y]])
        axs1[0].set_ylim([lows_unnorm[var_Y], highs_unnorm[var_Y]])

    # get ticks positions and labels for later use
    xticks_unnorm_pos, xticks_unnorm_lab = axs1[0].get_xticks(), axs1[0].get_xticklabels()
    yticks_unnorm_pos, yticks_unnorm_lab = axs1[0].get_yticks(), axs1[0].get_yticklabels()

    # figure 1 plot mid & right
    axs1[1].clear()
    axs1[1].scatter(vis_data_sparse_test_pts_x, vis_data_sparse_test_pts_y, c=labels_pred_sparse_test_cols, s=2, linewidth=0)
    axs1[1].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[var_X]])
    axs1[1].set_xlim([lows_unnorm[var_X], highs_unnorm[var_X]])
    axs1[1].set_title('Testing on test data (N %d/%d %1.1f%%)' % (num_sparse_test_data, num_gt_data_all, 100 * (1 - NN_train_v_all_perc)))
    if num_bioms == 1:
        axs1[1].set_yticks([])
        axs1[1].set_ylim([np.min(vis_data_sparse_test_pts_y), np.max(vis_data_sparse_test_pts_y)])
    else:
        axs1[1].set_ylabel('%s' % long_biom_descriptions[names_BIOMs[var_Y]])
        axs1[1].set_ylim([lows_unnorm[var_Y], highs_unnorm[var_Y]])

    plt.pause(0.1)

    #endregion figure 1

    #region Figure 2 softmax map and risk map
    fig2, axs2 = plt.subplots(1, 4, figsize=(fig_w, fig_h))

    # preparing visualization data points grid

    # prepare x/y ticks
    xticks_pos, xticks_lab = [], []
    for i in range(len(xticks_unnorm_pos)):
        pos_unnorm = xticks_unnorm_pos[i]
        px = unnorm2pix(pos_unnorm, lows_unnorm[var_X], highs_unnorm[var_X], grid_width)
        xticks_pos.append(px)
        xticks_lab.append(xticks_unnorm_lab[i].get_text())
    if num_bioms>1:
        yticks_pos, yticks_lab = [], []
        for i in range(len(yticks_unnorm_pos)):
            pos_unnorm = yticks_unnorm_pos[i]
            py = grid_height - 1 - unnorm2pix(pos_unnorm, lows_unnorm[var_Y], highs_unnorm[var_Y], grid_height)
            yticks_pos.append(py)
            yticks_lab.append(yticks_unnorm_lab[i].get_text())

    # plot softmax grid
    axs2[0].imshow(softmax_map2D, vmin=0, vmax=1, cmap=class_colmap)
    axs2[0].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[var_X]])
    axs2[0].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[0].set_yticks([])
    else:
        axs2[0].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        axs2[0].set_ylabel('%s' % long_biom_descriptions[names_BIOMs[var_Y]])
    axs2[0].set_xlim([0, grid_width])
    axs2[0].set_ylim([grid_height, 0])
    axs2[0].set_title('Testing on regular biomarker grid')

    #region overlaying points onto soft-max map
    axs2[1].imshow(softmax_map2D, vmin=0, vmax=1, cmap=class_colmap)
    vis_data_all_pts_x = data_BIOMs_gt[indxs_sparse_all, var_X]
    vis_data_all_pts_y = (np.random.rand(num_sparse_all_data) - 0.5)  # initializing Ys as random for case when num_biomarkers=1
    if num_bioms > 1:
        vis_data_all_pts_y = data_BIOMs_gt[indxs_sparse_all, var_Y]

    labels_gt_sparse_all_np = data_CON_gt[indxs_sparse_all]
    labels_all_cols = setPointColors(labels_gt_sparse_all_np, col_class_pos, col_class_neg, col_class_unc)     # setting colours for ground truth labels - for visualization only

    pts_x = unnorm2pix(vis_data_all_pts_x, lows_unnorm[var_X], highs_unnorm[var_X], grid_width)
    if num_bioms>1:
        pts_y = grid_height - 1 - unnorm2pix(vis_data_all_pts_y, lows_unnorm[var_Y], highs_unnorm[var_Y], grid_height)
    else:
        pts_y = grid_height - 1 - unnorm2pix(vis_data_all_pts_y, np.min(vis_data_all_pts_y), np.max(vis_data_all_pts_y), grid_height)

    axs2[1].scatter(pts_x, pts_y, c=labels_all_cols, alpha=0.2, edgecolor=[0.1,0.1,0.1,0.9], s=10, linewidth=1)

    axs2[1].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[var_X]])
    axs2[1].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[1].set_yticks([])
    else:
        axs2[1].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        # axs2[1].set_ylabel('%s' % fun_descriptions[names_BIOMs[var_Y]])
    axs2[1].set_xlim([0, grid_width])
    axs2[1].set_ylim([grid_height, 0])
    axs2[1].set_title('Superimposing labelled points')
    #endregion

    # plot risk map grid
    axs2[2].imshow(risk_map2D, vmin=0, vmax=1, cmap=risk_colmap)
    axs2[2].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[var_X]])
    axs2[2].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[2].set_yticks([])
    elif num_bioms>1:
        axs2[2].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        # axs2[2].set_ylabel('%s' % fun_descriptions[names_BIOMs[var_Y]])
    axs2[2].set_xlim([0, grid_width])
    axs2[2].set_ylim([grid_height, 0])
    axs2[2].set_title('Health risk map',fontsize=12)

    Delta_from_botm, txt_size = 2.5, 8
    L = len(cutoff_sftmx_padded)-1
    for i in range(L):
        [_, _, p_low, p_high, risk] = risk_profile[i,:]
        col = risk_colmap(risk / 100.0)
        axs2[3].plot([0.1*L,0.35*L],[Delta_from_botm + 0.5*i,Delta_from_botm + 0.5*i],'-',color=col,linewidth=3*L) # coloured rectabgle
        axs2[3].text(0.15*L, Delta_from_botm + 0.5*i -0.1, '%1.1f%%          (%d%%)' % (risk, p_high - p_low))            # item text

    axs2[3].text(0, 0.6*Delta_from_botm,'%s, age %d-%d, train size %d / %d' % (gender_str, age_min, age_max, num_sparse_train_data, num_gt_data_all), fontsize=txt_size)
    txt = 'biomarkers'
    for i in range(len(names_BIOMs)):
        txt = txt + ' %s' % short_biom_descriptions[names_BIOMs[i]]
    axs2[3].text(0,0.5*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'var fixed  '
    for biom_id in range(num_bioms):
        if biom_id==var_X or biom_id==var_Y:
            txt += ' V '
        else:
            txt += ' %1.1f' %norm2unnorm(fixed_biom_vals[biom_id], mus_unnorm[biom_id], stds_unnorm[biom_id])
    axs2[3].text(0,0.4*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'conditions'
    for i in range(len(names_CONDs)):
        txt = txt + ' %s' %names_CONDs[i]
    axs2[3].text(0,0.3*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'architecture '
    for i in range(len(NN_architecture)):
        txt = txt + ' %d' %NN_architecture[i]
    axs2[3].text(0,0.2*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    axs2[3].text(0, 0.1 * Delta_from_botm,'ensemble size %d, net idx %d' % (NN_ensemble_size, NN_selected_net_in_ensemble), fontsize=txt_size)
    axs2[3].text(0, 0.0 * Delta_from_botm,'risk disc score %1.1f%% (test acc %1.1f%%)' % (risk_disc_score, test_sparse_accuracy), fontsize=txt_size)
    axs2[3].set_xlim([0,L])
    axs2[3].set_ylim([-0.5,L+0.5])
    axs2[3].axis('off')
    axs2[3].set_title('Legend: health risk (population %)',fontsize=10,loc='left')

    plt.pause(0.1)

    if SAVE_TEST_AND_RISK:
        saveFigureToFile(plt,names_BIOMs,names_CONDs,gender_str,'2riskmaps_sizeEnsemb%d' %NN_ensemble_size)
    #endregion

    # print('Finished Test and Risk Maps')
    plt.show()

    #region Figure 2 softmax map and risk map
    fig2, axs2 = plt.subplots(1, 4, figsize=(fig_w, fig_h))

    # preparing visualization data points grid

    # prepare x/y ticks
    xticks_pos, xticks_lab = [], []
    for i in range(len(xticks_unnorm_pos)):
        pos_unnorm = xticks_unnorm_pos[i]
        px = unnorm2pix(pos_unnorm, lows_unnorm[0], highs_unnorm[0], grid_width)
        xticks_pos.append(px)
        xticks_lab.append(xticks_unnorm_lab[i].get_text())
    if num_bioms>1:
        yticks_pos, yticks_lab = [], []
        for i in range(len(yticks_unnorm_pos)):
            pos_unnorm = yticks_unnorm_pos[i]
            py = grid_height - 1 - unnorm2pix(pos_unnorm, lows_unnorm[1], highs_unnorm[1], grid_height)
            yticks_pos.append(py)
            yticks_lab.append(yticks_unnorm_lab[i].get_text())

    # plot softmax grid
    axs2[0].imshow(softmax_map2D, vmin=0, vmax=1, cmap=class_colmap)
    axs2[0].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[0]])
    axs2[0].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[0].set_yticks([])
    else:
        axs2[0].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        axs2[0].set_ylabel('%s' % long_biom_descriptions[names_BIOMs[1]])
    axs2[0].set_xlim([0, grid_width])
    axs2[0].set_ylim([grid_height, 0])
    axs2[0].set_title('Testing on regular biomarker grid')

    #region overlaying points onto soft-max map
    unc_map_2D = np.abs(softmax_map2D - 0.5)
    unc_map_2D /= np.max(unc_map_2D)
    axs2[1].imshow(unc_map_2D, vmin=0, vmax=1, cmap=class_colmap)
    axs2[1].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[0]])
    axs2[1].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[1].set_yticks([])
    else:
        axs2[1].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        axs2[1].set_ylabel('%s' % long_biom_descriptions[names_BIOMs[1]])
    axs2[1].set_xlim([0, grid_width])
    axs2[1].set_ylim([grid_height, 0])
    axs2[1].set_title('Uncertainty map')
    #endregion

    # plot risk map grid
    axs2[2].imshow(risk_map2D, vmin=0, vmax=1, cmap=risk_colmap)
    axs2[2].set_xlabel('%s' % long_biom_descriptions[names_BIOMs[0]])
    axs2[2].set_xticks(ticks=xticks_pos, labels=xticks_lab)
    if num_bioms == 1:
        axs2[2].set_yticks([])
    elif num_bioms>1:
        axs2[2].set_yticks(ticks=yticks_pos, labels=yticks_lab)
        # axs2[2].set_ylabel('%s' % fun_descriptions[names_BIOMs[1]])
    axs2[2].set_xlim([0, grid_width])
    axs2[2].set_ylim([grid_height, 0])
    axs2[2].set_title('Health risk map',fontsize=12)

    Delta_from_botm, txt_size = 2.5, 8
    L = len(cutoff_sftmx_padded)-1
    for i in range(L):
        [_, _, p_low, p_high, risk] = risk_profile[i,:]
        col = risk_colmap(risk / 100.0)
        axs2[3].plot([0.1*L,0.35*L],[Delta_from_botm + 0.5*i,Delta_from_botm + 0.5*i],'-',color=col,linewidth=3*L) # coloured rectabgle
        axs2[3].text(0.15*L, Delta_from_botm + 0.5*i -0.1, '%1.1f%%          (%d%%)' % (risk, p_high - p_low))            # item text

    axs2[3].text(0, 0.6*Delta_from_botm,'%s, age %d-%d, train size %d / %d' % (gender_str, age_min, age_max, num_sparse_train_data, num_gt_data_all), fontsize=txt_size)
    txt = 'biomarkers'
    for i in range(len(names_BIOMs)):
        txt = txt + ' %s' % short_biom_descriptions[names_BIOMs[i]]
    axs2[3].text(0,0.5*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'var fixed  '
    for biom_id in range(num_bioms):
        vf = fixed_biom_vals[biom_id]
        if vf!=-100:
            vf = norm2unnorm(vf, mus_unnorm[biom_id], stds_unnorm[biom_id])
            txt = txt + ' %1.1f' %vf
        else:
            txt = txt + ' v'
    axs2[3].text(0,0.4*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'conditions'
    for i in range(len(names_CONDs)):
        txt = txt + ' %s' %names_CONDs[i]
    axs2[3].text(0,0.3*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    txt = 'architecture '
    for i in range(len(NN_architecture)):
        txt = txt + ' %d' %NN_architecture[i]
    axs2[3].text(0,0.2*Delta_from_botm,'%s' %txt,fontsize=txt_size)
    axs2[3].text(0, 0.1 * Delta_from_botm,'ensemble size %d, net idx %d' % (NN_ensemble_size, NN_selected_net_in_ensemble), fontsize=txt_size)
    axs2[3].text(0, 0.0 * Delta_from_botm,'risk disc score %1.1f%% (test acc %1.1f%%)' % (risk_disc_score, test_sparse_accuracy), fontsize=txt_size)
    axs2[3].set_xlim([0,L])
    axs2[3].set_ylim([-0.5,L+0.5])
    axs2[3].axis('off')
    axs2[3].set_title('Legend: health risk (population %)',fontsize=10,loc='left')

    plt.pause(0.1)

    if SAVE_TEST_AND_RISK:
        saveFigureToFile(plt,names_BIOMs,names_CONDs,gender_str,'2riskmaps_sizeEnsemb%d' %NN_ensemble_size)
    #endregion


    #region Figure 3 - 3D soft-map and risk surfaces
    # if PLOT_3D_RISKMAP:
    #     fig3, axs3 = plt.subplots(1,2,figsize=(12,5),subplot_kw={"projection": "3d"})
    #     mat_X, mat_Y = np.meshgrid(range(grid_width), range(grid_height))
    #
    #     axs3[0].plot_surface(mat_X, mat_Y, np.flip(softmax_map_2D, 0), cmap=class_colmap, edgecolors ="None", linewidth=0, antialiased=False)
    #     axs3[0].set_xlabel('%s' % long_descriptions[names_BIOMs[0]])
    #     if num_bioms>1:
    #         axs3[0].set_ylabel('%s' % long_descriptions[names_BIOMs[1]])
    #     axs3[0].set_zlabel('soft-max')
    #     axs3[0].get_xaxis().set_ticklabels([])
    #     axs3[0].get_yaxis().set_ticklabels([])
    #     axs3[0].get_zaxis().set_ticklabels([])
    #     axs3[0].set_title('Soft-max',fontsize=12)
    #
    #     axs3[1].plot_surface(mat_X, mat_Y, np.flip(risk_map, 0), cmap=risk_colmap, edgecolors ="None", linewidth=0, antialiased=False)
    #     axs3[1].set_xlabel('%s' % long_descriptions[names_BIOMs[0]])
    #     if num_bioms>1:
    #         axs3[1].set_ylabel('%s' % long_descriptions[names_BIOMs[1]])
    #     axs3[1].set_zlabel('health risk')
    #     axs3[1].get_xaxis().set_ticklabels([])
    #     axs3[1].get_yaxis().set_ticklabels([])
    #     axs3[1].get_zaxis().set_ticklabels([])
    #     axs3[1].set_title('Risk map',fontsize=12)
    #
    #     plt.pause(0.5)
    #
    #     if SAVE_TEST_AND_RISK:
    #         saveFigureToFile(plt, names_BIOMs, names_CONDs, gender_str, '3D')
    #endregion

