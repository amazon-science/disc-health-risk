# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import random
from datetime import datetime
import itertools
import torch

from DiscRisk_utilities import genderStringFromId, construct_model_folder_name, ownColormap_3colors, get_optimizer, \
    get_loss_func, setPointColors, saveFigureToFile, cuda_is_available, numpy_to_cuda, cuda_to_numpy, \
    calculate_inverse_frequency_class_weights, save_config_object, testNNEnsemble
from DiscRisk_dataloader import DataLoader
import DiscRisk_constants as CONSTS
from DiscRisk_models import *
from DiscRisk_params import _C as cfg
from DiscRisk_2_test import evaluate_nn_ensemble


# Main evaluation loop
def evaluate_exhaustive(do_arch_search=False, include_and_flag_missing_bioms=False, all_possible_BIOMs=None):

    gender_str = genderStringFromId(cfg.GENDER_ID)

    if all_possible_BIOMs is None:
        all_possible_BIOMs = ['RIDAGEYR', 'BMXBMI', 'RIDRETH1', 'RIAGENDR']

    archs = []

    if do_arch_search:
        max_depth = 2
        # hidden_layer_1_widths = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        if len(all_possible_BIOMs) == 1:
            hidden_layer_1_widths = [0]
        else:
            hidden_layer_1_widths = [0, 2, 3, 4, 6, 8, 16, 32, 64]
            try:
                idx = hidden_layer_1_widths.index(len(all_possible_BIOMs))
            except Exception as detail:
                hidden_layer_1_widths.append(len(all_possible_BIOMs))
            #try:
            #    idx = hidden_layer_1_widths.index(2*len(all_possible_BIOMs))
            #except Exception as detail:
            #    hidden_layer_1_widths.append(2*len(all_possible_BIOMs))

        for width in hidden_layer_1_widths:
            if width > 0:
                arch_this = [width]
            else:
                arch_this = []
            archs.append(arch_this)
        # pyramid
        reduction_fracs = [0.5]
        for depth in range(2, max_depth+1, 1):
            for hidden_layer_1_width in hidden_layer_1_widths:
                if hidden_layer_1_width == 0:
                    continue
                for reduction_frac in reduction_fracs:
                    arch_this = [hidden_layer_1_width]
                    for d in range(1, depth):
                        hidden_layer_this_width = int(arch_this[-1] * reduction_frac)
                        if hidden_layer_this_width < 2:
                            hidden_layer_this_width = 2
                        arch_this.append(hidden_layer_this_width)
                    try:
                        idx = archs.index(arch_this)
                    except Exception as detail:
                        archs.append(arch_this)
    else:
        archs.append(cfg.NN_ARCHITECTURE)

    big_table = [copy.deepcopy(all_possible_BIOMs)]
    big_table[0].append('TrainPos')
    big_table[0].append('TrainNeg')
    big_table[0].append('ValPos')
    big_table[0].append('ValNeg')
    big_table[0].append('TestPos')
    big_table[0].append('TestNeg')

    big_table[0].append('TestPos_female')
    big_table[0].append('TestNeg_female')
    big_table[0].append('TestPos_male')
    big_table[0].append('TestNeg_male')
    big_table[0].append('ValPos_female')
    big_table[0].append('ValNeg_female')
    big_table[0].append('ValPos_male')
    big_table[0].append('ValNeg_male')
    big_table[0].append('TrainPos_female')
    big_table[0].append('TrainNeg_female')
    big_table[0].append('TrainPos_male')
    big_table[0].append('TrainNeg_male')

    big_table[0].append('ArchVersion')
    big_table[0].append('TimeID')

    if cfg.IMPUTE_MISSING:
        fname_imputed = './data/NHANES/imputed_all.csv'
    else:
        fname_imputed = None
    dloader = DataLoader(fname_imputed=fname_imputed)

    cache_gender = copy.deepcopy(cfg.GENDER_ID)
    cfg.GENDER_ID = 2
    dloader_female = DataLoader(fname_imputed=fname_imputed)
    cfg.GENDER_ID = 1
    dloader_male = DataLoader(fname_imputed=fname_imputed)
    cfg.GENDER_ID = cache_gender


    fname_save_raw = '{}/{}_{}_raw_test.csv'.format(cfg.DIR_RESULTS, cfg.CSV_FNAME_TAG, cfg.NAMES_CONDS[0])

    for n_inputs in range(1, len(all_possible_BIOMs)+1):
        perms = list(itertools.combinations(all_possible_BIOMs, n_inputs))
        for perm in perms:
            names_BIOMs = list(perm)

            if len(names_BIOMs) == 0:
                continue

            # HACK
            if len(names_BIOMs) != len(all_possible_BIOMs):
                continue

            print('\nTrying BIOMs: {}'.format(names_BIOMs))
            print('\nIMPUTE_MISSING: {}'.format(cfg.IMPUTE_MISSING))

            cfg.NAMES_BIOMS = names_BIOMs

            # if include_ang_flag_missing_bioms is True, and complete_examples_only is True
            # we still filter for age, pregnant, smoking (depending on params), but let through any BIOMs
            # that are -1 (missing)
            dloader.load_nhanes(cfg, complete_examples_only=True, impute_missing=cfg.IMPUTE_MISSING,
                                include_and_flag_missing_bioms=include_and_flag_missing_bioms)

            cache_gender = copy.deepcopy(cfg.GENDER_ID)
            cfg.GENDER_ID = 2
            dloader_female.load_nhanes(cfg, complete_examples_only=True, impute_missing=cfg.IMPUTE_MISSING,
                                include_and_flag_missing_bioms=include_and_flag_missing_bioms)
            cfg.GENDER_ID = 1
            dloader_male.load_nhanes(cfg, complete_examples_only=True, impute_missing=cfg.IMPUTE_MISSING,
                                include_and_flag_missing_bioms=include_and_flag_missing_bioms)
            cfg.GENDER_ID = cache_gender


            for arch_version in range(len(archs)):

                time_id_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

                cfg.NN_ARCHITECTURE = archs[arch_version]
                arch_str = str(cfg.NN_ARCHITECTURE)
                arch_str = arch_str.replace(' ', '')
                arch_str = arch_str.replace(',', '-')
                print('Arch: {}'.format(arch_str))

                NN_ensemble = train(dloader, include_and_flag_missing_bioms=include_and_flag_missing_bioms)

                """
                data = dloader.get_data()
                is_test = data['is_train']
                num_test = np.count_nonzero(is_test)
                data_BIOMS_gt_norm_np = dloader.get_normalized_bioms(include_and_flag_missing_bioms=include_and_flag_missing_bioms)
                bioms_test_torch = numpy_to_cuda(data_BIOMS_gt_norm_np[is_test, :].astype(np.float32))
                NN_selected_net_in_ensemble = -1                # index of single NN to use, set to -1 for all ensemble
                softmax_test = testNNEnsemble(NN_ensemble, bioms_test_torch, NN_selected_net_in_ensemble)
                softmax_test_diff = softmax_test[:, 1] - softmax_test[:, 0]
                labels_gt_test_numpy = data['CON_gt'][is_test]
                if cfg.APPLY_SAMPLE_WEIGHTS:
                    sample_weights = data['SAMPLE_WEIGHT'][is_test]
                    sample_weights = sample_weights / np.sum(sample_weights)
                else:
                    sample_weights = np.ones(num_test) / num_test
                best_accuracy = 0
                best_thresh = 0
                for d in range(softmax_test_diff.size):
                    thresh = softmax_test_diff[d]
                    positive = (softmax_test_diff > thresh).astype(labels_gt_test_numpy.dtype)
                    accuracy = (positive == labels_gt_test_numpy)
                    accuracy = np.multiply(accuracy, sample_weights)
                    accuracy = np.sum(accuracy) * 100.0
                    if accuracy > best_accuracy:
                        best_thresh = thresh
                        best_accuracy = accuracy
                """

                fname_save_rds_array = '{}/rds/{}_val.csv'.format(cfg.DIR_RESULTS, time_id_str)
                eval_dict_val = evaluate_nn_ensemble(NN_ensemble, dloader,
                                                     apply_sample_weights=cfg.APPLY_SAMPLE_WEIGHTS,
                                                     partition='val',
                                                     include_and_flag_missing_bioms=include_and_flag_missing_bioms,
                                                     fname_save_rds_array=fname_save_rds_array,
                                                     fname_save_raw=fname_save_raw.replace('test', 'val'))
                fname_save_rds_array = '{}/rds/{}_val.csv'.format(cfg.DIR_RESULTS, time_id_str)
                eval_dict_test = evaluate_nn_ensemble(NN_ensemble, dloader,
                                                      apply_sample_weights=cfg.APPLY_SAMPLE_WEIGHTS,
                                                      partition='test',
                                                      include_and_flag_missing_bioms=include_and_flag_missing_bioms,
                                                      fname_save_raw=fname_save_raw,
                                                      fname_save_rds_array=fname_save_rds_array)
                eval_dict_train = evaluate_nn_ensemble(NN_ensemble, dloader,
                                                       apply_sample_weights=cfg.APPLY_SAMPLE_WEIGHTS,
                                                       partition='train',
                                                       include_and_flag_missing_bioms=include_and_flag_missing_bioms,
                                                       fname_save_raw=fname_save_raw.replace('test', 'train'))

                eval_dict_test_female = evaluate_nn_ensemble(NN_ensemble, dloader_female,
                                                             apply_sample_weights=cfg.APPLY_SAMPLE_WEIGHTS,
                                                             partition='test',
                                                             include_and_flag_missing_bioms=include_and_flag_missing_bioms)

                eval_dict_test_male = evaluate_nn_ensemble(NN_ensemble, dloader_male,
                                                           apply_sample_weights=cfg.APPLY_SAMPLE_WEIGHTS,
                                                           partition='test',
                                                           include_and_flag_missing_bioms=include_and_flag_missing_bioms)

                print('\n\nTrain Acc, Val Acc, Test Acc, Train RDS, Val RDS, Test RDS, Train AUC, Val AUC, Test AUC')
                print('{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(
                    eval_dict_train['accuracy'],
                    eval_dict_val['accuracy'],
                    eval_dict_test['accuracy'],
                    eval_dict_train['risk_disc_score'],
                    eval_dict_val['risk_disc_score'],
                    eval_dict_test['risk_disc_score'],
                    eval_dict_train['roc_area_under_curve'],
                    eval_dict_val['roc_area_under_curve'],
                    eval_dict_test['roc_area_under_curve']))

                stats = dloader.get_stats()
                stats_female = dloader_female.get_stats()
                stats_male = dloader_male.get_stats()

                # Update big table row
                for key in eval_dict_train.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'train_' + key
                    if full_key_name not in big_table[0]:
                        big_table[0].append(full_key_name)
                for key in eval_dict_val.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'val_' + key
                    if full_key_name not in big_table[0]:
                        big_table[0].append(full_key_name)
                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_' + key
                    if full_key_name not in big_table[0]:
                        big_table[0].append(full_key_name)

                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_female_' + key
                    if full_key_name not in big_table[0]:
                        big_table[0].append(full_key_name)
                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_male_' + key
                    if full_key_name not in big_table[0]:
                        big_table[0].append(full_key_name)

                big_table_row = [0] * len(big_table[0])
                for name_BIOM in names_BIOMs:
                    idx = all_possible_BIOMs.index(name_BIOM)
                    big_table_row[idx] = 1

                big_table_row[big_table[0].index('ArchVersion')] = arch_str
                big_table_row[big_table[0].index('TrainPos')] = stats['Train']['pos']
                big_table_row[big_table[0].index('TrainNeg')] = stats['Train']['neg']
                big_table_row[big_table[0].index('ValPos')] = stats['Val']['pos']
                big_table_row[big_table[0].index('ValNeg')] = stats['Val']['neg']
                big_table_row[big_table[0].index('TestPos')] = stats['Test']['pos']
                big_table_row[big_table[0].index('TestNeg')] = stats['Test']['neg']

                big_table_row[big_table[0].index('TestPos_female')] = stats_female['Test']['pos']
                big_table_row[big_table[0].index('TestNeg_female')] = stats_female['Test']['neg']
                big_table_row[big_table[0].index('TestPos_male')] = stats_male['Test']['pos']
                big_table_row[big_table[0].index('TestNeg_male')] = stats_male['Test']['neg']

                big_table_row[big_table[0].index('ValPos_female')] = stats_female['Val']['pos']
                big_table_row[big_table[0].index('ValNeg_female')] = stats_female['Val']['neg']
                big_table_row[big_table[0].index('ValPos_male')] = stats_male['Val']['pos']
                big_table_row[big_table[0].index('ValNeg_male')] = stats_male['Val']['neg']

                big_table_row[big_table[0].index('TrainPos_female')] = stats_female['Train']['pos']
                big_table_row[big_table[0].index('TrainNeg_female')] = stats_female['Train']['neg']
                big_table_row[big_table[0].index('TrainPos_male')] = stats_male['Train']['pos']
                big_table_row[big_table[0].index('TrainNeg_male')] = stats_male['Train']['neg']

                for key in eval_dict_train.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'train_' + key
                    big_table_row[big_table[0].index(full_key_name)] = eval_dict_train[key]
                for key in eval_dict_val.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'val_' + key
                    big_table_row[big_table[0].index(full_key_name)] = eval_dict_val[key]
                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_' + key
                    big_table_row[big_table[0].index(full_key_name)] = eval_dict_test[key]

                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_female_' + key
                    big_table_row[big_table[0].index(full_key_name)] = eval_dict_test_female[key]
                for key in eval_dict_test.keys():
                    if key == 'labels_pred':
                        continue
                    full_key_name = 'test_male_' + key
                    big_table_row[big_table[0].index(full_key_name)] = eval_dict_test_male[key]

                big_table.append(big_table_row)

                big_table_row[big_table[0].index('TimeID')] = time_id_str
                fname_save_params = '{}/params/{}.cfg'.format(cfg.DIR_RESULTS, time_id_str)
                save_config_object(cfg, fname_save_params)

                fname_save_big_table = '{}/csvs/{}_{}_table.csv'.format(cfg.DIR_RESULTS, cfg.CSV_FNAME_TAG, gender_str)
                dir_save_big_table = os.path.dirname(fname_save_big_table)
                if not os.path.exists(dir_save_big_table):
                    os.makedirs(dir_save_big_table)
                with open(fname_save_big_table, 'wt') as fid:
                    writer = csv.writer(fid)
                    for row in big_table:
                        writer.writerow(row)


    # Find best val
    idx_val = big_table[0].index('val_roc_area_under_curve')
    idx_test = big_table[0].index('test_roc_area_under_curve')
    max_roc_auc = -1
    best_test_roc_auc = -1
    for rowidx in range(1, len(big_table)):
        roc_auc = float(big_table[rowidx][idx_val])
        if roc_auc > max_roc_auc:
            best_test_roc_auc = float(big_table[rowidx][idx_test])
            max_roc_auc = roc_auc
    return best_test_roc_auc


def train(dloader, include_and_flag_missing_bioms=False):

    # default return values
    NN_ensemble = []

    plt.rcParams['toolbar'] = 'None'

    #print("\ncfg:\n{}".format(cfg))

    ############################################################################################################
    # MAIN -----------------------------------------------------------------------------------------------
    ############################################################################################################

    np.random.seed(12345)
    torch.manual_seed(12345)

    folder_name = construct_model_folder_name(cfg)

    #print('{}'.format(folder_name))

    # colormaps
    class_colmap = ownColormap_3colors(CONSTS.col_class_neg, CONSTS.col_class_unc, CONSTS.col_class_pos)
    risk_colmap = ownColormap_3colors(CONSTS.col_risk_neg, CONSTS.col_risk_unc, CONSTS.col_risk_pos)
    #endregion preliminaries

    data = dloader.get_data()

    num_bioms = data['BIOMs_gt'].shape[1]
    num_conds = data['CONDs_gt'].shape[1]
    num_rows = data['BIOMs_gt'].shape[0]
    gender_str = genderStringFromId(cfg.GENDER_ID)

    #region split train/test data, bagging, normalize and prepare data for Torch

    # getting low and high limits for visualization only
    lows_unnorm, highs_unnorm = np.zeros(num_bioms), np.zeros(num_bioms)
    for bio_id in range(num_bioms):
        full_col = data['BIOMs_gt'][:, bio_id]
        vec = full_col[full_col >= 0.0]
        lows_unnorm[bio_id], highs_unnorm[bio_id] = np.min(vec), np.max(vec)

    # normalize input features
    # if include_and_flag_missing_bios is True, then any remaining missing bioms will be set to to mean values
    # (note that imputation can be turned on) AND the valid mask will be concatenated to each example
    data_BIOMS_gt_norm_np = dloader.get_normalized_bioms(include_and_flag_missing_bioms=include_and_flag_missing_bioms)

    bioms_torch = numpy_to_cuda(data_BIOMS_gt_norm_np.astype(np.float32))
    labels_gt_torch = numpy_to_cuda(data['CON_gt'])

    #endregion prepare data for Torch

    #region Figure 1 plot left - training data points
    if cfg.PLOT_TRAIN_PROGRESS:
        labels_train_cols = setPointColors(labels_gt_sparse_train_to, CONSTS.col_class_pos, CONSTS.col_class_neg, CONSTS.col_class_unc)     # setting colours for ground truth labels - for visualization only

        vis_data_train_pts_x = data['BIOMs_gt'][indxs_sparse_train, 0]
        vis_data_train_pts_y = (np.random.rand(num_sparse_train_data) - 0.5)   # initializing Ys as random for case when num_biomarkers=1
        if num_bioms > 1:
            vis_data_train_pts_y = data['BIOMs_gt'][indxs_sparse_train, 1]

        fig1, axs1 = plt.subplots(1, 3, figsize=(cfg.FIG_W, cfg.FIG_H))
        axs1[0].scatter(vis_data_train_pts_x, vis_data_train_pts_y, c=labels_train_cols, s=2, linewidth=0)
        axs1[0].set_xlabel('%s' % CONSTS.long_biom_descriptions[cfg.NAMES_BIOMS[0]])
        axs1[0].set_xlim([lows_unnorm[0], highs_unnorm[0]])
        axs1[0].set_title('Training (%s, age %d-%d, n_conds %d)' % (gender_str, cfg.AGE_MIN, cfg.AGE_MAX, num_conds))
        if num_bioms == 1:
            axs1[0].set_yticks([])
            axs1[0].set_ylim([np.min(vis_data_train_pts_y), np.max(vis_data_train_pts_y)])
        else:
            axs1[0].set_ylabel('%s' % CONSTS.long_biom_descriptions[cfg.NAMES_BIOMS[1]])
            axs1[0].set_ylim([lows_unnorm[1], highs_unnorm[1]])

        plt.pause(0.5)

        # get ticks positions and labels for later use
        xticks_unnorm_pos, xticks_unnorm_lab = axs1[0].get_xticks(), axs1[0].get_xticklabels()
        yticks_unnorm_pos, yticks_unnorm_lab = axs1[0].get_yticks(), axs1[0].get_yticklabels()
    #endregion figure 1 - training data points

    # getting the right neural network architecture/model
    NN_architecture = copy.deepcopy(cfg.NN_ARCHITECTURE)
    NN_architecture.insert(0, data_BIOMS_gt_norm_np.shape[1])
    NN_architecture.append(2)

    loss_func = get_loss_func(cfg.NN_LOSS)
    is_val = copy.deepcopy(data['is_val'])
    num_val = np.count_nonzero(is_val)

    if cfg.APPLY_SAMPLE_WEIGHTS:
        val_sample_weights = numpy_to_cuda(data['SAMPLE_WEIGHT'][is_val].astype(np.float32))
    else:
        val_sample_weights = numpy_to_cuda(torch.ones(num_val, dtype=torch.FloatTensor))

    if cfg.APPLY_INVERSE_CLASS_FREQUENCY_WEIGHTS:
        new_weights = calculate_inverse_frequency_class_weights(labels_gt_torch[is_val])
        val_sample_weights = torch.multiply(val_sample_weights, new_weights)

    val_sample_weights = torch.div(val_sample_weights, torch.sum(val_sample_weights))

    #region train NN model
    NN_ensemble_max_val_acc = np.zeros((cfg.NN_ENSEMBLE_SIZE,))
    NN_ensemble = [0] * cfg.NN_ENSEMBLE_SIZE
    for net_idx in range(cfg.NN_ENSEMBLE_SIZE):

        # net file to save to disk
        for file_idx in range(10000):
            file_name_NN_model = '%s/net_%02d.pkl' %(folder_name, file_idx)
            if not os.path.exists(file_name_NN_model):
                break
        file_name_NN_model = '%s/net_%02d.pkl' %(folder_name, file_idx)
        print('\nTraining net %d/%d in ensemble (file %02d) [biom_model_id %d, gender %d]' %(net_idx,
                                                                                             cfg.NN_ENSEMBLE_SIZE,
                                                                                             file_idx,
                                                                                             cfg.BIOM_MODEL_IDX,
                                                                                             cfg.GENDER_ID))

        net = NeuralNet_General(NN_architecture, activation=cfg.NN_ACTIVATION, add_batch_norm=cfg.NN_ADD_BATCH_NORM, momentum=cfg.NN_BATCH_NORM_MOMENTUM)
        NN_learning_rate = cfg.NN_LEARNING_RATE_INIT
        optimizer = get_optimizer(net, cfg.NN_OPTIM, NN_learning_rate, momentum=cfg.NN_OPTIMIZER_MOMENTUM)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100,
                                                               verbose=False,
                                                               min_lr=1e-8)

        # randomly sample bag of training data
        is_train = copy.deepcopy(data['is_train'])
        is_bag = np.random.uniform(size=is_train.size) > (1 - cfg.NN_PERCTG_IN_BAG)
        num_train = np.count_nonzero(is_train)
        is_train = np.logical_and(is_train, is_bag)
        num_bag_train_data = np.count_nonzero(is_train)

        if cfg.APPLY_SAMPLE_WEIGHTS:
            train_sample_weights = numpy_to_cuda(data['SAMPLE_WEIGHT'][is_train].astype(np.float32))
        else:
            train_sample_weights = numpy_to_cuda(torch.ones(num_bag_train_data, dtype=torch.FloatTensor))

        if cfg.APPLY_INVERSE_CLASS_FREQUENCY_WEIGHTS:
            new_weights = calculate_inverse_frequency_class_weights(labels_gt_torch[is_train])
            train_sample_weights = torch.multiply(train_sample_weights, new_weights)

        train_sample_weights = torch.div(train_sample_weights, torch.sum(train_sample_weights))


        """
        if cfg.PLOT_TRAIN_PROGRESS:
            vis_bag_train_pts_x = vis_data_train_pts_x[indxs_bag_train]
            vis_bag_train_pts_y = vis_data_train_pts_y[indxs_bag_train]
        """
        print('\ttraining bag size %d/%d' % (num_bag_train_data, num_train))

        if cuda_is_available:
            net.to('cuda:0')

        # Train!
        accuracy_profile = []
        max_val_acc, max_train_acc, max_val_it, max_train_it, max_val_prev = 0, 0, 0, 0, 0
        for iter_id in range(cfg.NN_MAX_NUM_TRAIN_ITERS):       # training iterations

            nn_activations_train = net(bioms_torch[is_train, :])      # apply network to input biomarkers to obtain activations of output layer
            losses_train = loss_func(nn_activations_train, labels_gt_torch[is_train])  # loss current value

            if len(losses_train.shape) > 1:
                losses_train = torch.mean(losses_train, dim=1)

            losses_train = torch.multiply(losses_train, train_sample_weights)
            loss_train = torch.sum(losses_train)

            if cfg.NN_WEIGHT_L1_REG > 0.0:
                num_layers = len(net.Layers)
                x = []
                for layer_idx in range(num_layers):
                    x.append(net.Layers[layer_idx].weight.view(-1))
                x = torch.cat(x)
                l1_regularization = cfg.NN_WEIGHT_L1_REG * torch.norm(x, 1)
                loss_train += l1_regularization

            #if iter_id>0 and (iter_id % cfg.NN_LR_REDUCTION_STEP) == 0:
            #    NN_learning_rate = 0.5 * NN_learning_rate               # halving the learning rate
            #    optimizer.param_groups[0]['lr'] = NN_learning_rate      # setting the new learning rate value
            optimizer.zero_grad()                                       # clear gradients in preparation for next trainining iteration
            loss_train.backward()                                    # backpropagation, compute gradients
            optimizer.step()                                            # apply gradients to modify network weights

            if iter_id > 0 and (iter_id == cfg.NN_MAX_NUM_TRAIN_ITERS - 1 or iter_id % cfg.NN_ACC_COMPUTE_STEP == 0):

                with torch.no_grad():
                    # applying network to estimate class labels for training points
                    labels_pred_train_to = torch.max(nn_activations_train, 1)[1]          # predicted labels - applying max to hard-assign class labels to input

                    if len(labels_gt_torch.shape) > 1:
                        train_accuracy = (labels_pred_train_to == labels_gt_torch[is_train, -1])
                    else:
                        train_accuracy = (labels_pred_train_to == labels_gt_torch[is_train])
                    #train_accuracy = train_accuracy.type(torch.FloatTensor)
                    train_accuracy = torch.multiply(train_accuracy, train_sample_weights)
                    train_accuracy = torch.sum(train_accuracy) * 100.0

                    #num_correct_train = np.count_nonzero(labels_pred_train_to == labels_gt_torch[is_train])  # counting number of correct predictions
                    #train_accuracy = (100.0 * num_correct_train) / num_bag_train_data  # measuring training accuracy

                    # applying network to estimate class labels for sparse testing points
                    nn_activations_val = net(bioms_torch[is_val, :])                # apply network to input biomarkers to obtain activations of output layer
                    losses_val = loss_func(nn_activations_val, labels_gt_torch[is_val])

                    if len(losses_val.shape) > 1:
                        losses_val = torch.mean(losses_val, dim=1)

                    losses_val = torch.multiply(losses_val, val_sample_weights)
                    loss_val = torch.sum(losses_val)

                    labels_pred_val_to = torch.max(nn_activations_val, 1)[1]          # predicted labels - applying max to hard-assign class labels to input
                    if len(labels_gt_torch.shape) > 1:
                        val_accuracy = (labels_pred_val_to == labels_gt_torch[is_val, -1])
                    else:
                        val_accuracy = (labels_pred_val_to == labels_gt_torch[is_val])
                    val_accuracy = torch.multiply(val_accuracy, val_sample_weights)
                    val_accuracy = torch.sum(val_accuracy) * 100.0

                scheduler.step(loss_val)

                #num_correct_val = np.count_nonzero(labels_pred_val_to == labels_gt_torch[is_val]) # counting number of correct predictions
                #val_accuracy = (100.0 * num_correct_val) / num_val   # measuring test accuracy

                # accuracy profile todo improve here
                accuracy_profile.append([iter_id, cuda_to_numpy(train_accuracy), cuda_to_numpy(val_accuracy)])  # append latest measurements
                acc_prof_np = np.array(accuracy_profile)  # list to np
                sm_x, sm_train, sm_val = acc_prof_np[:, 0], acc_prof_np[:, 1], acc_prof_np[:, 2]

                # max accuracies and saving NN model
                max_train_acc, max_train_it = np.max(sm_train), sm_x[np.argmax(sm_train)]
                max_val_acc, max_val_it = np.max(sm_val), sm_x[np.argmax(sm_val)]
                if max_val_acc > max_val_prev:
                    max_val_prev = max_val_acc
                    NN_ensemble_max_val_acc[net_idx] = max_val_acc
                    NN_ensemble[net_idx] = net
                    if cfg.SAVE_BEST_TEACC:
                        torch.save(net, file_name_NN_model) # save best model to file for highest test accuracy

                #if not cfg.SAVE_BEST_TEACC:
                #    torch.save(net, file_name_NN_model)  # save the latest model to file independent of its test accuracy

                current_lr = optimizer.param_groups[0]['lr']

                print('\r[%d/%d] lr %f   max val acc %1.3f%% (%1.1f%%)   max train acc %1.3f%% (%1.1f%%)   val loss %f  train loss %f' \
                      % (iter_id, cfg.NN_MAX_NUM_TRAIN_ITERS, current_lr, max_val_acc, val_accuracy, max_train_acc, train_accuracy, loss_val.item(), loss_train.item()), end='')

                if current_lr < 1e-6:
                    break

                #region figure 1 plot mid & right - training iterations
                """
                if cfg.PLOT_TRAIN_PROGRESS:
                    softmax_train = torch.nn.functional.softmax(nn_activations_train.detach(),1).numpy()[:,1]
                    labels_pred_train_cols = setPointColors(softmax_train, col_class_pos, col_class_neg, col_class_unc)  # colours of predicted labels, for visualization only

                    axs1[1].clear()
                    axs1[1].scatter(vis_bag_train_pts_x, vis_bag_train_pts_y, c=labels_pred_train_cols, s=2, linewidth=0)
                    # axs1[1].scatter(vis_data_train_pts_x, vis_data_train_pts_y, c=labels_pred_train_cols, s=2, linewidth=0)
                    axs1[1].set_xlabel('%s' % CONSTS.long_biom_descriptions[cfg.NAMES_BIOMS[0]])
                    axs1[1].set_xlim([lows_unnorm[0], highs_unnorm[0]])
                    axs1[1].set_title('NN %d/%d Class tr. data (N %d/%d/%d %1.1f%%)' % (net_idx, cfg.NN_ENSEMBLE_SIZE, num_bag_train_data, num_sparse_train_data, num_gt_data_all, 100 * cfg.NN_TRAIN_V_ALL_PERC))
                    if cfg.NUM_BIOMS == 1:
                        axs1[1].set_yticks([])
                        axs1[1].set_ylim([np.min(vis_data_train_pts_y), np.max(vis_data_train_pts_y)])
                    else:
                        axs1[1].set_ylabel('%s' % CONSTS.long_biom_descriptions[cfg.NAMES_BIOMS[1]])
                        axs1[1].set_ylim([lows_unnorm[1], highs_unnorm[1]])

                    axs1[2].clear()
                    axs1[2].plot(sm_x, sm_train, '-', color=col_train_accuracy, linewidth=1)    # training accuracy
                    axs1[2].plot(max_train_it, max_train_acc, '.', color=col_train_accuracy, markersize=8)
                    axs1[2].plot(sm_x, sm_test, '-', color=col_test_accuracy, linewidth=2)     # testing accuracy
                    axs1[2].plot(max_test_it, max_test_acc, '.', color=col_test_accuracy, markersize=10)
                    axs1[2].set_title('NN %d/%d Test acc. %.1f%% Train acc. %.1f%%' % (net_idx, NN_ensemble_size, max_test_acc, max_train_acc))
                    axs1[2].set_xlabel('Training Iterations')
                    axs1[2].set_ylabel('Test / Train accuracy')
                    plt.pause(0.1)
                """
                #endregion  figure 1 - training iterations

    if cfg.SAVE_TRAIN_PROGRESS:
        saveFigureToFile(plt, cfg.NAMES_BIOMS, cfg.NAMES_CONDS, gender_str, '1train_%d' % net_idx)

    plt.show()
    #endregion train NN model

    return NN_ensemble


def parse_args():

    """Returns arguments parsed from command line."""

    class ArgsFormatter(argparse.ArgumentDefaultsHelpFormatter,
                        argparse.RawDescriptionHelpFormatter):
        """Formats help with default argument values."""
        pass

    parser = argparse.ArgumentParser(
        formatter_class=ArgsFormatter,
        description='Train measure net')
    parser.add_argument('--config_file', type=str, default='./configs/default.yaml',
                        help='Specify the config file to use')

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_args()

    print("Loading config from {}".format(args.config_file))
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    roc_auc = evaluate_exhaustive(do_arch_search=False, include_and_flag_missing_bioms=False, all_possible_BIOMs=cfg.NAMES_BIOMS)
