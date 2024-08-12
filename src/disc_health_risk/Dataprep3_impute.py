# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import argparse
import csv
import time
import torch

from DiscRisk_utilities import get_optimizer, get_loss_func, norm2unnorm, randomly_omit_columns, numpy_to_cuda, \
    cuda_to_numpy, cuda_is_available
from DiscRisk_dataloader import DataLoader
import DiscRisk_constants as CONSTS
from DiscRisk_models import *
from DiscRisk_params import _C as cfg


def impute_values():

    # Assume for now that values are scalar instead of categorical
    dloader = DataLoader()
    dloader.load_nhanes(cfg, complete_examples_only=False, impute_missing=False)

    data = dloader.get_data()
    data_BIOMS_gt_norm_np = dloader.get_normalized_bioms()

    valid_gt_np = data_BIOMS_gt_norm_np >= 0

    bioms_torch = numpy_to_cuda(data_BIOMS_gt_norm_np.astype(np.float32))
    valid_torch = numpy_to_cuda(valid_gt_np.astype(np.float32))

    loss_func = get_loss_func(cfg.NN_LOSS)
    is_val = copy.deepcopy(data['is_val'])
    is_test = copy.deepcopy(data['is_test'])
    num_val = np.count_nonzero(is_val)

    if cfg.APPLY_SAMPLE_WEIGHTS:
        val_sample_weights = numpy_to_cuda(data['SAMPLE_WEIGHT'][is_val].astype(np.float32))
    else:
        val_sample_weights = numpy_to_cuda(torch.ones(num_val, dtype=torch.FloatTensor))
    val_sample_weights = torch.div(val_sample_weights, torch.sum(val_sample_weights))

    NN_ensemble_min_val = np.zeros((cfg.NN_ENSEMBLE_SIZE,))
    NN_ensemble = [0] * cfg.NN_ENSEMBLE_SIZE
    for net_idx in range(cfg.NN_ENSEMBLE_SIZE):

        np.random.seed(net_idx)

        print('\nModel %d of %d:' % (net_idx, cfg.NN_ENSEMBLE_SIZE))

        # set fake_np to zeros, because that is what we get when we take mu and normalize it (mu - mu ==> 0)
        fake_np = np.zeros_like(data_BIOMS_gt_norm_np, dtype=np.float32)
        fake_torch = numpy_to_cuda(fake_np)

        num_features = data_BIOMS_gt_norm_np.shape[1]
        net = AENet_General(num_features * 2,
                            activation=cfg.NN_ACTIVATION,
                            depth_to_feature_representation=cfg.DEPTH_TO_IMPUTE_FEATURE_REPRESENTATION,
                            width_of_feature_representation=cfg.WIDTH_OF_IMPUTE_FEATURE_REPRESENTATION,
                            momentum=cfg.NN_BATCH_NORM_MOMENTUM,
                            add_batch_norm=cfg.NN_ADD_BATCH_NORM)

        if cuda_is_available:
            net.to('cuda:0')

        NN_learning_rate = cfg.NN_LEARNING_RATE_INIT
        optimizer = get_optimizer(net, cfg.NN_OPTIM, NN_learning_rate, momentum=cfg.NN_OPTIMIZER_MOMENTUM)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3,
                                                               verbose=False, min_lr=1e-8)

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
        train_sample_weights = torch.div(train_sample_weights, torch.sum(train_sample_weights))

        # Train!
        initial_time = time.time()
        min_val_loss = np.inf
        min_val_iter = 0
        min_train_loss = np.inf
        min_train_iter = 0
        for iter_id in range(cfg.NN_MAX_NUM_TRAIN_ITERS):       # training iterations

            np.random.seed(net_idx*1000000 + iter_id)

            # Randomly choose one value to mask off
            inputs_torch_train = bioms_torch[is_train, :].detach().clone()
            fake_torch_train = fake_torch[is_train, :].detach().clone()
            valid_torch_train_orig = valid_torch[is_train, :].detach().clone()

            valid_torch_train_omit = randomly_omit_columns(valid_torch_train_orig)
            missing_mask = valid_torch_train_omit < 0.5
            inputs_torch_train[missing_mask] = fake_torch_train[missing_mask]
            inputs_torch_train = torch.cat((inputs_torch_train, valid_torch_train_omit), dim=1)

            nn_activations_train = net(inputs_torch_train)  # apply network to input biomarkers to obtain activations of output layer

            losses_train = loss_func(nn_activations_train[:, 0:num_features], bioms_torch[is_train, :])  # loss current value

            train_sample_weights_this = torch.multiply(train_sample_weights[:, None], valid_torch_train_orig)
            den = torch.sum(train_sample_weights_this)
            den = torch.clamp(den, min=0.000001)
            train_sample_weights_this = torch.div(train_sample_weights_this, den)
            losses_train = torch.multiply(losses_train, train_sample_weights_this)
            loss_train = torch.sum(losses_train)

            optimizer.zero_grad()  # clear gradients in preparation for next trainining iteration
            loss_train.backward()  # backpropagation, compute gradients
            optimizer.step()       # apply gradients to modify network weights

            if iter_id > 0 and (iter_id == cfg.NN_MAX_NUM_TRAIN_ITERS - 1 or iter_id % cfg.NN_ACC_COMPUTE_STEP == 0):

                with torch.no_grad():
                    # applying network to estimate class labels for sparse testing points
                    inputs_torch_val = bioms_torch[is_val, :].detach().clone()
                    fake_torch_val = fake_torch[is_val, :].detach().clone()
                    valid_torch_val_orig = valid_torch[is_val, :].detach().clone()

                    valid_torch_val_omit = randomly_omit_columns(valid_torch_val_orig)
                    missing_mask = valid_torch_val_omit < 0.5
                    inputs_torch_val[missing_mask] = fake_torch_val[missing_mask]
                    inputs_torch_val = torch.cat((inputs_torch_val, valid_torch_val_omit), dim=1)

                    nn_activations_val = net(inputs_torch_val)  # apply network to input biomarkers to obtain activations of output layer

                    val_sample_weights_this = torch.multiply(val_sample_weights[:, None], valid_torch_val_orig)
                    den = torch.sum(val_sample_weights_this)
                    den = torch.clamp(den, min=0.000001)
                    val_sample_weights_this = torch.div(val_sample_weights_this, den)

                    losses_val = loss_func(nn_activations_val[:, 0:num_features], bioms_torch[is_val, :])
                    losses_val = torch.multiply(losses_val, val_sample_weights_this)
                    loss_val = torch.sum(losses_val)

                current_lr = optimizer.param_groups[0]['lr']
                scheduler.step(loss_val)

                if loss_val.item() < min_val_loss:
                    min_val_loss = loss_val.item()
                    min_val_iter = iter_id
                    NN_ensemble[net_idx] = net
                    NN_ensemble_min_val[net_idx] = min_val_loss

                    if cfg.SAVE_BEST_TEACC:
                        #torch.save(net, file_name_NN_model) # save best model to file for highest test accuracy
                        pass

                if loss_train.item() < min_train_loss:
                    min_train_loss = loss_train.item()
                    min_train_iter = iter_id

                time_now = time.time()
                elapsed_time = time_now - initial_time
                time_per_epoch = elapsed_time / (iter_id+1)
                print('\r\t[%d/%d] tpe %f, lr %f, curr val %f \tmin val %f (it %d) \tcurr train %f \tmin train %f (it %d) \t' \
                      % (iter_id, cfg.NN_MAX_NUM_TRAIN_ITERS, time_per_epoch, current_lr, loss_val.item(), min_val_loss, min_val_iter,
                         loss_train.item(), min_train_loss, min_train_iter), end='')

                if current_lr < 1e-7:
                    break

            if iter_id > 0 and (iter_id == cfg.NN_MAX_NUM_TRAIN_ITERS - 1 or iter_id % cfg.NN_STEPS_TO_IMPUTE_FAKE == 0):
                with torch.no_grad():
                    # applying network to estimate class labels for sparse testing points
                    inputs_torch_val = bioms_torch.detach().clone()
                    fake_torch_val = fake_torch.detach().clone()
                    valid_torch_val_orig = valid_torch.detach().clone()

                    missing_mask = valid_torch_val_orig < 0.5
                    inputs_torch_val[missing_mask] = fake_torch_val[missing_mask]
                    inputs_torch_val = torch.cat((inputs_torch_val, valid_torch_val_orig), dim=1)

                    nn_activations_val = net(inputs_torch_val)      # apply network to input biomarkers to obtain activations of output layer
                    fake_torch = nn_activations_val[:, 0:num_features]

    with torch.no_grad():
        # applying network to estimate class labels for sparse testing points
        inputs_torch_val = bioms_torch.detach().clone()
        valid_torch_val_orig = valid_torch.detach().clone()
        missing_mask = valid_torch_val_orig < 0.5

        fake_np = np.zeros_like(data_BIOMS_gt_norm_np, dtype=np.float32)
        fake_torch_val = numpy_to_cuda(fake_np)

        for fake_iter in range(100):
            inputs_torch_val[missing_mask] = fake_torch_val[missing_mask]
            inputs_torch_val_cat = torch.cat((inputs_torch_val, valid_torch_val_orig), dim=1)
            nn_activations_val = net(inputs_torch_val_cat)      # apply network to input biomarkers to obtain activations of output layer
            fake_torch_val = nn_activations_val[:, 0:num_features]

    for only_one in range(2):
        errs_test_all = dict()
        mape_test_all = dict()
        for net_idx, net in enumerate(NN_ensemble):
            # Show real errors
            with torch.no_grad():
                inputs_torch_test = bioms_torch[is_test, :].detach().clone()
                fake_torch_test = fake_torch[is_test, :].detach().clone()
                valid_torch_test_orig = valid_torch[is_test, :].detach().clone()

                if only_one == 1:
                    valid_torch_test_omit = randomly_omit_columns(valid_torch_test_orig, frac_omit_one=1.0, frac_omit_mask=0.0)
                else:
                    valid_torch_test_omit = randomly_omit_columns(valid_torch_test_orig, frac_omit_one=0.0, frac_omit_mask=1.0)
                missing_mask = valid_torch_test_omit < 0.5
                inputs_torch_test[missing_mask] = fake_torch_test[missing_mask]
                inputs_torch_test = torch.cat((inputs_torch_test, valid_torch_test_omit), dim=1)

                nn_activations_test = net(inputs_torch_test)
                nn_activations_test = cuda_to_numpy(nn_activations_test[:, 0:num_features])
                nn_activations_test = norm2unnorm(nn_activations_test, dloader.mu, dloader.std)

                gt_bioms = cuda_to_numpy(bioms_torch[is_test, :])
                gt_bioms = norm2unnorm(gt_bioms, dloader.mu, dloader.std)
                select_mask = cuda_to_numpy(valid_torch_test_orig - valid_torch_test_omit) > 0.5
                errs = nn_activations_test - gt_bioms
                errs_mape = np.divide(errs, np.clip(gt_bioms, 0.000001, None))
                errs_mape = np.abs(errs_mape)
                errs = np.abs(errs)
                for c in range(select_mask.shape[1]):
                    errs_this = errs[select_mask[:, c], c]
                    mape_this = errs_mape[select_mask[:, c], c]
                    name = cfg.NAMES_BIOMS[c]
                    if name in errs_test_all:
                        errs_test_all[name] = np.concatenate((errs_test_all[name], errs_this), axis=0)
                        mape_test_all[name] = np.concatenate((mape_test_all[name], mape_this), axis=0)
                    else:
                        errs_test_all[name] = errs_this
                        mape_test_all[name] = mape_this

        print('')
        if only_one == 1:
            print('Test impute errors for {}-model ensemble, ONLY OMIT ONE:'.format(len(NN_ensemble)))
        else:
            print('Test impute errors for {}-model ensemble, OMIT FULL MASKS:'.format(len(NN_ensemble)))
        print('Biometric, Examples, MAE, P90, MAPE')
        for name in errs_test_all.keys():
            errs_this = errs_test_all[name]
            mape_this = mape_test_all[name]
            num_errs = errs_this.size
            if num_errs == 0:
                mae = -1.0
                p90 = -1.0
                mape = -1.0
            else:
                mae = np.mean(errs_this)
                p90 = np.percentile(errs_this, 90)
                mape = 100.0 * np.mean(mape_this)
            long_name = CONSTS.long_biom_descriptions[name]
            print('{}, {}, {:.2f}, {:.2f}, {:.2f}'.format(long_name, num_errs, mae, p90, mape))

    # save out
    inputs_torch_all = bioms_torch.detach().clone()
    fake_torch_all = fake_torch.detach().clone()
    valid_torch_orig = valid_torch.detach().clone()
    missing_mask = valid_torch_orig < 0.5
    inputs_torch_all[missing_mask] = fake_torch_all[missing_mask]
    inputs_torch_all = torch.cat((inputs_torch_all, valid_torch_orig), dim=1)
    nn_activations_accum = 0
    for net_idx, net in enumerate(NN_ensemble):
        # Show real errors
        with torch.no_grad():
            nn_activations_all = net(inputs_torch_all)
            nn_activations_all = cuda_to_numpy(nn_activations_all[:, 0:num_features])
            nn_activations_all = norm2unnorm(nn_activations_all, dloader.mu, dloader.std)
            nn_activations_accum += nn_activations_all
    nn_activations_accum /= len(NN_ensemble)

    fname_save = '{}/imputed_part{}.csv'.format(CONSTS.dir_NHANES_dataset, cfg.PARTITION_VERSION)
    fieldnames = ['ID', 'GEN', 'AGE', 'SAMPLE_WEIGHT']
    for name in cfg.NAMES_TO_IMPUTE:
        fieldnames.append(name)
    for cond in cfg.NAMES_CONDS:
        fieldnames.append(cond)
    fieldnames.append('CON_gt')
    valid_rows = np.zeros((nn_activations_accum.shape[0]), dtype=bool)
    with open(fname_save, 'w') as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx, id in enumerate(data['ID']):
            row_dict = dict()
            row_dict['ID'] = id
            row_dict['GEN'] = data['GEN'][row_idx]
            row_dict['AGE'] = data['AGE'][row_idx]
            row_dict['SAMPLE_WEIGHT'] = data['SAMPLE_WEIGHT'][row_idx]
            for name_idx, name in enumerate(cfg.NAMES_BIOMS):
                if name in cfg.NAMES_TO_IMPUTE:
                    row_dict[name] = nn_activations_accum[row_idx, name_idx]
            for cond_idx, cond in enumerate(cfg.NAMES_CONDS):
                row_dict[cond] = data['CONDs_gt'][row_idx, cond_idx]
            row_dict['CON_gt'] = data['CON_gt'][row_idx]

            good = True
            for key_this in row_dict.keys():
                try:
                    if row_dict[key_this] == -1:
                        good = False
                        break
                except Exception as detail:
                    print('key = {}, value = {}'.format(key_this, row_dict[key_this]))
                    import pudb; pudb.set_trace()

            if good:
                writer.writerow(row_dict)

            valid_rows[row_idx] = good

    print('{} valid rows out of {} total rows'.format(np.count_nonzero(valid_rows), nn_activations_accum.shape[0]))

    avg_min_val = np.mean(NN_ensemble_min_val)
    return avg_min_val


def aggregate_csv(num_parts):

    print('Aggregating CSVs')

    fname_save = '{}/imputed_all.csv'.format(CONSTS.dir_NHANES_dataset)

    all_rows_accum = dict()
    all_rows_total = dict()

    for part in range(1, num_parts + 1):
        fname_part = '{}/imputed_part{}.csv'.format(CONSTS.dir_NHANES_dataset, part)
        with open(fname_part, 'rt') as fid_in:
            reader = csv.DictReader(fid_in)
            for row in reader:
                id_this = float(row['ID'])
                if id_this in all_rows_accum.keys():
                    for key in all_rows_accum[id_this].keys():
                        all_rows_accum[id_this][key] = float(all_rows_accum[id_this][key]) + float(row[key])
                    all_rows_total[id_this] += 1.0
                else:
                    all_rows_accum[id_this] = copy.deepcopy(row)
                    all_rows_total[id_this] = 1.0

    ids_all = list(all_rows_accum.keys())
    ids_all.sort()

    with open(fname_save, 'wt') as fid_out:
        writer = csv.DictWriter(fid_out, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row_idx, id_this in enumerate(ids_all):
            row = all_rows_accum[id_this]
            for key in row.keys():
                row[key] = float(row[key]) / all_rows_total[id_this]
            writer.writerow(row)


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

    cfg.NN_LOSS = 'l1loss'
    cfg.NAMES_BIOMS = [
        'BMXWT',  #'WEIGHT (Kg)',
        'BMXHT',  #'HEIGHT (cm)',
        'BMXTHICR',  #'THIGH CIRC (cm)',
        'BMXWAIST',  #'WAIST CIRC (cm)',
        'BMXHIP',  #'HIP CIRC (cm)',
        'BMXARMC',  #"ARM CIRC (cm)",
        'BMXLEG',  #'Upper leg length (cm)',
        'BMXCALF',  #'Maximum calc circumference (cm)',
        'BMXTRI',  #'Triceps skinfold (mm)',
        'BMXSUB',  #'Subscapular Skinfold (mm)',
        'BMXRECUM',  #'Recumbent Length (cm)',
        'BMXSAD1',  #'Sagittal Abdominal Diameter 1st (cm)',
        'BMXSAD2',  #'Sagittal Abdominal Diameter 2nd (cm)',
        'BMXSAD3',  #'Sagittal Abdominal Diameter 3rd (cm)',
        'BMXSAD4',  #'Sagittal Abdominal Diameter 4th (cm)',
        'BMDAVSAD',  #'Average Sagittal Abdominal Diameter (cm)',
        'BMXBMI',  #'BMI (Kg/m2)',
        'DXDTOPF',  #'PERC BODY FAT (%)',
        'DXDTOFAT',  #'BODY FAT MASS (g)',
        'DXDTRPF',  #'PERC TRUNK FAT (%)',
        'DXXRLFAT',  #'RIGHT LEG FAT (g)',
        'DXXRAFAT',  #'RIGHT ARM FAT (g)',
        'DXXLLFAT',  #'LEFT LEG FAT (g)',
        'DXXLAFAT',  #'LEFT ARM FAT (g)',
        'DXDLAPF',  #'PERC LEFT ARM (%)',
        'DXDRAPF',  #'PERC RIGHT ARM (%)',
        'DXDLLPF',  #'PERC LEFT LEG (%)',
        'DXDRLPF',  #'PERC RIGHT LEG (%)',
        'DXDTOBMC',  #'TOTAL BODY MINERAL CONTENT (g)',
        'DXDTOBMD',  #'TOTAL BODY MINERAL DENSITY (g/cm^2)',
        'DXXHEA',  #'Head Area (cm^2)',
        'DXXHEBMC',  #'Head Bone Mineral Content (g)',
        'DXXHEBMD',  #'Head Bone Mineral Density (g/cm^2)',
        'DXXHEFAT',  #'Head Fat (g)',
        'DXDHELE',  #'Head Lean excl BMC (g)',
        'DXXHELI',  #'Head Lean incl BMC (g)',
        'DXDHETOT',  #'Head Total (g)',
        'DXDHEPF',  #'Head Percent Fat',
        'DXXLAA',  #'Left Arm Area (cm^2)',
        'DXXLABMC',  #'Left Arm BMC (g)',
        'DXXLABMD',  #'Left Arm BMD (g/cm^2)',
        'DXDLALE',  #'Left Arm Lean excl BMC (g)',
        'DXXLALI',  #'Left Arm Lean incl BMC (g)',
        'DXDLATOT',  #'Left Arm Total (g)',
        'DXXLLA',  #'Left Leg Area (cm^2)',
        'DXXLLBMC',  #'Left Leg BMC (g)',
        'DXXLLBMD',  #'Left Leg BMD (g/cm^2)',
        'DXDLLLE',  #'Left Leg Lean excl BMC (g)',
        'DXXLLLI',  #'Left Leg Lean incl BMC (g)',
        'DXDLLTOT',  #'Left Leg Total (g)',
        'DXXRAA',  #'Right Arm Area (cm^2)',
        'DXXRABMC',  #'Right Arm BMC (g)',
        'DXXRABMD',  #'Right Arm BMD (g/cm^2)',
        'DXDRALE',  #'Right Arm Lean excl BMC (g)',
        'DXXRALI',  #'Right Arm Lean incl BMC (g)',
        'DXDRATOT',  #'Right Arm Total (g)',
        'DXXRLA',  #'Right Leg Area (cm^2)',
        'DXXRLBMC',  #'Right Leg BMC (g)',
        'DXXRLBMD',  #'Right Leg BMD(g/cm^2)',
        'DXDRLLE',  #'Right Leg Lean excl BMC (g)',
        'DXXRLLI',  #'Right Leg Lean incl BMC (g)',
        'DXDRLTOT',  #'Right Leg Total (g)',
        'DXXLRA',  #'Left Ribs Area (cm^2)',
        'DXXLRBMC',  #'Left Ribs BMC (g)',
        'DXXLRBMD',  #'Left Ribs BMD (g/cm^2)',
        'DXXRRA',  #'Right Ribs Area (cm^2)',
        'DXXRRBMC',  #'Right Ribs BMC (g)',
        'DXXRRBMD',  #'Right Ribs BMD (g/cm^2)',
        'DXXTSA',  #'Thoracic Spine Area (cm^2)',
        'DXXTSBMC',  #'Thoracic Spine BMC (g)',
        'DXXTSBMD',  #'Thoracic Spine BMD (g/cm^2)',
        'DXXLSA',  #'Lumbar Spine Area (cm^2)',
        'DXXLSBMC',  #'Lumbar Spine BMC (g)',
        'DXXLSBMD',  #'Lumbar Spine BMD (g/cm^2)',
        'DXXPEA',  #'Pelvis Area (cm^2)',
        'DXXPEBMC',  #'Pelvis BMC (g)',
        'DXXPEBMD',  #'Pelvis BMD (g/cm^2)',
        'DXDTRA',  #'Trunk Bone area (cm^2)',
        'DXDTRBMC',  #'Trunk BMC (g)',
        'DXDTRBMD',  #'Trunk Bone BMD (g/cm^2)',
        'DXXTRFAT',  #'Trunk Fat (g)',
        'DXDTRLE',  #'Trunk Lean excl BMC (g)',
        'DXXTRLI',  #'Trunk Lean incl BMC (g)',
        'DXDTRTOT',  #'Trunk Total (g)',
        'DXDSTA',  #'Subtotal Area (cm^2)',
        'DXDSTBMC',  #'Subtotal BMC (g)',
        'DXDSTBMD',  #'Subtotal BMD (g/cm^2)',
        'DXDSTFAT',  #'Subtotal Fat (g)',
        'DXDSTLE',  #'Subtotal Lean excl BMC (g)',
        'DXDSTLI',  #'Subtotal Lean incl BMC (g)',
        'DXDSTTOT',  #'Subtotal (Total excl Head) (g)',
        'DXDSTPF',  #'Subtotal Percent Fat',
        'DXDTOA',  #'Total Area (cm^2)',
        'DXDTOLE',  #'Total Lean excl BMC (g)',
        'DXDTOLI',  #'Total Lean incl BMC (g)',
        'DXDTOTOT',  #'Total Lean+Fat (g)',
        'DXXANFM',  #'Android fat mass',
        'DXXANLM',  #'Android lean mass',
        'DXXANTOM',  #'Android total mass',
        'DXXGYFM',  #'Gynoid fat mass',
        'DXXGYLM',  #'Gynoid lean mass',
        'DXXGYTOM',  #'Gynoid total mass',
        'DXXAGRAT',  #'Android to Gynoid ratio',
        'DXXAPFAT',  #'Android percent fat',
        'DXXGPFAT',  #'Gynoid percent fat',
        'RIDAGEYR',  #'AGE (y)',
        'RIDRETH1',  #'ETHNICITY',
        'SMD057',  # cigarettes smoked per day when quit
        'SMQ020',  # 'smoked at least 100 cigs in life',
        'SMQ040',  # 'now smokes'
        'SMD650',  # avg number of cigarettes mosked in past 30 days
        'RIAGENDR'  #'GENDER (2 female, 1 male)',
    ]

    cfg.NAMES_TO_IMPUTE = [
        'BMXWT',  #'WEIGHT (Kg)',
        'BMXHT',  #'HEIGHT (cm)',
        'BMXTHICR',  #'THIGH CIRC (cm)',
        'BMXWAIST',  #'WAIST CIRC (cm)',
        'BMXHIP',  #'HIP CIRC (cm)',
        'BMXARMC',  #"ARM CIRC (cm)",
        'BMXLEG',  #'Upper leg length (cm)',
        'BMXCALF',  #'Maximum calc circumference (cm)',
        'BMXRECUM',  #'Recumbent Length (cm)',
        'BMXSAD1',  #'Sagittal Abdominal Diameter 1st (cm)',
        'BMXSAD2',  #'Sagittal Abdominal Diameter 2nd (cm)',
        'BMXSAD3',  #'Sagittal Abdominal Diameter 3rd (cm)',
        'BMXSAD4',  #'Sagittal Abdominal Diameter 4th (cm)',
        'BMDAVSAD',  #'Average Sagittal Abdominal Diameter (cm)',
        'BMXBMI',  #'BMI (Kg/m2)',
        'DXDTOPF',  #'PERC BODY FAT (%)',
        'DXDTOFAT',  #'BODY FAT MASS (g)',
        'DXDTRPF',  #'PERC TRUNK FAT (%)',
        'DXXRLFAT',  #'RIGHT LEG FAT (g)',
        'DXXRAFAT',  #'RIGHT ARM FAT (g)',
        'DXXLLFAT',  #'LEFT LEG FAT (g)',
        'DXXLAFAT',  #'LEFT ARM FAT (g)',
        'DXDLAPF',  #'PERC LEFT ARM (%)',
        'DXDRAPF',  #'PERC RIGHT ARM (%)',
        'DXDLLPF',  #'PERC LEFT LEG (%)',
        'DXDRLPF',  #'PERC RIGHT LEG (%)',
        'DXXLAA',  #'Left Arm Area (cm^2)',
        'DXDLALE',  #'Left Arm Lean excl BMC (g)',
        'DXDLATOT',  #'Left Arm Total (g)',
        'DXXLLA',  #'Left Leg Area (cm^2)',
        'DXDLLLE',  #'Left Leg Lean excl BMC (g)',
        'DXDLLTOT',  #'Left Leg Total (g)',
        'DXXRAA',  #'Right Arm Area (cm^2)',
        'DXDRALE',  #'Right Arm Lean excl BMC (g)',
        'DXDRATOT',  #'Right Arm Total (g)',
        'DXXRLA',  #'Right Leg Area (cm^2)',
        'DXDRLLE',  #'Right Leg Lean excl BMC (g)',
        'DXDRLTOT',  #'Right Leg Total (g)',
        'DXXLRA',  #'Left Ribs Area (cm^2)',
        'DXXRRA',  #'Right Ribs Area (cm^2)',
        'DXXTRFAT',  #'Trunk Fat (g)',
        'DXDTRLE',  #'Trunk Lean excl BMC (g)',
        'DXDTRTOT',  #'Trunk Total (g)',
        'DXDTOA',  #'Total Area (cm^2)',
        'DXDTOLE',  #'Total Lean excl BMC (g)',
        'DXDTOTOT',  #'Total Lean+Fat (g)',
        'DXXANFM',  #'Android fat mass',
        'DXXANLM',  #'Android lean mass',
        'DXXANTOM',  #'Android total mass',
        'DXXGYFM',  #'Gynoid fat mass',
        'DXXGYLM',  #'Gynoid lean mass',
        'DXXGYTOM',  #'Gynoid total mass',
        'DXXAGRAT',  #'Android to Gynoid ratio',
        'DXXAPFAT',  #'Android percent fat',
        'DXXGPFAT'
    ]

    # important not to set this too large
    cfg.NN_LEARNING_RATE_INIT = 0.005

    nn_acts = ['swish']
    feat_widths = [45]
    num_parts = 5

    """
    avg_min_vals = []
    for nn_act in nn_acts:
        cfg.NN_ACTIVATION = nn_act
        tmp = []
        for feat_width in feat_widths:
            cfg.WIDTH_OF_IMPUTE_FEATURE_REPRESENTATION = int(np.round_(feat_width))
            val = 0
            for part in range(1, num_parts+1):
                cfg.PARTITION_VERSION = part
                val_this = impute_values()
                val += val_this
            val = val / num_parts
            tmp.append(val)
        avg_min_vals.append(tmp)
    avg_min_vals = np.array(avg_min_vals)

    print('')
    for row, nn_act in enumerate(nn_acts):
        for col, feat_width in enumerate(feat_widths):
            print('{}, {}: {}'.format(nn_act, int(np.round_(feat_width)), avg_min_vals[row, col]))
    """

    aggregate_csv(num_parts)


