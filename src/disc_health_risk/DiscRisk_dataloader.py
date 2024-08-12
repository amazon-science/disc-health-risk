# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import csv

import numpy as np
import numpy.lib.recfunctions as rfn

import DiscRisk_constants as CONSTS
from DiscRisk_utilities import loadRawDataFromPickleFile, checkFieldsExistInData, appendToData, unnorm2norm, \
    genderStringFromId, has_hypertension, has_diabetes, has_arthritis, has_coronary_heart_disease_or_related


class DataLoader:

    def __init__(self, fname_imputed=None):

        #region load all labeled  data
        self.data = None
        self.stats = None
        self.mu = None
        self.std = None
        self.valid_mask = None

        self.imputed_dict = None
        if fname_imputed is not None:
            self.imputed_dict = dict()
            with open(fname_imputed, 'r') as fid:
                reader = csv.DictReader(fid)
                for row in reader:
                    id = int(float(row['ID']))
                    self.imputed_dict[id] = dict()
                    for key_this in row.keys():
                        if key_this == id:
                            continue
                        self.imputed_dict[id][key_this] = float(row[key_this])


    def print_stats(self):

        if self.stats is None:
            return

        print('Stats:')
        for desc in self.stats.keys():
            num_all = self.stats[desc]['all']
            num_pos = self.stats[desc]['pos']
            num_neg = self.stats[desc]['neg']
            prevalence = self.stats[desc]['prevalence']
            print('  {}: {} total, {} pos, {} neg, {}% weighted prevalence'.format(desc, num_all, num_pos, num_neg, prevalence))
        print('')

    def compute_stats(self):

        if self.data is None:
            return

        descs = ['Global', 'Train', 'Test', 'Val']
        is_train = self.data['is_train']
        is_test = self.data['is_test']
        is_val = self.data['is_val']
        is_global = np.logical_or(np.logical_or(is_train, is_test), is_val)

        self.stats = dict()
        for desc in descs:
            if desc == 'Global':
                is_this = is_global
            elif desc == 'Train':
                is_this = is_train
            elif desc == 'Test':
                is_this = is_test
            elif desc == 'Val':
                is_this = is_val
            num_all = np.count_nonzero(is_this)
            num_pos = np.count_nonzero(self.data['CON_gt'][is_this])
            num_neg = num_all - num_pos
            w = self.data['SAMPLE_WEIGHT'][is_this]
            total_w = np.sum(w)

            weighted_prev = 100 * np.sum(np.multiply(self.data['CON_gt'][is_this], w)) / np.maximum(0.000001, total_w)


            self.stats[desc] = dict()
            self.stats[desc]['all'] = num_all
            self.stats[desc]['pos'] = num_pos
            self.stats[desc]['neg'] = num_neg
            self.stats[desc]['prevalence'] = weighted_prev  #100 * num_pos / np.maximum(0.000001, num_all)

    def mask_data(self, mask):

        mask_len = mask.size
        for key in self.data.keys():
            if mask_len == self.data[key].shape[0]:
                self.data[key] = self.data[key][mask]

    def filter_all_data(self, cfg, include_and_flag_missing_bioms=False):

        gender_str = genderStringFromId(cfg.GENDER_ID)

        print('\n%d \t original dataset' % len(self.data['GEN']))

        #region filter data
        # filter data by age
        mask_age = np.logical_and(self.data['AGE'] >= cfg.AGE_MIN,
                                  self.data['AGE'] <= cfg.AGE_MAX)
        self.mask_data(mask_age)
        print('\n%d \t after age filtering' % len(self.data['GEN']))

        # filter data by gender
        if cfg.GENDER_ID > 0:
            mask_gen = self.data['GEN'] == cfg.GENDER_ID   # 1=male, 2=female
            self.mask_data(mask_gen)
            print('%d \t %s' % (self.data['BIOMs_gt'].shape[0], gender_str))

        # remove smokers
        if cfg.REMOVE_SMOKERS:
            mask_non_smokers = self.data['SMQ'] == 2                  # smoked less than 100 cigarettes in lifetime
            self.mask_data(mask_non_smokers)
            print('%d \t %s not smoking' % (self.data['BIOMs_gt'].shape[0], gender_str))

        # remove invalid conditions
        masks_valid_CONDs_all = self.data['CONDs_gt'][:, 0] != -1
        num_conds = self.data['CONDs_gt'].shape[1]
        for cond_id in range(1, num_conds):
            mask_valid_id = self.data['CONDs_gt'][:, cond_id] != -1
            masks_valid_CONDs_all = np.logical_and(masks_valid_CONDs_all, mask_valid_id)
        self.mask_data(masks_valid_CONDs_all)
        print('%d \t %s valid condition %s' % (self.data['BIOMs_gt'].shape[0], gender_str, cfg.NAMES_CONDS))

        # remove invalid biomarkers
        if not include_and_flag_missing_bioms:
            masks_valid_BIOMs_all = self.data['BIOMs_gt'][:, 0] != -1
            num_bioms = self.data['BIOMs_gt'].shape[1]
            for biom_id in range(1, num_bioms):
                mask_valid_id = self.data['BIOMs_gt'][:, biom_id] != -1
                masks_valid_BIOMs_all = np.logical_and(masks_valid_BIOMs_all, mask_valid_id)
            self.mask_data(masks_valid_BIOMs_all)
        print('%d \t %s valid biomarkers\n' % (self.data['BIOMs_gt'].shape[0], gender_str))

        # remove pregnant women
        if cfg.REMOVE_PREGNANT:
            mask_pregnant = self.data['PRG'] == 1
            mask_pregnant = np.logical_and(mask_pregnant, self.data['GEN'] == 2)
            mask_not_pregnant = np.logical_not(mask_pregnant)
            self.mask_data(mask_not_pregnant)
            print('%d \t %s not pregnant' % (self.data['BIOMs_gt'].shape[0], gender_str))

    def compute_data_partitions(self, cfg):

        num_examples = self.data['BIOMs_gt'].shape[0]
        is_val = np.zeros(num_examples, dtype=bool)
        is_test = np.zeros(num_examples, dtype=bool)
        is_train = np.zeros(num_examples, dtype=bool)

        """
        fraction_val = cfg.FRAC_VAL
        if fraction_val < 0.0 or fraction_val >= 1.0:
            raise Exception("Inside DataLoader::partition_data. VAL_FRAC ({}) must be in [0.0, 1.0)".format(fraction_val))
        fraction_tst = cfg.FRAC_TEST
        if fraction_val < 0.0 or fraction_val >= 1.0:
            raise Exception("Inside DataLoader::partition_data. TEST_FRAC ({}) must be in [0.0, 1.0)".format(fraction_tst))

        fraction_val = np.round_(fraction_val, decimals=2)
        fraction_tst = np.round_(fraction_tst, decimals=2)

        thresh_val = fraction_val * 100
        thresh_tst = (fraction_tst + fraction_val) * 100

        for idx in range(num_examples):
            id = self.data['ID'][idx]
            id_100 = id % 100
            if id_100 <= thresh_val:
                is_val[idx] = True
            elif id_100 <= thresh_tst:
                is_test[idx] = True
            else:
                is_train[idx] = True
        """

        if cfg.PARTITION_VERSION == 1:
            range_val = list(range(0, 21))
            range_tst = list(range(21, 61))
        elif cfg.PARTITION_VERSION == 2:
            range_val = list(range(21, 41))
            range_tst = list(range(41, 81))
        elif cfg.PARTITION_VERSION == 3:
            range_val = list(range(41, 61))
            range_tst = list(range(61, 101))
        elif cfg.PARTITION_VERSION == 4:
            range_val = list(range(61, 81))
            range_tst = list(range(81, 101))
            range_tst.extend(list(range(0, 20)))
        elif cfg.PARTITION_VERSION == 5:
            range_val = list(range(81, 101))
            range_tst = list(range(0, 40))
        else:
            raise Exception('Invalid cfg.PARTITION_VERSION')

        for idx in range(num_examples):
            id_this = self.data['ID'][idx]
            id_100 = id_this % 100
            if id_100 in range_val:
                is_val[idx] = True
            elif id_100 in range_tst:
                is_test[idx] = True
            else:
                is_train[idx] = True

        empirical_frac_val = float(np.count_nonzero(is_val)) / num_examples
        empirical_frac_tst = float(np.count_nonzero(is_test)) / num_examples
        empirical_frac_trn = float(np.count_nonzero(is_train)) / num_examples

        print('Data partition:')
        print('  Train target: {}%, Train actual: {}%'.format(100*(1.0-cfg.FRAC_TEST-cfg.FRAC_VAL), 100*empirical_frac_trn))
        print('  Test target: {}%, test actual: {}%'.format(100*cfg.FRAC_TEST, 100*empirical_frac_tst))
        print('  Val target: {}%, val actual: {}%'.format(100*cfg.FRAC_VAL, 100*empirical_frac_val))

        self.data['is_val'] = is_val
        self.data['is_test'] = is_test
        self.data['is_train'] = is_train

        if np.count_nonzero(is_val) <= 10:
            print('WARNING: Validation set is very small ({} examples)'.format(np.count_nonzero(is_val)))
        if np.count_nonzero(is_test) <= 10:
            print('WARNING: Testing set is very small ({} examples)'.format(np.count_nonzero(is_test)))
        if np.count_nonzero(is_train) <= 10:
            print('WARNING: Training set is very small ({} examples)'.format(np.count_nonzero(is_test)))

        print('')

    def get_normalized_bioms(self, include_and_flag_missing_bioms=False):

        self.valid_mask = self.data['BIOMs_gt'] >= 0

        # Compute mu and std if needed
        if self.mu is None or self.std is None:
            data_train = self.data['BIOMs_gt'][self.data['is_train']].copy()
            valid_mask_train = self.valid_mask[self.data['is_train']]

            valid_cols = np.ones(len(self.names_bioms), dtype=bool)
            for col_idx, biom in enumerate(self.names_bioms):
                if biom == 'RIAGENDR':
                    valid_cols[col_idx] = False
                elif biom == 'RIDRETH1_1':
                    valid_cols[col_idx] = False
                elif biom == 'RIDRETH1_2':
                    valid_cols[col_idx] = False
                elif biom == 'RIDRETH1_3':
                    valid_cols[col_idx] = False
                elif biom == 'RIDRETH1_4':
                    valid_cols[col_idx] = False
                elif biom == 'RIDRETH1_5':
                    valid_cols[col_idx] = False

            _, self.mu, self.std = unnorm2norm(data_train, valid_mask=valid_mask_train, valid_cols=valid_cols)
            """
            for c in range(data_train.shape[1]):
                unique_vals = np.unique(data_train[:, c])
                if unique_vals.size == 2 and unique_vals[0] == 0 and unique_vals[1] == 1:
                    # apply no normalization to this dimension
                    self.mu[0, c] = 0.0
                    self.std[0, c] = 1.0
            """

        # Normalize
        data = self.data['BIOMs_gt'].copy()
        data, _, _ = unnorm2norm(data, mu=self.mu, std=self.std, valid_mask=self.valid_mask)

        if include_and_flag_missing_bioms:
            invalid = np.logical_not(self.valid_mask)
            num_rows = data.shape[0]
            mu_array = np.repeat(self.mu, num_rows, axis=0)
            data[invalid] = mu_array[invalid]
            data = np.concatenate((data, self.valid_mask.astype(np.float32)), axis=1)

        return data

    def compute_sample_weights(self, data_raw_yslot):

        wtmec2yr = data_raw_yslot['WTMEC2YR']
        weight_version_a = wtmec2yr/10
        mec20yr = weight_version_a
        try:
            sddsrvyr = data_raw_yslot['SDDSRVYR']
            wtmec4yr = data_raw_yslot['WTMEC4YR']
            weight_version_b = 2*wtmec4yr/10
            mec20yr[sddsrvyr == 1.0] = weight_version_b[sddsrvyr == 1.0]
            mec20yr[sddsrvyr == 2.0] = weight_version_b[sddsrvyr == 2.0]
        except Exception as detail:
            pass  # sometimes 'WTMEC4Y' is not available
        return mec20yr


    def replace_missing_with_imputed(self, cfg, data_raw_yslot):

        if self.imputed_dict is None:
            raise Exception('replace_missing_with_imputed was called, but self.imputed_dict is None')

        ids_this = data_raw_yslot[CONSTS.name_ID].astype(int)
        num_examples = ids_this.size

        # Fill missing columns/fields with all -1 (indicating missing) values
        for name in cfg.NAMES_BIOMS:
            if 'RIDRETH1' in name:
                name = 'RIDRETH1'
            if name not in data_raw_yslot.dtype.names:
                data_raw_yslot = rfn.append_fields(data_raw_yslot, [name], [-np.ones(ids_this.size)])
        for cond in cfg.NAMES_CONDS:
            if cond not in data_raw_yslot.dtype.names:
                data_raw_yslot = rfn.append_fields(data_raw_yslot, [cond], [-np.ones(ids_this.size)])

        num_replaced = 0
        for name in cfg.NAMES_BIOMS:
            # don't try to impute ethnicity
            if 'RIDRETH1' in name:
                continue
            # don't try to impute smoking
            if name.startswith('SMD') or name.startswith('SMQ'):
                continue

            missing_mask = data_raw_yslot[name] == -1
            for i in range(missing_mask.size):
                if missing_mask[i]:
                    missing_id = int(ids_this[i])
                    try:
                        data_raw_yslot[name][i] = self.imputed_dict[missing_id][name]
                        num_replaced += 1
                    except KeyError:
                        pass

        """
        for row_idx, id in enumerate(list(ids_this)):
            for name in cfg.NAMES_BIOMS:
                if data_raw_yslot[name][row_idx] == -1 and id in self.imputed_dict.keys() and name in self.imputed_dict[id].keys():
                    data_raw_yslot[name][row_idx] = self.imputed_dict[id][name]
            for cond in cfg.NAMES_CONDS:
                if data_raw_yslot[cond][row_idx] == -1 and id in self.imputed_dict.keys() and cond in self.imputed_dict[id].keys():
                    data_raw_yslot[cond][row_idx] = self.imputed_dict[id][cond]
            name = CONSTS.name_AGE
            if data_raw_yslot[name][row_idx] == -1 and id in self.imputed_dict.keys() and name in self.imputed_dict[id].keys():
                data_raw_yslot[name][row_idx] = self.imputed_dict[id][name]
            name = CONSTS.name_GEN
            if data_raw_yslot[name][row_idx] == -1 and id in self.imputed_dict.keys() and name in self.imputed_dict[id].keys():
                data_raw_yslot[name][row_idx] = self.imputed_dict[id][name]
            name = CONSTS.name_PRG
            if data_raw_yslot[name][row_idx] == -1 and id in self.imputed_dict.keys() and name in self.imputed_dict[id].keys():
                data_raw_yslot[name][row_idx] = self.imputed_dict[id][name]
        """

        return data_raw_yslot


    def filter_minimum_required_bioms(self, data_raw_yslot, minimum_required_bioms):

        ids_this = data_raw_yslot[CONSTS.name_ID].astype(int)

        ok_mask = np.ones(len(ids_this), dtype=bool)
        for name in minimum_required_bioms:

            if 'RIDRETH1' in name:
                # Could be RIDRETH1_1, RIDRETH1_2, RIDRETH1_3, RIDRETH1_4, RIDRETH1_5
                name = 'RIDRETH1'

            if name not in data_raw_yslot.dtype.names:
                ok_mask[:] = False
                break
            ok_mask_this = data_raw_yslot[name] != -1
            ok_mask = np.logical_and(ok_mask, ok_mask_this)

        if np.count_nonzero(ok_mask) < ok_mask.size:
            remove_rows = np.nonzero(np.logical_not(ok_mask))[0]
            data_raw_yslot = np.delete(data_raw_yslot, remove_rows, 0)

        return data_raw_yslot


    def load_nhanes(self, cfg, complete_examples_only=True, impute_missing=False, include_and_flag_missing_bioms=False):

        self.data = dict()
        self.names_bioms = cfg.NAMES_BIOMS
        self.names_conds = cfg.NAMES_CONDS
        num_bioms = len(cfg.NAMES_BIOMS)
        num_conds = len(cfg.NAMES_CONDS)
        self.data['BIOMs_gt'] = np.zeros((0, num_bioms))         # all gt-labelled biomarker data points
        self.data['CONDs_gt'] = np.zeros((0, num_conds))    # all conditions ground truth (labels)
        self.data['GEN'] = np.zeros(0)    # gender
        self.data['AGE'] = np.zeros(0)    # age
        self.data['PRG'] = np.zeros(0)    # pregnant
        self.data['SMQ'] = np.zeros(0)    # smoker
        self.data['ID'] = np.zeros(0)     # participant ID
        self.data['SAMPLE_WEIGHT'] = np.zeros(0)   # the NHANES sample weights

        cond_includes_dead_by = False
        age_at_exam_buckets = [[18, 24],
                               [25, 29],
                               [30, 34],
                               [35, 39],
                               [40, 44],
                               [45, 49],
                               [50, 54],
                               [55, 59],
                               [60, 64],
                               [65, 69],
                               [70, 74],
                               [75, 79],
                               [80, 84]]  # [inclusive, inclusive]

        exam_age_vs_died_by_alive = np.zeros((len(age_at_exam_buckets), ), dtype=int)
        exam_age_vs_died_by_dead = np.zeros((len(age_at_exam_buckets), ), dtype=int)
        for cond in cfg.NAMES_CONDS:
            if 'DEAD_BY' in cond:
                cond_includes_dead_by = True
                break

        # Clear these in case we're reloading
        self.mu = None
        self.std = None
        self.valid_mask = None

        num_year_slots = len(CONSTS.year_slot_names)            # number of NHANES year slots
        for year_slot_id_tmp in range(num_year_slots):

            # load data for year slot from pickle file
            year_slot_name = CONSTS.year_slot_names[year_slot_id_tmp]
            file_name = r'%s/NHANES_merged_%s.pickle' %(CONSTS.dir_merged_datasets, year_slot_name)
            data_raw_yslot = loadRawDataFromPickleFile(file_name)   # mload raw data from file

            variables = ['PAD680']
            codes = [7777, 9997]  # refused, don't know
            for variable in variables:
                if variable in data_raw_yslot.dtype.names:
                    for code in codes:
                        mask = data_raw_yslot[variable] == code
                        count = np.count_nonzero(mask)
                        if count > 0:
                            data_raw_yslot[variable][mask] = -1  # set to missing since the code is not really useful

            variables = ['SMD650', 'SMD057']
            codes = [777, 999]  # refused, don't know
            for variable in variables:
                if variable in data_raw_yslot.dtype.names:
                    for code in codes:
                        mask = data_raw_yslot[variable] == code
                        count = np.count_nonzero(mask)
                        if count > 0:
                            data_raw_yslot[variable][mask] = -1  # set to missing since the code is not really useful
                            #print('{}: {} entries with {} == {}'.format(file_name, count, variable, code))

            variables = ['PAQ710', 'PAQ715']
            codes = [77, 99]  # refused, don't know
            for variable in variables:
                if variable in data_raw_yslot.dtype.names:
                    for code in codes:
                        mask = data_raw_yslot[variable] == code
                        count = np.count_nonzero(mask)
                        if count > 0:
                            data_raw_yslot[variable][mask] = -1  # set to missing since the code is not really useful

            variables = ['SMQ020', 'SMQ040', 'PAQ665', 'PAQ650', 'PAQ620', 'PAQ605']
            codes = [7, 9]
            for variable in variables:
                if variable in data_raw_yslot.dtype.names:
                    for code in codes:
                        mask = data_raw_yslot[variable] == code
                        count = np.count_nonzero(mask)
                        if count > 0:
                            data_raw_yslot[variable][mask] = -1  # set to missing since the code is not really useful
                            #print('{}: {} entries with {} == {}'.format(file_name, count, variable, code))

            data_raw_yslot = self.filter_minimum_required_bioms(data_raw_yslot, cfg.MINIMUM_REQUIRED_BIOMS)
            if data_raw_yslot.shape[0] == 0:
                continue

            if impute_missing:
                data_raw_yslot = self.replace_missing_with_imputed(cfg, data_raw_yslot)

            if complete_examples_only:
                if include_and_flag_missing_bioms:
                    tmp_NAMES_BIOMS = cfg.MINIMUM_REQUIRED_BIOMS
                else:
                    tmp_NAMES_BIOMS = cfg.NAMES_BIOMS

                if cond_includes_dead_by:
                    tmp_NAMES_CONDS = []
                else:
                    tmp_NAMES_CONDS = cfg.NAMES_CONDS

                if not checkFieldsExistInData(data_raw_yslot, tmp_NAMES_BIOMS, tmp_NAMES_CONDS,
                                                                     CONSTS.name_SMQ, cfg.REMOVE_SMOKERS, year_slot_name):
                    continue

            if cond_includes_dead_by:
                mort_file_name = file_name.replace('z_merged_datasets', 'mortality')
                mort_file_name = mort_file_name.replace('.pickle', '_MORT_2019_PUBLIC.csv')
                mort_file_name = mort_file_name.replace('merged_A_', '')
                mort_file_name = mort_file_name.replace('merged_B_', '')
                mort_file_name = mort_file_name.replace('merged_C_', '')
                mort_file_name = mort_file_name.replace('merged_D_', '')
                mort_file_name = mort_file_name.replace('merged_E_', '')
                mort_file_name = mort_file_name.replace('merged_F_', '')
                mort_file_name = mort_file_name.replace('merged_G_', '')
                mort_file_name = mort_file_name.replace('merged_H_', '')
                mort_file_name = mort_file_name.replace('merged_I_', '')
                mort_file_name = mort_file_name.replace('merged_J_', '')
                with open(mort_file_name, 'rt') as csvfid:
                    reader = csv.DictReader(csvfid)
                    for row in reader:
                        permth_exm = float(row['PERMTH_EXM'])
                        seqn_mort = int(row['SEQN'])
                        match_mask = data_raw_yslot['SEQN'].data == seqn_mort
                        if np.count_nonzero(match_mask) == 0:
                            continue
                        row_idx = np.nonzero(match_mask)[0][0]
                        if not int(row['Eligible']):
                            for cond in cfg.NAMES_CONDS:
                                if 'DEAD_BY' in cond:
                                    data_raw_yslot[cond][row_idx] = -1
                        else:
                            # eligible
                            earliest_year_of_exam = int(year_slot_name[2:6])
                            assert(earliest_year_of_exam >= 1999)
                            assert(earliest_year_of_exam <= 2018)
                            elapsed_years = 2019 - earliest_year_of_exam
                            age = data_raw_yslot['RIDAGEYR'][row_idx]  # age at the time of exam
                            age_in_2019 = age + elapsed_years
                            if int(row['AssumedAlive']):
                                for cond in cfg.NAMES_CONDS:
                                    if 'DEAD_BY' in cond:
                                        age_thresh = float(cond[8:])
                                        years_between_thresh_and_age_at_exam = abs(age_thresh - age)
                                        if years_between_thresh_and_age_at_exam > cfg.YEARS_TO_DEATH_THRESH:
                                            # The dead-by age we're looking at cannot be more than YEARS_TO_DEATH_THRESH from the age at exam
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif age >= age_thresh:
                                            # if someone is examined at, say, 70-years-old, then clearly they are not dead by 65
                                            # Omit such examples as they are trivial
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif age_in_2019 < age_thresh:
                                            # We don't know what happened after 2019, whether they died or not
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif (age_in_2019-age) < (age_thresh-age):
                                            # At this point, these cases are ONLY dead, i.e., no information about living
                                            # so omit
                                            data_raw_yslot[cond][row_idx] = -1
                                        else:
                                            data_raw_yslot[cond][row_idx] = 0
                                            for arowidx, age_range in enumerate(age_at_exam_buckets):
                                                if age >= age_range[0] and age <= age_range[1]:
                                                    exam_age_vs_died_by_alive[arowidx] += 1
                            elif int(row['AssumedDeceased']) and permth_exm > 0:
                                age_at_death = age + permth_exm / 12.0
                                for cond in cfg.NAMES_CONDS:
                                    if 'DEAD_BY' in cond:
                                        age_thresh = float(cond[8:])
                                        years_between_thresh_and_age_at_exam = abs(age_thresh - age)
                                        if years_between_thresh_and_age_at_exam > cfg.YEARS_TO_DEATH_THRESH:
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif age >= age_thresh:
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif age_in_2019 < age_thresh:
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif (age_in_2019-age) < (age_thresh-age):
                                            data_raw_yslot[cond][row_idx] = -1
                                        elif age_at_death <= age_thresh:
                                            # died before X in DEAD_BY_X
                                            data_raw_yslot[cond][row_idx] = 1
                                            for arowidx, age_range in enumerate(age_at_exam_buckets):
                                                if age >= age_range[0] and age <= age_range[1]:
                                                    exam_age_vs_died_by_dead[arowidx] += 1
                                        else:
                                            # died after X in DEAD_BY_X
                                            data_raw_yslot[cond][row_idx] = 0
                                            for arowidx, age_range in enumerate(age_at_exam_buckets):
                                                if age >= age_range[0] and age <= age_range[1]:
                                                    exam_age_vs_died_by_alive[arowidx] += 1
                            else:
                                for cond in enumerate(cfg.NAMES_CONDS):
                                    if 'DEAD_BY' in cond:
                                        data_raw_yslot[cond][row_idx] = -1

            if cfg.USE_UNDIAGNOSED_CONDITIONS:
                #print('{}:'.format(year_slot_name))
                field_names = ['BPQ020', 'DIQ010'] #, 'MCQ160C']
                # BPQ020: high blood pressure / hypertension
                # DIQ010: diabetes
                # MCQ160C: coronary heart disease
                # MCQ160A: arthritis
                for field_name in field_names:
                    if field_name in cfg.NAMES_CONDS and field_name in data_raw_yslot.dtype.names:
                        #print('    {} ({})'.format(CONSTS.long_cond_descriptions[field_name], field_name))
                        if field_name == 'BPQ020':
                            positive_diagnosis = has_hypertension(data_raw_yslot)
                        elif field_name == 'DIQ010':
                            positive_diagnosis = has_diabetes(data_raw_yslot)
                        elif field_name == 'MCQ160C':
                            positive_diagnosis = has_coronary_heart_disease_or_related(data_raw_yslot)
                        else:
                            continue
                        count_diagnosed = np.count_nonzero(data_raw_yslot[field_name] == 1)
                        count_undiagnosed = np.count_nonzero(positive_diagnosis == 1)
                        count_diagnosed_and_undiagnosed = np.count_nonzero(np.logical_and(positive_diagnosis == 1, data_raw_yslot[field_name] == 1))
                        #print('        diagnosed: {}'.format(count_diagnosed))
                        #print('        undiagnosed: {}'.format(count_undiagnosed))
                        #print('        diagnosed AND undiagnosed: {}'.format(count_diagnosed_and_undiagnosed))
                        data_raw_yslot[field_name][positive_diagnosis == 1] = 1  # 2 means no
                        count_diagnosed_or_undiagnosed = np.count_nonzero(data_raw_yslot[field_name] == 1)
                        #print('        diagnosed OR undiagnosed: {}'.format(count_diagnosed_or_undiagnosed))
                """
                if 'MCQ220' in cfg.NAMES_CONDS:  # cancer
                    # nothing to use except for diagnosed
                    pass
                """

            # append new data for current year slot into global arrays
            self.data['BIOMs_gt'] = appendToData(data_raw_yslot, self.data['BIOMs_gt'], cfg.NAMES_BIOMS)     # Biomarkers
            self.data['CONDs_gt'] = appendToData(data_raw_yslot, self.data['CONDs_gt'], cfg.NAMES_CONDS)     # Conditions
            self.data['GEN'] = np.concatenate((self.data['GEN'], data_raw_yslot[CONSTS.name_GEN]), axis=0)   # Gender
            self.data['AGE'] = np.concatenate((self.data['AGE'], data_raw_yslot[CONSTS.name_AGE]), axis=0)   # Age
            self.data['PRG'] = np.concatenate((self.data['PRG'], data_raw_yslot[CONSTS.name_PRG]), axis=0)   # Pregnant
            self.data['ID'] = np.concatenate((self.data['ID'], data_raw_yslot[CONSTS.name_ID]), axis=0)      # Participant ID
            sample_weights_this = self.compute_sample_weights(data_raw_yslot)
            self.data['SAMPLE_WEIGHT'] = np.concatenate((self.data['SAMPLE_WEIGHT'], sample_weights_this), axis=0)
            if cfg.REMOVE_SMOKERS:
                self.data['SMQ'] = np.concatenate((self.data['SMQ'], data_raw_yslot[CONSTS.name_SMQ]), axis=0)  # Smoking


        #endregion load all labeled training data

        if complete_examples_only:
            self.filter_all_data(cfg, include_and_flag_missing_bioms=include_and_flag_missing_bioms)

        self.data['SAMPLE_WEIGHT'] = self.data['SAMPLE_WEIGHT'] / np.sum(self.data['SAMPLE_WEIGHT'])

        # map ground truth condition labels to integers
        self.data['idx_cond_positive_as_1'] = np.logical_and(self.data['CONDs_gt'] >= cfg.MIN_VAL_COND_FOR_POS,
                                                             self.data['CONDs_gt'] <= cfg.MAX_VAL_COND_FOR_POS)

        # merge conditions together through an OR operation
        self.data['CON_gt'] = np.any(self.data['idx_cond_positive_as_1'], axis=1).astype('int') # False->0 (negative to condition) True->1 (positive to condition)
        #endregion filter data

        # DEBUG
        for cond in cfg.NAMES_CONDS:
            if 'DEAD_BY' in cond:
                age_thresh = float(cond[8:])
                diff = np.abs(self.data['AGE'] - age_thresh)
                assert(np.alltrue(diff <= cfg.YEARS_TO_DEATH_THRESH))
                print('\nDied by {}:'.format(age_thresh))
                for arowidx, age_range in enumerate(age_at_exam_buckets):
                    print('{}-{}\t{}\t/\t{}'.format(age_range[0], age_range[1], exam_age_vs_died_by_dead[arowidx],
                                                    exam_age_vs_died_by_alive[arowidx]+exam_age_vs_died_by_dead[arowidx]))

        self.compute_data_partitions(cfg)

        for col_idx, biom in enumerate(cfg.NAMES_BIOMS):
            if biom == 'RIAGENDR':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 3
            elif biom == 'RIDRETH1_1':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 1
            elif biom == 'RIDRETH1_2':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 1
            elif biom == 'RIDRETH1_3':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 1
            elif biom == 'RIDRETH1_4':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 1
            elif biom == 'RIDRETH1_5':
                self.data['BIOMs_gt'][:, col_idx] = self.data['BIOMs_gt'][:, col_idx] * 2 - 1


        self.compute_stats()
        self.print_stats()


    def get_data(self):

        if self.data is None:
            raise Exception("Inside get_data(). No data to return. Call load_nhanes() first.")

        return self.data

    def get_stats(self):

        if self.stats is None:
            raise Exception('Inside get_stats(). No stats to return. Call load_nhanes() first.')

        return self.stats
