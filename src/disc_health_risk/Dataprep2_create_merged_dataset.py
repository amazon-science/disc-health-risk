# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import pickle as pkl
import numpy as np

from DiscRisk_constants import dir_NHANES_dataset, dir_merged_datasets, year_slot_names, data_collections_names

# directory for the merged csv files
dir_merged_datasets = '%s/' % dir_merged_datasets      # output directory
if not os.path.exists(dir_merged_datasets):     # creating output directory
    os.mkdir(dir_merged_datasets)

# main loop
num_year_slots = len(year_slot_names)
num_tot_valid_participants = 0
for year_slot_id in range(num_year_slots):

    year_slot_name = year_slot_names[year_slot_id]
    print('\nWorking on merged file for year slot %s (%s)' % (year_slot_name, dir_merged_datasets))

    # reading all input collection data
    data_all_collections = []
    for coll_tag in data_collections_names:

        dir_collection = r'%s/%s_Data' % (dir_NHANES_dataset, coll_tag)  # each individual collection data directory

        file_name_csv = dir_collection + r'/%s_%s.csv' % (coll_tag, year_slot_name) # name of input csv file for collection

        if os.path.exists(file_name_csv):       # check file exists
            print('\treading %s' % file_name_csv)
            data_collection = np.genfromtxt(file_name_csv, delimiter=',', names=True, case_sensitive=True) # reaad data as np structure
        else:
            print('\t*** WARNING *** - File %s does not exist. Adding empty collection.' % file_name_csv)
            data_collection = []    # appending empty collection

        data_all_collections.append(data_collection)    # appending to main data list for all collections

    # getting the lead data collection and the list of participant SEQN indeces
    data_DEM = data_all_collections[0]
    col_DEM_SEQN = data_DEM['SEQN']

    # building the re-sorted data structures while matching individual participants
    data_all_collections_sorted = []
    data_DEM_sorted = data_DEM
    data_all_collections_sorted.append(data_DEM_sorted)

    num_DEM_rows = len(col_DEM_SEQN)
    for coll_id in range(1, len(data_collections_names)):

        # print('\r\t%s %1.1f%%' % (data_collections_names[coll_id], 100 * coll_id / (len(data_collections_names) - 1)), end='')

        data_coll = data_all_collections[coll_id]

        if len(data_coll)==0:   # empty collection
            data_all_collections_sorted.append([])  # append empty collection
            continue

        SEQN_coll = data_coll['SEQN']
        names_coll = data_coll.dtype.names
        data_coll_sorted = []
        for idx_row_DEM in range(num_DEM_rows):

            seqn_DEM = col_DEM_SEQN[idx_row_DEM]      # identifier of current participant in DXX

            npw = np.where(SEQN_coll == seqn_DEM)[0]
            if len(npw)==1:
                row = data_coll[npw.item()]      # corresponding position in the file
            elif len(npw) > 1:
                row = data_coll[npw[0].item()]   # this deals with possible multiple imputations by taking only the first one
            else:
                row = -1*np.ones(len(names_coll))
                row[0] = seqn_DEM                # writing proper SEQN value rather than -1
            data_coll_sorted.append(row)         # appending row to sorted list for collection

        data_all_collections_sorted.append(data_coll_sorted)

    num_tot_valid_participants += num_DEM_rows

    print('\n\tnum participants in year slot %d (tot %d)' % (num_DEM_rows,num_tot_valid_participants))

    # region saving the unified, output csv file
    out_file_name = r'%s/NHANES_merged_%s.csv' %(dir_merged_datasets, year_slot_name)

    print('\tsaving output file %s' %out_file_name)

    fid = open(out_file_name, 'w')

    # writing field names for all collections
    for coll_id in range(len(data_collections_names)):
        data_coll = data_all_collections[coll_id]

        if len(data_coll)==0:   # empty collection - skip
            continue

        names_coll = data_coll.dtype.names
        for j in range(len(names_coll)):
            fid.write('%s,' % names_coll[j])
    fid.write('\n')

    # writing numerical fields from sorted collections into output csv file
    for row_id in range(len(data_DEM_sorted)):
        for coll_id in range(len(data_collections_names)):
            data_coll_sorted = data_all_collections_sorted[coll_id]

            if len(data_coll_sorted) == 0:  # empty collection - skip
                continue

            row_coll_sorted = data_coll_sorted[row_id]
            for j in range(len(row_coll_sorted)):
                fid.write('%s,' % row_coll_sorted[j])
        fid.write('\n')
    fid.close()

    # region saving the unified, output pickle file - todo - This can be done more efficiently/cleanly but ok for now
    print('\tre-loading csv file as np')
    data_csv_np = np.genfromtxt(out_file_name, delimiter=',', names=True, case_sensitive=True)

    out_file_name_pickle = r'%s/NHANES_merged_%s.pickle' %(dir_merged_datasets, year_slot_name)
    print('\tsaving output file %s' %out_file_name_pickle)
    fid = open(out_file_name_pickle, 'wb')
    pkl.dump(data_csv_np, fid)
    fid.close()
