# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import glob, os
import pandas as pd
from DiscRisk_constants import dir_NHANES_dataset

dirs_collections = glob.glob(dir_NHANES_dataset + r'/*Data')
for dir_collection in dirs_collections:
    if not os.path.isdir(dir_collection):
        continue

    print('working on collection %s -----------------------' % dir_collection)

    files_xpt = glob.glob(dir_collection + r'/*.XPT')
    for file_name_xpt in files_xpt:
        file_name_csv = file_name_xpt.replace('.XPT', '.csv')

        print('\t%s...' %file_name_csv)

        data_xpt = pd.read_sas(file_name_xpt)
        data_xpt.to_csv(file_name_csv, index=False, na_rep='-1')