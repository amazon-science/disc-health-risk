# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from yacs.config import CfgNode as CN


_C = CN()
_C.BIOM_MODEL_IDX = 0
_C.GENDER_ID = -1                    # 1=male, 2=female, -1=both

_C.NN_ENSEMBLE_SIZE = 16             # 10 number of nets to train in ensemble
_C.NN_PERCTG_IN_BAG = 0.5           # 0.5 percentage of training data going into each random bag
_C.NN_OPTIM = 'adam'                # ADAM

"""
_C.NN_LEARNING_RATE_INIT = 0.05  --> val 76.4 acc, 80.3 RDS 
_C.NN_LEARNING_RATE_INIT = 0.06  --> val 75.9 acc, 79.1 RDS
_C.NN_LEARNING_RATE_INIT = 0.07  --> val 75.8 acc, 79.0 RDS
_C.NN_LEARNING_RATE_INIT = 0.08  --> val 75.5 acc, 76.5 RDS
_C.NN_LEARNING_RATE_INIT = 0.09  --> val 76.1 acc, 76.7 RDS
_C.NN_LEARNING_RATE_INIT = 0.10  --> val 76.4 acc, 78.7 RDS
_C.NN_LEARNING_RATE_INIT = 0.11  --> val 75.1 acc, 81.8 RDS
_C.NN_LEARNING_RATE_INIT = 0.12  --> val 75.4 acc, 76.7 RDS
_C.NN_LEARNING_RATE_INIT = 0.13  --> val 75.2 acc, 79.9 RDS
_C.NN_LEARNING_RATE_INIT = 0.14  --> val 76.0 acc, 78.4 RDS
_C.NN_LEARNING_RATE_INIT = 0.15  --> val 75.9 acc, 79.5 RDS
"""
_C.NN_LEARNING_RATE_INIT = 0.01
_C.NN_WEIGHT_L1_REG = 0.01

_C.NN_OPTIMIZER_MOMENTUM = 0.9      # 0.9
_C.NN_BATCH_NORM_MOMENTUM = 0.9     # 0.9 - internally this is converted to PyTorch BachNorm's momentum convention,
                                    # which is one minus what is expected for optimizer momentum
_C.NN_ADD_BATCH_NORM = True        # Don't really need this if inputs are normalized
_C.NN_MAX_NUM_TRAIN_ITERS = 20000  # 1000
_C.NN_LR_REDUCTION_STEP = 1      # 300
_C.NN_ACC_COMPUTE_STEP = 10         # 10 - visualize every N training iterations

_C.SAVE_BEST_TEACC = False           # True - saving best NN model for highest test accuracy
_C.PLOT_TRAIN_PROGRESS = False      # False
_C.SAVE_TRAIN_PROGRESS = False      # False

_C.REMOVE_PREGNANT = True           # True
_C.REMOVE_SMOKERS = False           # False

# only accept examples from the dataset if at least this subset of values is present; only relevent if IMPUTE_MISSING is True
_C.MINIMUM_REQUIRED_BIOMS = ['BMXWT', 'BMXHT', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1']
_C.NAMES_BIOMS = ['RIDAGEYR', 'BMXWAIST', 'BMXTHICR', 'BMXHIP', 'DXDTOPF', 'BMXWT', 'BMXHT', 'RIAGENDR', 'RIDRETH1_1', 'RIDRETH1_2', 'RIDRETH1_3', 'RIDRETH1_4', 'RIDRETH1_5']
_C.NAMES_CONDS = ['BPQ020', 'MCQ160A', 'DIQ010', 'MCQ160C', 'MCQ220', 'MCQ160B', 'MCQ160D', 'MCQ160E', 'MCQ160F']  # BPQ020 MCQ160A DIQ010 MCQ160C MCQ220 - chosen based on max num responses
_C.NAMES_TO_IMPUTE = []

# These denote architecture layers after the input layer, and before the final layer.
# The input size will be calculated based on the length of NAMES_BIOMS.
# The output size is always 2.
# NN_architecture = [4]                     # SMALL ARCH - two-class output
# NN_architecture = [4, 4, 4]               # LARGER ARCH - two-class output
_C.NN_ARCHITECTURE = [2]  #[3, 3]      #       # two-class output

_C.FIG_W = 18
_C.FIG_H = 5.5

_C.AGE_MIN = 18
_C.AGE_MAX = 120

_C.MIN_VAL_COND_FOR_POS = 1
_C.MAX_VAL_COND_FOR_POS = 1

_C.USE_UNDIAGNOSED_CONDITIONS = True
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
# Arthritis (MCQ160A): doctor ever told you had arthritis
#     MCQ160n: doctor ever told you that you had gout
# Diabetes (DIQ010): fasting glucose >= 7.0 mmol/L (>=126 mg/dL) and HbA1c >= ≥6.5%
#                    LBDGLUSI is mmol/L
#                    LBXGLU is mg/dL
#                    LBXGH is HbA1c is glycohemoglobin
# Coronary heart disease (MCQ160C): a history of heart failure, coronary heart disease, angina and/or angina pectoris, or myocardial infarction -- might have already been taken care of in NHANES
#                                   MCQ160e: ever told you had heart attack
#                                   MCQ160d: ever told you had angina/angina pectoris
#                                   MCQ160b: ever told had congestive heart failure (the relationship is opposite: CHF from CHD)
#                                   MCQ160f: ever told you had a stroke
# Cancer (MCQ220): no measurements to detect undiagnosed

# BPQ080: doctor told you have high cholesterol level
#     According to https://www.cdc.gov/cholesterol/index.htm: >= 200 mg/dL is high cholesterol


_C.FRAC_VAL = 0.2
_C.FRAC_TEST = 0.4
_C.PARTITION_VERSION = 1

_C.NN_LOSS = 'CrossEntropyLoss'
_C.NN_ACTIVATION = 'sigmoid'

_C.APPLY_SAMPLE_WEIGHTS = True
_C.APPLY_INVERSE_CLASS_FREQUENCY_WEIGHTS = True

_C.IMPUTE_MISSING = True
_C.NN_STEPS_TO_IMPUTE_FAKE = 100

_C.CSV_FNAME_TAG = 'results'
_C.DIR_RESULTS = 'results'

_C.YEARS_TO_DEATH_THRESH = 100

#### Inputation
_C.DEPTH_TO_IMPUTE_FEATURE_REPRESENTATION = 2
_C.WIDTH_OF_IMPUTE_FEATURE_REPRESENTATION = 25
