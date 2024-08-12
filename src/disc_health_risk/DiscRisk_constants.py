# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import enum

##############################################################################################################
# Parameters
##############################################################################################################

dir_NHANES_dataset = r'data/NHANES'
dir_merged_datasets = '%s/z_merged_datasets' % dir_NHANES_dataset      # output directory
data_collections_names = ['DEM', 'ALQ', 'BIX', 'BMX', 'BPQ', 'BPX', 'CBQ', 'CDQ', 'CVX', 'DBQ', 'DIQ', 'DPQ', 'DXX', 'DXXAG', 'GHB', 'GLU', 'HSQ', 'MCQ', 'OGTT', 'PAQ', 'RDQ', 'SLQ', 'SMQ', 'WHQ']   # make sure the first entry is always DEM
year_slot_names = ['A_1999_2000', 'B_2001_2002', 'C_2003_2004', 'D_2005_2006', 'E_2007_2008', 'F_2009_2010', 'G_2011_2012', 'H_2013_2014', 'I_2015_2016', 'J_2017_2018']

name_GEN = 'RIAGENDR' # gender
name_AGE = 'RIDAGEYR' # age
name_PRG = 'RIDEXPRG' # pregnant
name_SMQ = 'SMQ020'   # smoker
name_ID  = 'SEQN'     # participant ID (respondent sequence number)

long_biom_descriptions = dict( # Most biomarkers are defined in https://www.omnicalculator.com/health/
    VOID='',
    BMXWT='WEIGHT (Kg)',
    BMXHT='HEIGHT (cm)',
    BMXHEAD='Head circumference (cm)',
    BMXTHICR='THIGH CIRC (cm)',
    BMXWAIST='WAIST CIRC (cm)',
    BMXHIP='HIP CIRC (cm)',
    BMXARMC="ARM CIRC (cm)",
    BMXLEG='Upper leg length (cm)',
    BMXCALF='Maximum calc circumference (cm)',
    BMXTRI='Triceps skinfold (mm)',
    BMXSUB='Subscapular Skinfold (mm)',
    BMXRECUM='Recumbent Length (cm)',
    BMXSAD1='Sagittal Abdominal Diameter 1st (cm)',
    BMXSAD2='Sagittal Abdominal Diameter 2nd (cm)',
    BMXSAD3='Sagittal Abdominal Diameter 3rd (cm)',
    BMXSAD4='Sagittal Abdominal Diameter 4th (cm)',
    BMDAVSAD='Average Sagittal Abdominal Diameter (cm)',
    BMDSADCM='Sagittal Abdominal Diameter Comment',
    BMI='BMI (Kg/m2)',
    BMXBMI='BMI (Kg/m2)',
    PBF='PERC BODY FAT (%)',
    DXDTOPF='PERC BODY FAT (%)',
    DXDTOFAT='BODY FAT MASS (g)',
    DXDTRPF='PERC TRUNK FAT (%)',
    PTF='PTF (%)',
    DXXRLFAT='RIGHT LEG FAT (g)',
    DXXRAFAT='RIGHT ARM FAT (g)',
    DXXLLFAT='LEFT LEG FAT (g)',
    DXXLAFAT='LEFT ARM FAT (g)',
    DXDLAPF='PERC LEFT ARM (%)',
    DXDRAPF='PERC RIGHT ARM (%)',
    DXDLLPF='PERC LEFT LEG (%)',
    DXDRLPF='PERC RIGHT LEG (%)',
    DXDTOBMC='TOTAL BODY MINERAL CONTENT (g)',
    DXDTOBMD='TOTAL BODY MINERAL DENSITY (g/cm^2)',
    DXXHEA='Head Area (cm^2)',
    DXXHEBMC='Head Bone Mineral Content (g)',
    DXXHEBMD='Head Bone Mineral Density (g/cm^2)',
    DXXHEFAT='Head Fat (g)',
    DXDHELE='Head Lean excl BMC (g)',
    DXXHELI='Head Lean incl BMC (g)',
    DXDHETOT='Head Total (g)',
    DXDHEPF='Head Percent Fat',
    DXXLAA='Left Arm Area (cm^2)',
    DXXLABMC='Left Arm BMC (g)',
    DXXLABMD='Left Arm BMD (g/cm^2)',
    DXDLALE='Left Arm Lean excl BMC (g)',
    DXXLALI='Left Arm Lean incl BMC (g)',
    DXDLATOT='Left Arm Total (g)',
    DXXLLA='Left Leg Area (cm^2)',
    DXXLLBMC='Left Leg BMC (g)',
    DXXLLBMD='Left Leg BMD (g/cm^2)',
    DXDLLLE='Left Leg Lean excl BMC (g)',
    DXXLLLI='Left Leg Lean incl BMC (g)',
    DXDLLTOT='Left Leg Total (g)',
    DXXRAA='Right Arm Area (cm^2)',
    DXXRABMC='Right Arm BMC (g)',
    DXXRABMD='Right Arm BMD (g/cm^2)',
    DXDRALE='Right Arm Lean excl BMC (g)',
    DXXRALI='Right Arm Lean incl BMC (g)',
    DXDRATOT='Right Arm Total (g)',
    DXXRLA='Right Leg Area (cm^2)',
    DXARLBV='Right Leg Bone Invalidity Code',
    DXXRLBMC='Right Leg BMC (g)',
    DXXRLBMD='Right Leg BMD(g/cm^2)',
    DXDRLLE='Right Leg Lean excl BMC (g)',
    DXXRLLI='Right Leg Lean incl BMC (g)',
    DXDRLTOT='Right Leg Total (g)',
    DXXLRA='Left Ribs Area (cm^2)',
    DXXLRBMC='Left Ribs BMC (g)',
    DXXLRBMD='Left Ribs BMD (g/cm^2)',
    DXXRRA='Right Ribs Area (cm^2)',
    DXXRRBMC='Right Ribs BMC (g)',
    DXXRRBMD='Right Ribs BMD (g/cm^2)',
    DXXTSA='Thoracic Spine Area (cm^2)',
    DXXTSBMC='Thoracic Spine BMC (g)',
    DXXTSBMD='Thoracic Spine BMD (g/cm^2)',
    DXXLSA='Lumbar Spine Area (cm^2)',
    DXXLSBMC='Lumbar Spine BMC (g)',
    DXXLSBMD='Lumbar Spine BMD (g/cm^2)',
    DXXPEA='Pelvis Area (cm^2)',
    DXXPEBMC='Pelvis BMC (g)',
    DXXPEBMD='Pelvis BMD (g/cm^2)',
    DXDTRA='Trunk Bone area (cm^2)',
    DXDTRBMC='Trunk BMC (g)',
    DXDTRBMD='Trunk Bone BMD (g/cm^2)',
    DXXTRFAT='Trunk Fat (g)',
    DXDTRLE='Trunk Lean excl BMC (g)',
    DXXTRLI='Trunk Lean incl BMC (g)',
    DXDTRTOT='Trunk Total (g)',
    DXDSTA='Subtotal Area (cm^2)',
    DXDSTBMC='Subtotal BMC (g)',
    DXDSTBMD='Subtotal BMD (g/cm^2)',
    DXDSTFAT='Subtotal Fat (g)',
    DXDSTLE='Subtotal Lean excl BMC (g)',
    DXDSTLI='Subtotal Lean incl BMC (g)',
    DXDSTTOT='Subtotal (Total excl Head) (g)',
    DXDSTPF='Subtotal Percent Fat',
    DXDTOA='Total Area (cm^2)',
    DXDTOLE='Total Lean excl BMC (g)',
    DXDTOLI='Total Lean incl BMC (g)',
    DXDTOTOT='Total Lean+Fat (g)',
    DXXANFM='Android fat mass',
    DXXANLM='Android lean mass',
    DXXANTOM='Android total mass',
    DXXGYFM='Gynoid fat mass',
    DXXGYLM='Gynoid lean mass',
    DXXGYTOM='Gynoid total mass',
    DXXAGRAT='Android to Gynoid ratio',
    DXXAPFAT='Android percent fat',
    DXXGPFAT='Gynoid percent fat',
    RIDAGEYR='AGE (y)',
    RIDRETH1='ETHNICITY',
    SMD057='SMOKING',  # cigarettes smoked per day when quit (code)
    SLD010H='SLEEP',
    PAQ635='WALK-CYCLE',
    PAQ710='TV VIDEOS',
    DBQ700='DIET',
    CBQ505='PIZZA',
    DPQ020='DEPRESSION',
    SMQ020='Smoked >100 cigarettes in life',  # smoked at least 100 cigarettes in life
    SMQ040="Now smokes cigarettes",
    SMD650='Avg # cigarettes/day during past 30 days',
    RIAGENDR='GENDER (2 female 1 male)',
)

short_biom_descriptions = dict(
    VOID='',
    BMXWT='WEIGHT',
    BMXHT='HEIGHT',
    BMXTHICR='THIGH',
    BMXWAIST='WAIST',
    BMXHIP='HIP',
    BMXBMI='BMI',
    DXDTOPF='PBF',
    DXDTOFAT='BODY FAT MASS',
    DXDTRPF='PTF',
    RIDAGEYR='AGE',
    RIDRETH1='ETHNIC',
    SMD057='SMK',
    SLD010H='SLEEP',
    PAQ635='WALK-CYCLE',
    PAQ710='TV VIDEOS',
    DBQ700='DIET',
    CBQ505='PIZZA',
    DPQ020='DEPRESSION',
    SMQ020='SMOKER',
    RIAGENDR='GENDER'
)

long_cond_descriptions = dict(
    VOID='',
    BPQ020='High blood pressure (hypertension)',
    DIQ010='Diabetes',
    MCQ160C='Coronary heart disease',
    MCQ220='Cancer',
    MCQ160A='Arthritis'
)

# colors for training testing on sparse points
col_class_pos = np.array([1.0, 0.4, 0.1])       #
col_class_unc = np.array([0.9, 0.9, 0.7])       #
col_class_neg = np.array([0.1, 0.4, 1.0])       #

# colors for risk map and dense points
col_risk_pos = np.array([0.6, 0.2, 0.2])        #
col_risk_unc = np.array([1.0, 1.0, 1.0])        #
col_risk_neg = np.array([0.2, 0.6, 0.2])        #

# colors for plots
col_train_accuracy = [0.8, 0.8, 0.8]
col_test_accuracy = [0.4, 0.4, 0.5]

epsi = 0.03                                     # 0.03

# Enumeration for NN optimizer
class NN_OPTIM(enum.Enum):
    ADAM = 1
    ADAMW = 2
    ADAMAX = 3
    ADAGRAD = 4
    SGD = 5
    RMSPROP = 6

WOMEN_ID = 2
MEN_ID = 1