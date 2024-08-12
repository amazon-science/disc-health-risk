# DiscHealthRisk

This package provides the experimental Python implementation for an upcoming
research paper titled _Modeling health risks using neural network ensembles._

## License

Apache 2.0: https://www.apache.org/licenses/LICENSE-2.0.txt

## Requirements

The source code was tested with the following library versions, although older version may work:

```
python 3.8.17
matplotlib 3.7.2
numpy 1.24.4
torch 2.0.1
sklearn 1.2.2
scipy 1.10.0
pandas 2.0.2
```

## Usage

### 1. Dataset Download

Download the data from the NHANES website: https://wwwn.cdc.gov/nchs/nhanes/. 
Organize files by folders named `./data/NHANES/<tag>_Data/<tag>_<year>.XPT`,
where `<tag>` is, e.g., `ALQ`, `DIQ`, and `<year>` is, e.g., `A_1999_2000`.  

We used the following years:
```
A_1999_2000
B_2001_2002
C_2003_2004
D_2005_2006
E_2007_2008
F_2009_2010
G_2011_2012
H_2013_2014
I_2015_2016
J_2017_2018
```
We used the following data files / `<tags>`:
```
ALQ_*.XPT
BIX_*.XPT
BMX_*.XPT
BPQ_*.XPT
BPX_*.XPT
CBQ_*.XPT
CDQ_*.XPT
CVX_*.XPT
DBQ_*.XPT
DEM_*.XPT  # DEMO
DIQ_*.XPT 
DPQ_*.XPT
DXXAG_*.XPT
DXX_*.XPT
GHB_*.XPT
GLU_*.XPT
HSQ_*.XPT
MCQ_*.XPT
OGTT_*.XPT
PAQ_*.XPT
RDQ_*.XPT
SLQ_*.XPT
SMQ_*.XPT
WHQ_*.XPT
```

### 2. Dataset Preparation

Navigate to `src/disc_health_risk`

Modify the paths to your NHANES dataset directory in `DiscRisk_constants.py`

Convert XPT files to CSV (comma-separated values) files:
```
python Dataprep1_convert_all_XPT_files_to_CSV.py
```

Merge CSV files:
```
python Dataprep2_create_merged_dataset.py
```

Impute missing data values:
```
python Dataprep3_impute.py
```

### 4. Model Training and Evaluation

Navigate to `src/disc_health_risk`

Train an ensemble of small neural networks, and then evaluate results on the test set.
First, update config parameters in `DiscRisk_params.py`, e.g.,
* Select one or both genders via `GENDER_ID`
* Select ensemble size via `NN_ENSEMBLE_SIZE`
* Select model inputs via `NAMES_BIOMS`
* Select the set of conditions that, together (union), constitute "positive to condition" in the model output via `NAMES_CONDS`
* Neural network architecture via `NN_ARCHITECTURE`
* Age minimum and maximum via `AGE_MIN` and `AGE_MAX`
* Output directory location via `DIR_RESULTS`

Run training and evaluation:
```
python DiscRisk_1_train.py
```