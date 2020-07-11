# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
# Pipeline encoding model
from encoding_tools import fit_model_across_subj, load_stim_brain_features
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
import pandas as pd
import os

# %%
# Load data from a given subject
root_dir = '/Volumes/LBCN8T_2/Stanford/data/encoding/'
task = 'VTCLoc'
data_dir = os.path.join(root_dir, 'raw', task)
result_dir = os.path.join(root_dir, 'results', task)
s_list = pd.read_csv(os.path.join(data_dir, 'subject_list.csv'))


# %%
model = Ridge(alpha=1e5)
cv = {'iterator': 'KFold', 'n_splits': 5}
fit_model_across_subj(model, cv, task, str(s_list.name[0]), data_dir, result_dir) 


# %%
fit_model_across_subj(model, cross_val_folds, task, str(s_list.name[0]), data_dir, result_dir)

# %%
load_stim_brain_features(data_dir, '62')




import numpy as np
from numpy import matlib



# %%
# Run and save model
model = Ridge(alpha=1e5)
cross_val_folds = 5
task_list = ['VTCLoc']  # ''  'VTCLoc', 'MMR' Memoria
for it in range(0, len(task_list)):
    task = task_list[it]
    print('fitting ' + task)
    data_dir = os.path.join(root_dir, 'raw', task)
    result_dir = os.path.join(root_dir, 'results', task)
    s_list = pd.read_csv(os.path.join(data_dir, 'subject_list.csv'))
    Parallel(n_jobs=10, verbose=0)(delayed(fit_model_across_subj)
                                   (model, cross_val_folds, task,
                                   str(s_list.name[i]),
                                   data_dir, result_dir)
                                   for i in range(len(s_list)))