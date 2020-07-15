# %%
# Encoding model based on the temporal receptive field framework
# Example pipeline

# %%
import numpy as np
import matplotlib.pyplot as plt
from encoding_model.encoding_tools import (fit_model_across_subj,
                                           get_stim_features,
                                           get_delay_params,
                                           delay_features,
                                           load_stim_brain_features)
from encoding_model.viz import (plot_coefficients,
                                plot_single_trials_fit,
                                combine_single_trial_scores,
                                plot_single_trial_scores,
                                plot_RainCloud_sorted)
from sklearn.linear_model import Ridge
from joblib import Parallel, delayed
import pandas as pd
import os

# %%
# Set paths and get list of subjects for a particular task
root_dir = '/Volumes/LBCN8T_2/Stanford/data/encoding/'
# Russ, you will need to create a root dir which with raw and results dirs
# In each dir (raw and results) you will also need a dir for each task:
# VTCLoc: object localizer.
# MMR: simultaneouly presented calculations and memory statementes.
# Memoria: sequencially presented calculations and memory statementes.
task = 'VTCLoc'
data_dir = os.path.join(root_dir, 'raw', task)
result_dir = os.path.join(root_dir, 'results', task)
s_list = pd.read_csv(os.path.join(data_dir, 'subject_list.csv'))
print(f'{task} task has a total o {s_list.shape[0]} subjects')

# %%
# Define model, cross validation squeme and single trial scoring
model = Ridge(alpha=1e5)
cv = {'iterator': 'KFold', 'n_splits': 5}
single_trial_scoring = 'corr'

# %%
# Fit the model and get the results
subject_ID = '132'  # choose a given subject
model_final = fit_model_across_subj(model, cv, single_trial_scoring, task,
                                    subject_ID, data_dir, result_dir,
                                    save_results=False)

# %%
# Load subject's brain and stimuli features
X, y = load_stim_brain_features(data_dir, subject_ID)
features_list = get_stim_features(data_dir, task, subject_ID)
delay_params = get_delay_params(task)
X_delay, delays = delay_features(features_list, X, delay_params)

# %%
# Plot coefficients of a given electrode
electrode = 77
coefficients = model_final['coefficients'][electrode, :]
plot_coefficients(coefficients, features_list, delays)

# %%
# Plot single trial scores
electrode = 77
features_plot = ['faces']
plot_single_trials_fit(model_final, electrode, X, y, X_delay,
                       features_plot, delays)

# %% Plot single electrode, single trial predicitons per feature
electrode = 77
data = combine_single_trial_scores(electrode, X, features_list,
                                   model_final['score_single_trials'])
plot_RainCloud_sorted(data, 'median')

# %%
# Run and a bunch of subjects in parallel
model = Ridge(alpha=1e5)
cv = {'iterator': 'KFold', 'n_splits': 5}
single_trial_scoring = 'corr'
task_list = ['VTCLoc']  # ''  'VTCLoc', 'MMR' Memoria
for it in range(0, len(task_list)):
    task = task_list[it]
    print('fitting ' + task)
    data_dir = os.path.join(root_dir, 'raw', task)
    result_dir = os.path.join(root_dir, 'results', task)
    s_list = pd.read_csv(os.path.join(data_dir, 'subject_list.csv'))
    Parallel(n_jobs=10, verbose=0)(delayed(fit_model_across_subj)
                                   (model, cv, single_trial_scoring, task,
                                   str(s_list.name[i]),
                                   data_dir, result_dir)
                                   for i in range(len(s_list)))