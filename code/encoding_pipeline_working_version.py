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
task = 'MMR'
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
subject_ID = '57'  # choose a given subject
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
electrode = 60
coefficients = model_final['coefficients'][electrode, :]
plot_coefficients(coefficients, features_list, delays)

# %%
# Plot single trial scores
electrode = 60
features_plot = ['math']
plot_single_trials_fit(model_final, electrode, X, y, X_delay,
                       features_plot, delays)

# %% Plot single electrode, single trial predicitons per feature
electrode = 60
data = combine_single_trial_scores(electrode, X, features_list,
                                   model_final['score_single_trials'])
plot_RainCloud_sorted(data, 'median')

# %%
df = data
df['descriptive_data'] = df.groupby('labels').pipe(lambda x: x.data.transform('median'))

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


data 

descriptive_data = np.zeros(df.shape[0])
for i in range(0, data_frame.shape[0]):
    descriptive_data[i] = np.median(df.data[df.labels == df.labels[i]])
df['descriptive_data'] = descriptive_data


data_frame.groupby(level="labels").mean()


data_frame.groupby(['Label']).mean()



# %%
import pandas as pd
from dplython import (DplyFrame, X, diamonds, select, sift, sample_n,
    sample_frac, head, arrange, mutate, group_by, summarize, DelayFunction) 

# %%
df = DplyFrame(df)
(df >>
  group_by(df.labels) >>
  summarize(median_data = df.data.median())
)

# %%
def median_plus_50(X):
    X = np.median(X)
    return X
#(df >> group_by(X.labels) >> mutate(median_vanis = apply(median_plus_50)))

#(df >> group_by(X.labels) >> summarize(median_vanis = X.data.median_plus_50()))


# %%
df.data.apply(median_plus_50)

# %%

df.groupby(df.labels).apply(median_plus_50)


# %%
df.groupby('labels').apply(median_plus_50)


# %%
def median_plus_50(X):
    X = np.median(X) + 50
    return X
df['mean_var1'] = df.groupby('labels').pipe(lambda x: x.data.transform(median_plus_50))





# %%
def median_plus_50(X):
    X = np.median(X)
    return X
df['descriptive_data'] = df.groupby('labels').pipe(lambda x: x.data.transform('mean'))
df


# %%
def plot_RainCloud_sorted(df, descriptive_stats):
    df['descriptive_data'] = df.groupby('labels').pipe(lambda x: x.data.transform(descriptive_stats))
    # RainCloud plots
    f, ax = plt.subplots(figsize=(20, 8))
    pt.RainCloud(data=df, x='labels', y='data', ax=ax,
                 palette='viridis', box_showfliers=True,
                 box_whis=[10, 10], rain_alpha=0.5,
                 rain_edgecolor='white')
    ax.set_xlabel('Stimuli features', size=20)
    ax.set_ylabel('Prediction (correlation)', size=20)
    plt.tick_params(labelsize=18)
    plt.tick_params(rotation=45, axis='x')
    plt.xlim([-1, len(np.unique(df['labels']))])

# %%
plot_RainCloud_sorted(df, 'median')

# %%
descriptive_stats = 'median'
df['descriptive_data'] = df.groupby('labels').pipe(lambda x: x.data.transform(descriptive_stats))


# %%
df = data.iloc[:,0:2]
def plot_RainCloud_sorted(df, descriptive_stats):
    df['descriptive_data'] = df.groupby('labels').pipe(lambda x: x.data.transform(descriptive_stats))
    df_sorted = df.sort_values(by=['descriptive_data'], ascending=False)
    # RainCloud plots
    f, ax = plt.subplots(figsize=(20, 8))
    pt.RainCloud(data=df_sorted, x='labels', y='data', ax=ax,
                 palette='viridis', box_showfliers=True,
                 box_whis=[10, 10], rain_alpha=0.5,
                 rain_edgecolor='white')
    ax.set_xlabel('Stimuli features', size=20)
    ax.set_ylabel('Prediction (correlation)', size=20)
    plt.tick_params(labelsize=18)
    plt.tick_params(rotation=45, axis='x')
    plt.xlim([-1, len(np.unique(df['labels']))])

plot_RainCloud_sorted(df, 'median')



# %%
from sklearn.preprocessing import scale
def plot_single_trials_fit(model_final, elec, X, y, X_delay, features_plot, delays):
    """
    Plot single trials fit and predictions

    Parameters
    ----------
    model_final :
    elec :
    X :
    y :
    X_delay :
    features_list :
    delays :

    """
    for iif in range(0, len(features_plot)):
        trials_plot = np.unique(X['trial'][X[features_plot[iif]] > 0])
        scores_trials = model_final['score_single_trials'][elec, trials_plot-1]
        y_pred = X_delay[X['trial'] == trials_plot[0]]
        predictions = model_final['model'].predict(y_pred)
        print(features_plot[iif])
        subplot_dim = int(np.round(np.sqrt(len(trials_plot))))
        fig, axs = plt.subplots(nrows=subplot_dim,  ncols=subplot_dim, 
                                figsize=(40, 30))
        for ax, it in zip(axs.flatten(), range(0, len(trials_plot))):
            y_true = scale(y[X['trial'] == trials_plot[it], elec])
            y_pred = scale(predictions[:, elec])
            ax.plot(y_true, color='k', alpha=.2, lw=2)
            ax.plot(y_pred, color='r', lw=2)
            score_tmp = str(scores_trials[it])
            ax.text(300, 2, score_tmp[0:4], fontsize=20)


# %%


# %%
for i in range(10): print(i);print(f'pedro is {i}')

# %%
[i**2 for i in range(20) if i < 5]

# %%
list(map(lambda x: x * np.sqrt(x), [1,2]))

# %%
import numpy as np

# %%
7//2

# %%
1 & 2
# %%
print([i**2 for i in range(20) if i < 5])



# %%
v = [2, -3, 1]
[print(x) for x in v]

# %%
list(map(lambda x: x * np.sqrt(x), [i**2 for i in range(19) if i < 5]))


# %%
[i**2 for i in range(5) if i < 5]


# %%
list(map(lambda x: x * np.sqrt(x), range(10)))


# %%
my_gen = (i*i for i in range(10))

# %%
[print(i) for i in (i*i for i in range(10))]

# %%
[np.array for i in np.array([1,2])]

# %%
