### import libraries
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/pinheirochagas/Pedro/Stanford/code/lbcn_encoding_decoding/functions/')
import encoding_functions

from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
import sys
from sklearn import cross_validation as cv
#####################################################################################################################


## Import list of subjects
s_list = pd.read_csv('/Volumes/LBCN8T_2/Stanford/data/encoding/subject_list.csv')

data_dir = '/Volumes/LBCN8T_2/Stanford/data/encoding/raw/VTCLoc/'
subject_number = '132'


stim = load_stim_features(data_dir, subject_number)

brain = load_brain_features(data_dir, subject_number)




def load_stim_features(data_dir, subject_number):
    stim = pd.read_csv(data_dir + subject_number + '_stim_features.csv')
    return stim


def load_brain_features(data_dir,subject_number):
    brain_tmp = pd.read_csv(data_dir + subject_number + '_brain_features.csv', header=None)
    brain = brain_tmp.to_numpy()
    return brain


def define_trials(stim):
    # Define trials
    times = np.shape(np.unique(stim.loc[:,'time']))
    times = int(times[0])
    r,c = np.shape(stim)
    trials = int(r/times)
    trials_rs = np.matlib.repmat(np.arange(1,trials+1),times,1).T
    trials_rs = trials_rs.reshape(-1,1)
    stim['trials'] = trials_rs
    return trials, times


def select_features(task):
    features_list = {'MMR': {'math', 'memory'},
                     'Memoria': {'math', 'memory'},
                     'VTCLoc': {'faces','numbers', 'words'}}
    features = list(features_list[task])
    return features


def delay_features(features_list, stim,  start, stop, step):
    # Add delayed features
    delays = np.arange(start, stop + step, step)
    n_delays = int(len(delays))

    X_all = np.zeros((stim.shape[0], n_delays), int)

    for fi in range(0, len(features_list)):
        print(len(features_list))
        print(fi)
        print(features_list[fi])
        fs = 500
        features = np.array(stim.loc[:, features_list[fi]])  # result
        times = np.shape(np.unique(stim.loc[:, 'time']))
        times = int(times[0])
        r, c = np.shape(stim)
        trials = int(r / times)

        # Reshape features
        features_reshape = np.reshape(features, (trials, times))
        features_reshape = np.expand_dims(features_reshape, axis=1)

        X_delayed = np.zeros((trials, 1, n_delays, times))
        for i in range(trials):
            for ii in range(n_delays):
                window = [int(np.round(delays[ii] * fs)), int(np.round((delays[ii] + step) * fs))]
                X_delayed[i, 0, ii, window[0]:window[1]] = int(np.unique(features_reshape[i]))

        # Concatenate back the delayed features
        X_env = X_delayed.reshape([X_delayed.shape[0], -1, X_delayed.shape[-1]])
        X = np.hstack(X_env).T
        print(X.shape)
        X_all = np.append(X_all, X, axis=1)

    X_all = X_all[:, n_delays - 1:-1]
    print(X_all.shape)
    return X_all


def cross_validator(trials, n_folds):
    trials = np.arange(1, trials)
    cross_val_iterator = cv.KFold(len(trials), n_folds=n_folds, shuffle=True)
    return cross_val_iterator


def fit_encoding_model(model, cross_val_iterator, y, X, X_all):
    scores_all = np.zeros([y.shape[1], cross_val_iterator.n_folds])
    coefs_all = np.zeros([y.shape[1], X.shape[1], cross_val_iterator.n_folds])
    intercept_all = np.zeros([y.shape[1], cross_val_iterator.n_folds])
    scores_cv = np.zeros(y.shape[1])
    counter = 0
    for (tr, tt) in cross_val_iterator:
        # Pull the training / testing data for the brain features
        y_tr = y[X['trials'].isin(tr)]
        y_tt = y[X['trials'].isin(tt)]

        # Pull the training / testing data for the stim features
        X_tr = X_all[X['trials'].isin(tr)]
        X_tt = X_all[X['trials'].isin(tt)]

        # Fit the model, and use it to predict on new data
        model.fit(X_tr, y_tr)
        predictions = model.predict(X_tt)

        # Get the average (R2)
        for i in range(0, y.shape[1]):
            scores_cv[i] = r2_score(y_tt[:, i], predictions[:, i])

        scores_all[:, counter] = scores_cv
        coefs_all[:, :, counter] = model.coef_
        intercept_all[:, counter] = model.intercept_
        counter = counter + 1

    scores_all = np.mean(scores_all, axis=1)
    coefs_all = np.mean(coefs_all, axis=2)
    intercept_all = np.mean(intercept_all, axis=1)
    return scores_all, coefs_all, intercept_all
