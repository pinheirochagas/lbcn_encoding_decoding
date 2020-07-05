### import libraries
import numpy as np
import pandas as pd
import sys
sys.path.append('/Users/pinheirochagas/Pedro/Stanford/code/lbcn_encoding_decoding/functions/')
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
import sys
from sklearn import cross_validation as cv
#####################################################################################################################


def load_stim_features(data_dir, subject_number):
    X = pd.read_csv(data_dir + subject_number + '_stim_features.csv')
    return X


def load_brain_features(data_dir,subject_number):
    y = pd.read_csv(data_dir + subject_number + '_brain_features.csv', header=None)
    y = y.to_numpy()
    return y


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

    X_delay = np.zeros((stim.shape[0], n_delays), int)

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
        X_delay = np.append(X_delay, X, axis=1)

    X_delay = X_delay[:, n_delays - 1:-1]
    print(X_delay.shape)
    print(n_delays)
    return X_delay, delays


def cross_validator(trials, n_folds):
    trials = np.arange(1, trials)
    cross_val_iterator = cv.KFold(len(trials), n_folds=n_folds, shuffle=True)
    return cross_val_iterator


def fit_encoding_model(model, cross_val_iterator, y, X, X_delay):
    scores_all = np.zeros([y.shape[1], cross_val_iterator.n_folds])
    coefs_all = np.zeros([y.shape[1], X_delay.shape[1], cross_val_iterator.n_folds])
    intercept_all = np.zeros([y.shape[1], cross_val_iterator.n_folds])
    scores_cv = np.zeros(y.shape[1])
    counter = 0
    for (tr, tt) in cross_val_iterator:
        # Pull the training / testing data for the brain features
        y_tr = y[X['trials'].isin(tr)]
        y_tt = y[X['trials'].isin(tt)]

        # Pull the training / testing data for the stim features
        X_tr = X_delay[X['trials'].isin(tr)]
        X_tt = X_delay[X['trials'].isin(tt)]

        # Scale all the features
        X_tr = scale(X_tr)
        X_tt = scale(X_tt)
        y_tr = scale(y_tr)
        y_tt = scale(y_tt)

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
    return model, scores_all, coefs_all, intercept_all


def single_trials_prediciton(model, y, X, X_delay, trials):
    r2_scores_all = np.zeros([y.shape[1], trials])
    #scores_all[:] = np.nan
    for ft in range (1, trials):
        X_pred = X_delay[X['trials']==ft]
        y_pred = model.predict(X_pred)
        y_true =  y[X['trials']==ft]
        r2_scores_all[:, ft] = r2_score(scale(y_true), scale(y_pred), multioutput='raw_values')
    return r2_scores_all


