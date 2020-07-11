import numpy as np
from numpy import matlib
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
import pickle
import os


def load_stim_brain_features(data_dir, subject_number):
    # X = stimuli features
    # y = brain features
    file_tmp = os.path.join(data_dir, (subject_number + '_stim_features.csv'))
    X = pd.read_csv(file_tmp)
    file_tmp = os.path.join(data_dir, (subject_number + '_brain_features.csv'))
    y = pd.read_csv(file_tmp, header=None)
    y = y.to_numpy()
    return X, y


def define_trials(X):
    # Extract trials and time samples from concatenated stim features
    times = np.shape(np.unique(X.loc[:, 'time']))
    times = int(times[0])
    trials = int(X.shape[0]/times)
    trials_rs = np.matlib.repmat(np.arange(1, trials+1), times, 1).T
    trials_rs = trials_rs.reshape(-1, 1)
    X['trials'] = trials_rs
    return trials, times


def get_stim_features(data_dir, task, subj):
    # Define stimuli features of interest based on the task
    file_tmp = os.path.join(data_dir, (subj + '_basic_stim_features.csv'))
    stim_features = pd.read_csv(file_tmp)
    features_lists = {'MMR': ['operand_min', 'operand_max',
                              'cross_decade', 'abs_deviant'],
                      'Memoria': ['operand_min', 'operand_max',
                                  'cross_decade', 'abs_deviant'],
                      'VTCLoc': []}
    features_list = features_lists[task] + list(stim_features['features'])
    return features_list


def get_delay_params(task):
    # Get the parameters for generating the delayes features per task
    delay_params = {'MMR': {'start': 0, 'stop': 4, 'step': 0.02},
                    'Memoria': {'start': 0, 'stop': 4, 'step': 0.02},
                    'VTCLoc': {'start': 0, 'stop': 0.85, 'step': 0.02}}
    delay_params = delay_params[task]
    return delay_params


def delay_features(features_list, stim,  dp):
    # Generates delayed versions of the stimuli features
    delays = np.arange(dp['start'], dp['stop'] + dp['step'], dp['step'])
    n_delays = int(len(delays))
    X_delay = np.zeros((stim.shape[0], n_delays), int)

    for fi in range(0, len(features_list)):
        fs = 500
        features = np.array(stim.loc[:, features_list[fi]])  # result
        times = np.shape(np.unique(stim.loc[:, 'time']))
        times = int(times[0])
        r, c = np.shape(stim)
        trials = int(r / times)

        # Reshape features
        features_rp = np.reshape(features, (trials, times))
        features_rp = np.expand_dims(features_rp, axis=1)

        X_delayed = np.zeros((trials, 1, n_delays, times))
        for i in range(trials):
            for ii in range(n_delays):
                window = [int(np.round(delays[ii] * fs)), int(np.round((delays[ii] + dp['step']) * fs))]
                X_delayed[i, 0, ii, window[0]:window[1]] = int(np.unique(features_rp[i]))

        # Concatenate back the delayed features
        X_env = X_delayed.reshape([X_delayed.shape[0], -1,
                                   X_delayed.shape[-1]])
        X = np.hstack(X_env).T
        # print(X.shape)
        X_delay = np.append(X_delay, X, axis=1)

    X_delay = X_delay[:, n_delays - 1:-1]
    # print(X_delay.shape)
    # print(n_delays)
    return X_delay, delays


def cross_validation_iterator(iterator, n_splits):
    if iterator == 'Kfold':
        cv_iterator = KFold(n_splits=n_splits,
                                   shuffle=True)
    return cv_iterator


def fit_encoding_model(model, cv_iterator, y, X, X_delay):
    scores_all = np.zeros([y.shape[1], cv_iterator.n_splits])
    coefs_all = np.zeros([y.shape[1], X_delay.shape[1],
                         cv_iterator.n_splits])
    intercept_all = np.zeros([y.shape[1], cv_iterator.n_splits])
    scores_cv = np.zeros(y.shape[1])
    counter = 0
    for train, test in cv_iterator:
        # Pull the training / testing data for the brain features
        y_train = y[X['trials'].isin(train)]
        y_test = y[X['trials'].isin(test)]

        # Pull the training / testing data for the stim features
        X_train = X_delay[X['trials'].isin(train)]
        X_test = X_delay[X['trials'].isin(test)]

        # Scale all the features
        X_train = scale(X_train)
        X_test = scale(X_test)
        y_train = scale(y_train)
        y_test = scale(y_test)

        # Fit the model, and use it to predict on new data
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        single_trial_score = single_trials_prediciton(model, y_test, X,
                                                      X_delay, trials, score)

        # Get the average (R2)
        for i in range(0, y.shape[1]):
            scores_cv[i] = r2_score(y_train[:, i], predictions[:, i])

        scores_all[:, counter] = scores_cv
        coefs_all[:, :, counter] = model.coef_
        intercept_all[:, counter] = model.intercept_
        counter = counter + 1

    scores_all = np.mean(scores_all, axis=1)
    coefs_all = np.mean(coefs_all, axis=2)
    intercept_all = np.mean(intercept_all, axis=1)
    return model, scores_all, coefs_all, intercept_all


def corr_vec(X, Y):
    Xs = X - X.mean(axis=0)
    Xs /= np.linalg.norm(Xs, axis=0) + 1e-9
    Ys = Y - Y.mean(axis=0)
    Ys /= np.linalg.norm(Ys, axis=0) + 1e-9

    corrs = np.einsum("ij, ij -> j", Xs, Ys)
    return corrs


def single_trials_prediciton(model, y, X, X_delay, trials, score):
    scores_all = np.zeros([y.shape[1], trials])
    trials_all = np.arange(0, trials)+1
    # scores_all[:] = np.nan
    for ft in range(0, trials):
        X_pred = X_delay[X['trials'] == trials_all[ft]]
        y_pred = model.predict(X_pred)
        y_true = y[X['trials'] == trials_all[ft]]
        if score == 'r2':
            scores_all[:, ft] = r2_score(scale(y_true), scale(y_pred),
                                         multioutput='raw_values')
        elif score == 'corr':
            scores_all[:, ft] = corr_vec(y_true, y_pred)
    return scores_all


def fit_model_across_subj(model, cross_val_n_splits, task, subj,
                          data_dir, result_dir):
    print('processing subject ' + subj + ' for task ' + task)
    X, y = load_stim_brain_features(data_dir, subj)

    # Get trials and times from X features
    trials, times = define_trials(X)

    # Get feature list from the task
    features_list = get_stim_features(data_dir, task, subj)

    # Define delayed features
    # times:
    delay_params = get_delay_params(task)
    X_delay, delays = delay_features(features_list, X, delay_params)
    # print('preparting delayed features')

    # Fit cross validated model
    print('training and fitting the model')
    cv_iterator = cross_validation_iterator('KFold', 5):
    model, scores_all, coefs_all, intercept_all = fit_encoding_model(model, cv_iterator, y, X, X_delay)
    
    # Save model and model parameters
    fn_model = result_dir + subj + '_model.sav'
    pickle.dump(model, open(fn_model, 'wb'))
    
    with open(filename, 'wb') as f:
              pickle.dump(model, f)
    
    
    np.savetxt(result_dir + subj + '_scores.csv', scores_all, delimiter=',')
    np.savetxt(result_dir + subj + '_coefs.csv', coefs_all, delimiter=',')
    np.savetxt(result_dir + subj + '_intercept.csv', intercept_all, delimiter=',')
    # print('done! and saving the results')

    # Get the scores for single trials across all electrodes
    score_metric = 'corr'  # Pearson's r, r2_score from scikit learn can also be used
    score_single_trials = single_trials_prediciton(model, y, X, X_delay, trials, score_metric)
    np.savetxt(result_dir + subj + '_scores_single_trials.csv', score_single_trials, delimiter=',')
    print('done!')
