import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.metrics import r2_score
from sklearn import cross_validation as cv
import pickle



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


def get_stim_features(data_dir, task, subj):
    basic_X = pd.read_csv(data_dir + subj + '_basic_stim_features.csv')

    features_lists = {'MMR': ['operand_min', 'operand_max', 'cross_decade', 'abs_deviant'],
                     'Memoria': ['operand_min', 'operand_max', 'cross_decade', 'abs_deviant', 'number_format'],
                     'VTCLoc': []}

    # 'VTCLoc': ['faces', 'numbers', 'words', 'bodies', 'buildings_scenes', 'falsefonts', 'logos',
    #            'objects', 'scrambled_images', 'shapes']}

    features_list = features_lists[task] + list(basic_X['features'])
    return features_list


def get_delay_params(task):
    # start, stop, step
    delay_params = {'MMR': {'start': 0, 'stop': 4, 'step': 0.02},
                    'Memoria': {'start': 0, 'stop': 4, 'step': 0.02},
                    'VTCLoc': {'start': 0, 'stop': 0.85, 'step': 0.02}}
    delay_params = delay_params[task]
    return delay_params


def delay_features(features_list, stim,  dp):
    # Add delayed features
    delays = np.arange(dp['start'], dp['stop'] + dp['step'], dp['step'])
    n_delays = int(len(delays))

    X_delay = np.zeros((stim.shape[0], n_delays), int)

    for fi in range(0, len(features_list)):
        #print(len(features_list))
        #print(fi)
        #print(features_list[fi])
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
                window = [int(np.round(delays[ii] * fs)), int(np.round((delays[ii] + dp['step']) * fs))]
                X_delayed[i, 0, ii, window[0]:window[1]] = int(np.unique(features_reshape[i]))

        # Concatenate back the delayed features
        X_env = X_delayed.reshape([X_delayed.shape[0], -1, X_delayed.shape[-1]])
        X = np.hstack(X_env).T
        #print(X.shape)
        X_delay = np.append(X_delay, X, axis=1)

    X_delay = X_delay[:, n_delays - 1:-1]
    #print(X_delay.shape)
    #print(n_delays)
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

def corr_vec(X,Y):
    Xs = X - X.mean(axis=0)
    Xs /= np.linalg.norm(Xs, axis=0) + 1e-9
    Ys = Y - Y.mean(axis=0)
    Ys /= np.linalg.norm(Ys, axis=0) + 1e-9

    corrs = np.einsum("ij, ij -> j", Xs, Ys)
    return corrs


def single_trials_prediciton(model, y, X, X_delay, trials, score):
    scores_all = np.zeros([y.shape[1], trials])
    trials_all = np.arange(0, trials)+1
    #scores_all[:] = np.nan
    for ft in range(0, trials):
        X_pred = X_delay[X['trials']==trials_all[ft]]
        y_pred = model.predict(X_pred)
        y_true =  y[X['trials']==trials_all[ft]]
        if score == 'r2':
            scores_all[:, ft] = r2_score(scale(y_true), scale(y_pred), multioutput='raw_values')
        elif score == 'corr':
            scores_all[:, ft] = corr_vec(y_true, y_pred)
    return scores_all


def fit_model_across_subj(model, cross_val_folds, task, subj, data_dir, result_dir):
    print('processing subject ' + subj + ' for task ' + task)
    # Load stim and brain features
    print('loading stim features')
    X = load_stim_features(data_dir, subj)
    print('loading brain features')
    y = load_brain_features(data_dir, subj)

    # Get trials and times from X features
    trials, times = define_trials(X)

    # Get feature list from the task
    features_list = get_stim_features(data_dir, task, subj)

    # Define delayed features
    # times:
    delay_params = get_delay_params(task)
    X_delay, delays = delay_features(features_list, X, delay_params)
    #print('preparting delayed features')

    # Fit cross validated model
    print('training and fitting the model')
    cross_val_iterator = cross_validator(trials, cross_val_folds)
    model, scores_all, coefs_all, intercept_all = fit_encoding_model(model, cross_val_iterator, y, X, X_delay)

    # Save model and model parameters
    fn_model = result_dir + subj + '_model.sav';
    pickle.dump(model, open(fn_model, 'wb'))
    np.savetxt(result_dir + subj + '_scores.csv', scores_all, delimiter=',')
    np.savetxt(result_dir + subj + '_coefs.csv', coefs_all, delimiter=',')
    np.savetxt(result_dir + subj + '_intercept.csv', intercept_all, delimiter=',')
    #print('done! and saving the results')

    # Get the scores for single trials across all electrodes
    score_metric = 'corr'  # Pearson's r, r2_score from scikit learn can also be used
    score_single_trials = single_trials_prediciton(model, y, X, X_delay, trials, 'corr')
    np.savetxt(result_dir + subj + '_scores_single_trials.csv', score_single_trials, delimiter=',')
    print('done!')

