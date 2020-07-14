# Some usefull tools by Pedro Pinheiro-Chagas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns


def plot_coefficients(coefficients, features_list, delays):
    """
    Plot regression coefficients of delayed predictios by stim features

    Parameters
    ----------
    coefficients :
    features_list :
    delays :
    """
    coefs_reshape = np.reshape(coefficients, (len(features_list),
                               len(delays)))
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.plot(np.abs(delays), coefs_reshape.T,  linewidth=3)
    ax.set_xlabel('Time Delay (delayed predictor)', size=20)
    ax.set_ylabel('Coefficient Value', size=20)
    _ = ax.legend(features_list, fontsize=18)
    plt.tick_params(labelsize=18)


def plot_single_trials_fit(model_final, elec, X, y, X_delay, features_list, delays):
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
    for iif in range(0, len(features_list)):
        trials_plot = np.unique(X['trial'][X[features_list[iif]] == 1])
        scores_trials = model_final['score_single_trials'][elec, trials_plot-1]
        y_pred = X_delay[X['trial'] == trials_plot[0]]
        predictions = model_final['model'].predict(y_pred)
        print(features_list[iif])
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


def combine_single_trial_scores(elec, X, features_list, score_single_trials):
    """
    Combine combine single trial scores per feature
    """
    data_cat = []
    labels_cat = []
    for it in range(0, len(features_list)):
        trials_tmp = np.unique(X['trial'][X[features_list[it]] == 1])
        scores_tmp = np.array(score_single_trials[elec, trials_tmp-1])
        data_cat = np.append(data_cat, scores_tmp)
        labels_tmp = [features_list[it]] * scores_tmp.shape[0]
        labels_cat.extend(labels_tmp)
    data = pd.DataFrame(data_cat, columns={'data'})
    labels = pd.DataFrame(labels_cat, columns={'labels'})
    df = pd.concat([data, labels], axis=1)
    return df


def plot_single_trial_scores(data_frame):
    sns.set(style='ticks')
    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(20, 10))
    # Plot the orbital period with horizontal boxes
    sns.boxplot(x='labels', y='data', data=data_frame,
                whis=[0, 100], palette='vlag')
    # Add in points to show each observation
    sns.swarmplot(x='labels', y='data', data=data_frame,
                  size=2, color='.1', linewidth=0)
    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set_ylabel('Pearson r', size=20)
    ax.set_xlabel('Features', size=20)
    sns.despine(trim=True, left=True)
    ax.tick_params(labelsize=16)
