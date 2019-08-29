import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


def plot_local_perturbations(explainer, type='probabilities', show_mean=True, plot_type='profiles', highlights=None,
                             dpi=72, figsize=(8, 6), title='Step probabilities and mean',
                             order_time_steps = False, y_lim=None):

    if type not in ['probabilities', 'probability_change']:
        raise Exception('type should be either  \'probabilities\' or \'probability_change\'. '
                        'The value of type was: {}'.format(type))

    if plot_type not in ['profiles', 'scores', 'nothing']:
        raise Exception('plot_type should be either  \'profiles\', \'scores\' or \'nothing\'. '
                        'The value of type was: {}'.format(plot_type))

    if type == 'probabilities':
        plot_data = explainer.perturbed_probabilities.T
        mean_data = explainer.mean_absolute
    else:
        plot_data = explainer.probability_change.T
        mean_data = explainer.mean_change

    time_steps = explainer.perturbed_probabilities.shape[1]

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    fig.canvas.draw()

    if plot_type == 'profiles':
        ax.plot(range(time_steps), plot_data, lw=0.75, color='grey')
    elif plot_type == 'scores':
        memory_scores = explainer.variances ** (1. / 2)
        ax.bar(range(time_steps), np.where(mean_data>=0,memory_scores,-memory_scores), color='grey')
        ax.axhline(0, color='black')

    if highlights is not None:

        RdPu = mpl.cm.get_cmap('RdPu_r')

        highlights_index = [explainer.vocabulary.index(element) for element in highlights]

        for i, index, label in zip(range(len(highlights)), highlights_index, highlights):
            ax.plot(range(time_steps),
                    plot_data[:, [index]], color=RdPu(i/len(highlights)), label=label, lw=2)

    if show_mean:
        ax.plot(range(time_steps), mean_data, color='red', lw=2, label='Average')

    chart_box = ax.get_position()
    ax.set_position([chart_box.x0, chart_box.y0, chart_box.width * 0.8, chart_box.height])
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 0.8), shadow=True, ncol=1)
    ax.set_title(title, fontsize= 16)
    ax.set_xticks(range(time_steps))
    if order_time_steps:
        ax.set_xticklabels(range(1, time_steps + 1))
    else:
        ax.set_xticklabels(explainer.time_steps)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.tick_params(axis='both', which='major', labelsize = 16)

    fig.tight_layout()


def plot_partial_predictions(explainer, class_dictionary=None, dpi=72, figsize=(8,6), title='Prediction progress'):

    RdPu = mpl.cm.get_cmap('RdPu_r')

    partial_predictions = explainer.partial_predictions

    if class_dictionary is not None:
        partial_predictions = list(map(class_dictionary.get, partial_predictions))

    predictions = list(set(partial_predictions))
    no_of_predictions = len(predictions)
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)

    for prediction in range(no_of_predictions):
        indices = [(i, 1) for i, x in enumerate(partial_predictions) if x == predictions[prediction]]
        ax.broken_barh(indices, (0, 10), facecolor=RdPu(prediction / no_of_predictions))

        ranges = []
        counter = 0

        if len(indices) > 1:
            for i in range(len(indices) - 1):
                if indices[i + 1][0] - indices[i][0] == 1:
                    counter += 1
                else:
                    ranges.append(indices[i][0] - counter / 2 + 0.5)
                    counter = 0
            ranges.append(indices[i + 1][0] - counter / 2 + 0.5)
        else:
            ranges.append(indices[0][0] + 0.5)

        for item in ranges:
            ax.text(item, 2, predictions[prediction], color='white', horizontalalignment='center', fontsize=11,
                    fontweight='bold')

    ax.set_ylim(0, 5)
    ax.set_xlim(0, len(partial_predictions))
    ax.set_xlabel(title)
    ax.set_yticks([], [])
    ax.set_xticks(range(len(partial_predictions) + 1))
    fig.tight_layout()


def plot_time_step_dispersion(explainer, dispersion_measure='std', dpi=72, figsize=(8,6), title = 'Variance vs time step'):

    if dispersion_measure not in ['var', 'std']:
        raise Exception('dispersion_measure should be either \'var\' or \'std\'. The value was: {}'.format(dispersion_measure))

    if dispersion_measure == 'var':
        bars = explainer.variances
    else:
        bars = explainer.variances**(1/2)
        title = 'Standard deviation vs time steps'

    time_steps = explainer.probability_change.shape[1]
    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    ax.barh(range(1, time_steps+1), bars, color='indigo')
    ax.set_title(title)
    ax.set_yticks(range(1,  time_steps+1))
    fig.tight_layout()
