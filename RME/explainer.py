import preprocessor
import plots
import numpy as np


class Explainer(object):

    def __init__(self, train_set, vocabulary=None):
        self.train_set = train_set
        if vocabulary is None:
            self.vocabulary = self.get_vocabulary()
        else:
            self.vocabulary = vocabulary
        self.time_steps = []
        self.perturbed_probabilities = np.array([])
        self.probability_change = np.array([])
        self.variances = np.array([])
        self.mean_absolute = 0
        self.mean_change = 0
        self.partial_predictions = []
        self.instance_value = 0

    def get_vocabulary(self):

        vocabulary = [preprocessor.Instance(a).time_steps for a in self.train_set]
        vocabulary = sorted(set([word for time_steps in vocabulary for word in time_steps]))

        return vocabulary

    def explain_instance(self, instance, predict_function, distance='L1', class_index=0):

        if distance not in ['L1', 'L2']:
            raise Exception(
                'distance should be either  \'L1\' or \'L2\'. The value of type was: {}'.format(
                    distance))

        self.compute_probabilities(instance, predict_function, distance, class_index)

    def compute_probabilities(self, instance, predict_function, distance='L2', class_index=0):

        processed_instance = preprocessor.Instance(instance)
        processed_instance.count_time_steps()
        self.instance_value = predict_function(instance)

        self.perturbed_probabilities = np.zeros([len(self.vocabulary), processed_instance.time_steps_len])
        self.time_steps = processed_instance.time_steps

        if len(self.instance_value.shape) > 1:
            self.instance_value = self.instance_value[:, class_index]
            for time_step in range(processed_instance.time_steps_len):
                self.perturbed_probabilities[:, time_step] = (
                    predict_function(processed_instance.perturbations(self.vocabulary, time_step))[:, class_index])
        else:
            for time_step in range(processed_instance.time_steps_len):
                self.perturbed_probabilities[:, time_step] = (
                    predict_function(processed_instance.perturbations(self.vocabulary, time_step)))

        if distance == 'L2':
            self.probability_change = (self.instance_value - self.perturbed_probabilities)**2
        elif distance == 'L1':
            self.probability_change = ((self.instance_value - self.perturbed_probabilities))

        self.mean_absolute = (self.perturbed_probabilities.mean(axis=0))
        self.mean_change = self.probability_change.mean(axis=0)
        self.variances = self.probability_change.var(axis=0)

    def step_by_step_prediction(self, instance, predict_function):

        processed_instance = preprocessor.Instance(instance)
        processed_instance.count_time_steps()
        self.time_steps = processed_instance.time_steps
        partial_predictions = []

        for i in range(processed_instance.time_steps_len):
            partial_instance = ','.join(processed_instance.time_steps[:i+1])
            partial_predictions.append(np.argmax(predict_function([partial_instance])))

        self.partial_predictions = partial_predictions

    def plot_local_perturbations(self, **kwargs):
        plots.plot_local_perturbations(self, **kwargs)

    def plot_single_perturbation(self, perturbation, **kwargs):
        plots.plot_single_perturbation(self, perturbation, **kwargs)

    def plot_time_step_dispersion(self, **kwargs):
        plots.plot_time_step_dispersion(self, **kwargs)

    def plot_partial_predictions(self, **kwargs):
        plots.plot_partial_predictions(self, **kwargs)


class GlobalExplainer(object):

    def __init__(self, train_set):
        self.train_set = train_set
        self.explainer = Explainer(self.train_set)
        self.variances = []
        self.time_steps = []

    def explain_train_set(self, classifier_function, distance='L1', class_index=0):

        no_of_observations = len(self.train_set)
        for i, instance in enumerate(self.train_set):
            self.explainer.explain_instance([instance], classifier_function, distance, class_index)
            self.variances.append(self.explainer.variances)
            print('Explained instances: ' + str(i) + ' of ' + str(no_of_observations) + ' ' + '[' + '=' * int(
                round(30 * (i / no_of_observations), 0)) + '>' + '-' *
                int(round(30 * ((no_of_observations - i) / no_of_observations), 0)) + ']', end='\r')

        self.time_steps = [len(variances) for variances in self.variances]