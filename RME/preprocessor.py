import re
import copy


class Instance(object):

    def __init__(self, raw_string, keep_non_words=False):
        self.raw_string = str(raw_string)
        self.keep_non_words = keep_non_words
        self.time_steps = []
        self.create_time_step_list()
        self.time_steps_len = 0

    def create_time_step_list(self, split_expression=r'\W+'):
        """Splits raw string into time steps.

        Arguments:
        * split_expression - regex split expression,
        * keep_non_words - boolean, indicates whether to keep non_word (punctuation etc.) or not"""

        time_step_list = (
            [x for x in re.split(r'(%s)|$' % split_expression, self.raw_string) if
             x is not None])

        non_word = re.compile(r'(%s)|$' % split_expression).match

        if self.keep_non_words:
            self.time_steps = time_step_list
        else:
            for i in range(len(time_step_list)):
                if non_word(time_step_list[i]):
                    continue
                self.time_steps.append(time_step_list[i])

    def count_time_steps(self):
        self.time_steps_len = len(self.time_steps)
        return self.time_steps_len

    def perturb_instance(self, vocabulary, time_step_id):
        """Perturbs instance at given time step with supplied vocabulary

        Arguments:
        * vocabulary - list of words
        * time_step_id - id of time step to be perturbed (starts from 0)"""

        perturbed = []

        for word in vocabulary:
            perturbed_instance = copy.copy(self.time_steps)
            perturbed_instance[time_step_id] = word
            perturbed.append(perturbed_instance)

        return perturbed

    def perturbations(self, vocabulary, time_step_id, linking_char=','):

        perturbed = self.perturb_instance(vocabulary, time_step_id)

        if self.keep_non_words:
            perturbations = [''.join(perturbation) for perturbation in perturbed]
        else:
            perturbations = [linking_char.join(perturbation) for perturbation in perturbed]

        return perturbations
