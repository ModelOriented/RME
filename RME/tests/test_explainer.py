import os
import pytest
import pandas as pd
import sklearn
import sklearn.linear_model
from RME.explainer import Explainer
from sklearn.feature_extraction.text import CountVectorizer


@pytest.fixture
def get_data():
    test_file_directory = (os.path.dirname(__file__))+"/test_set.csv"
    data = pd.read_csv(test_file_directory, sep=';')
    return data

@pytest.fixture
def classifier(get_data):
    vectorizer = CountVectorizer(min_df=1)
    vectorizer.fit(get_data.loc[:,'Sentence'])
    train_vectors = vectorizer.transform(get_data.loc[:,'Sentence'])

    c = sklearn.linear_model.LogisticRegression()
    c.fit(train_vectors, get_data.loc[:,'Sentiment'])

    def predictor_fn(text):
        return c.predict_proba(vectorizer.transform(text))

    return predictor_fn


def test_vocabulary_extraction(get_data):

    explainer = Explainer(get_data.loc[:, 'Sentence'])
    assert set(explainer.vocabulary) == {'this', 'is', 'positive', 'negative', 'sentence', 'review', 'comments'}


def test_instance_explanation(get_data, classifier):

    explainer = Explainer(get_data.loc[:, 'Sentence'], vocabulary=['this', 'is', 'positive', 'negative',
                                                                   'sentence', 'review', 'comments'])

    with pytest.raises(Exception, match=r"distance should be either  'L1' or 'L2'. The value of type was: L3"):
        explainer.explain_instance(instance=['To be explained'], predict_function=classifier, distance='L3',
                                   class_index=0)

    explainer.explain_instance(instance=['To be explained'], predict_function=classifier, distance='L1', class_index=0)

    assert len(explainer.time_steps) == 3
    assert explainer.perturbed_probabilities.shape == (len(explainer.vocabulary), 3)
    assert explainer.probability_change.shape == (len(explainer.vocabulary), 3)
    assert len(explainer.mean_absolute) == 3
    assert len(explainer.mean_change) == 3
    assert len(explainer.variances) == 3
    assert explainer.instance_value == 0.5


def test_step_by_step_prediction(get_data, classifier):

    explainer = Explainer(get_data.loc[:, 'Sentence'], vocabulary=['this', 'is', 'positive', 'negative',
                                                                   'sentence', 'review', 'comments'])
    explainer.step_by_step_prediction(instance = ['Getting positive'], predict_function=classifier)

    assert all([obtained == expected for obtained, expected
                in zip(explainer.partial_predictions, [0, 1])])


def test_plot_local_perturbations(get_data, classifier):

    explainer = Explainer(get_data.loc[:, 'Sentence'], vocabulary=['this', 'is', 'positive', 'negative',
                                                                   'sentence', 'review', 'comments'])

    explainer.explain_instance(instance=['To be explained'], predict_function=classifier, distance='L2', class_index=0)

    explainer.plot_local_perturbations(type='probabilities', show_mean=True, plot_type='profiles', highlights=None,
                                       title='Test plot - memory profiles')
    explainer.plot_local_perturbations(type='probability_change', show_mean=True, plot_type='profiles', highlights=None,
                                       title='Test plot - adjusted memory profiles', y_lim=(0, 1))
    explainer.plot_local_perturbations(type='probability_change', show_mean=True, plot_type='scores', highlights=None,
                                       title='Test plot - memory scores with AMP', order_time_steps = True)
    explainer.plot_local_perturbations(type='probability_change', show_mean=False, plot_type='scores', highlights=None,
                                       title='Test plot - memory scores without AMP')
    explainer.plot_local_perturbations(type='probability_change', show_mean=True, plot_type='nothing', highlights=['negative'],
                                       title='Test plot - memory profile for: negative')

    with pytest.raises(Exception, match=r"type should be either  'probabilities' or 'probability_change'. The value of type was: wrong type"):
        explainer.plot_local_perturbations(type='wrong type', show_mean=True, plot_type='nothing',
                                           highlights=['negative'],
                                           title='Test plot - type exception')

    with pytest.raises(Exception, match=r"plot_type should be either  'profiles', 'scores' or 'nothing'. The value of type was: wrong type"):
        explainer.plot_local_perturbations(type='probability_change', show_mean=True, plot_type='wrong type',
                                           highlights=['negative'],
                                           title='Test plot - plot type exception')
        

def test_plot_partial_predictions(get_data, classifier):

    explainer = Explainer(get_data.loc[:, 'Sentence'], vocabulary=['this', 'is', 'positive', 'negative',
                                                                   'sentence', 'review', 'comments'])
    explainer.step_by_step_prediction(instance=['Getting positive'], predict_function=classifier)
    explainer.plot_partial_predictions(class_dictionary={0: 'Negative', 1: 'Positive'} )

    explainer.step_by_step_prediction(instance=['Negative'], predict_function=classifier)
    explainer.plot_partial_predictions()


def test_plot_time_step_dispersion(get_data, classifier):

    explainer = Explainer(get_data.loc[:, 'Sentence'], vocabulary=['this', 'is', 'positive', 'negative',
                                                                   'sentence', 'review', 'comments'])
    explainer.explain_instance(instance=['To be explained'], predict_function=classifier, distance='L1', class_index=0)

    explainer.plot_time_step_dispersion(dispersion_measure='std')
    explainer.plot_time_step_dispersion(dispersion_measure='var')

    with pytest.raises(Exception, match=r"dispersion_measure should be either 'var' or 'std'. The value was: wrong measure"):
        explainer.plot_time_step_dispersion(dispersion_measure='wrong measure')
 
