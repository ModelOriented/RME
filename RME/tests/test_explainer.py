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
