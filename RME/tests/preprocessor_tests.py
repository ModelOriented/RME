from preprocessor import Instance


def test_time_steps_split():

    instance = Instance('Test instance with; nonwords.')
    instance.count_time_steps()

    assert instance.keep_non_words is False
    assert instance.time_steps_len == 4
    assert all([obtained == expected for obtained, expected
                in zip(instance.time_steps, ['Test', 'instance', 'with', 'nonwords'])])


def test_time_steps_split_with_nonwords():

    instance = Instance('Test;words:with!symbols', keep_non_words=True)
    instance.count_time_steps()

    assert instance.keep_non_words is True
    assert instance.time_steps_len == 8
    assert all([obtained == expected for obtained, expected
                in zip(instance.time_steps, ['Test', ';', 'words', ':', 'with', '!', 'symbols', ''])])


def test_pertubations():

    instance = Instance('To be perturbed')
    perturbed = instance.perturbations(vocabulary=['A', 'B', 'C'], time_step_id=1, linking_char=' ')

    assert len(perturbed) == 3
    assert all([obtained == expected for obtained, expected
                in zip(perturbed, ['To A perturbed', 'To B perturbed', 'To C perturbed'])])


def test_perturbations_nonwords():

    instance = Instance('To be: perturbed!', keep_non_words=True)
    perturbed = instance.perturbations(vocabulary=['A', 'B', 'C'], time_step_id=2, linking_char=' ')

    assert len(perturbed) == 3
    assert all([obtained == expected for obtained, expected
                in zip(perturbed, ['To A: perturbed!', 'To B: perturbed!', 'To C: perturbed!'])])

