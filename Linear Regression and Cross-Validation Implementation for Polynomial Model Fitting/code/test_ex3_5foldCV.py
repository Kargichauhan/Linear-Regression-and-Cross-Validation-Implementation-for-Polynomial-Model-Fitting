from code import cv
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return cv.exercise_3_5fold('data/synthdata2023.csv', 'figures/synthetic2023-5fold-CV.pdf', seed=29)


def test_exercise_3_5fold(exercise_results):
    """
    NOTE: This test will only pass if you implement your solution
    to exercise_3_5fold similar to the reference solution.
    It is possible for you to have a valid soluiton that does not
    match these exact values.
    However, if you pass this test, you know you are done.
    :param exercise_results:
    :return:
    """
    k5_best_model_order, k5_best_CVtest_log_MSE_loss, k5_best_model_w = exercise_results
    target_w = numpy.array([3.863364, 51.36047167, 19.71377298, -8.05367471])
    assert k5_best_model_order == 3
    assert k5_best_CVtest_log_MSE_loss == pytest.approx(5.391288076700061)
    assert numpy.allclose(k5_best_model_w, target_w)


def test_if_5foldCV_figure_exists():
    """
    NOTE: This test will only pass if you save your figure of the
    CV Train and Test plots as a single file with the name
        synthetic2019-5fold-CV.png
    If you save your plots a different way, you can ignore this
    test failure.
    :return:
    """
    assert os.path.exists('figures/synthetic2023-5fold-CV.pdf')
