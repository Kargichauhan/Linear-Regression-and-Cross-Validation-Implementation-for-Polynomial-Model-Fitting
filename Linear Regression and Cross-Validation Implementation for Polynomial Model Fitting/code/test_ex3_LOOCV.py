from code import cv
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return cv.exercise_3_LOOCV('data/synthdata2023.csv', 'figures/synthetic2023-LOOCV.pdf', seed=29)


def test_exercise_3_LOOCV(exercise_results):
    """
    NOTE: This test will only pass if you implement your solution
    to exercise_3_LOOCV similar to the reference solution.
    It is possible for you to have a valid solution that does not
    match these exact values.
    However, if you pass this test, you know you are done.
    :param exercise_results:
    :return:
    """
    loocv_best_model_order, loocv_best_CVtest_log_MSE_loss, loocv_best_model_w = exercise_results
    target_w = numpy.array([3.863364, 51.36047167, 19.71377298, -8.05367471])
    assert loocv_best_model_order == 3
    assert loocv_best_CVtest_log_MSE_loss == pytest.approx(5.3398720591654865)
    assert numpy.allclose(loocv_best_model_w, target_w)


def test_if_LOOCV_figure_exists():
    """
    NOTE: This test will only pass if you save your figure of the
    CV Train and Test plots as a single file with the name
        synthetic2019-LOOCV.png
    If you save your plots a different way, you can ignore this
    test failure.
    :return:
    """
    assert os.path.exists('figures/synthetic2023-LOOCV.pdf')
