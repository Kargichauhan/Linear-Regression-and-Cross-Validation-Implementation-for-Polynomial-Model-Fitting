from code import fitpoly
import numpy
import pytest
import os


@pytest.fixture
def exercise_results():
    return fitpoly.exercise_2('data/synthdata2023.csv', 'figures/synthetic2023-3rd-poly.pdf')


def test_ex2_w(exercise_results):
    target = numpy.array([3.863364, 51.36047167, 19.71377298, -8.05367471])
    w = exercise_results
    assert numpy.allclose(w, target)


def test_synthetic_poly_figure_exists():
    assert os.path.exists('figures/synthetic2023-3rd-poly.pdf')
