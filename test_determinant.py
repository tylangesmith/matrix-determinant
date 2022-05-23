""" Test the determinant function """

import numpy as np
import pytest
from determinant import determinant


@pytest.fixture(scope="class")
def one_by_one() -> np.ndarray:
    """
    1 x 1 matrix test fixture
    """
    return np.array([[7], [5]])

@pytest.fixture(scope="class")
def two_by_two() -> np.ndarray:
    """
    2 x 2 matrix test fixture
    """
    return np.array([[2, 2], [3, 4]])

@pytest.fixture(scope="class")
def three_by_three() -> np.ndarray:
    """
    3 x 3 matrix test fixture
    """
    return np.random.uniform(0, 10, (3, 3))

@pytest.fixture(scope="class")
def four_by_four() -> np.ndarray:
    """
    4 x 4 matrix test fixture
    """
    return np.random.uniform(0, 10, (4, 4))

@pytest.fixture(scope="class")
def ten_by_ten() -> np.ndarray:
    """
    10 x 10 matrix test fixture
    """
    return np.random.uniform(0, 10, (10, 10))


@pytest.mark.usefixtures("one_by_one", "two_by_two", "three_by_three", "four_by_four", "ten_by_ten")
class TestDet:
    """
    Tests for the determinant function
    """

    def test_two_by_two(self, two_by_two: np.ndarray):
        """
        Test to see if the value is correct for 2 x 2 matrix
        """

        got = determinant(two_by_two)
        want = np.linalg.determinant(two_by_two)
        assert np.isclose(got, want)

    def test_three_by_three(self, three_by_three: np.ndarray):
        """
        Test to see if value is correct for 3 x 3 matrix
        """

        got = determinant(three_by_three)
        want = np.linalg.determinant(three_by_three)

        assert np.isclose(got, want)

    def test_four_by_four(self, four_by_four: np.ndarray):
        """
        Test to see if value is correct for 4 x 4 matrix
        """

        got = determinant(four_by_four)
        want = np.linalg.determinant(four_by_four)

        assert np.isclose(got, want)

    def test_ten_by_ten(self, ten_by_ten: np.ndarray):
        """
        Test to see if value is correct for 10 x 10 matrix
        """

        got = determinant(ten_by_ten)
        want = np.linalg.determinant(ten_by_ten)

        assert np.isclose(got, want)
