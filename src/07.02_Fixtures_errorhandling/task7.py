import pytest
import math


def calculate_square_root(n):
    if not isinstance(n, (int, float)):
        raise TypeError("Input should be an integer or float")
    if n < 0:
        raise ValueError("Input should be a non-negative number")
    return math.sqrt(n)


def test_calculate_square_root_type_error():
    # Test case with integer input
    assert calculate_square_root(25) == 5.0

    assert calculate_square_root(9.0) == 3.0

    with pytest.raises(ValueError):
        calculate_square_root(-1)

    with pytest.raises(TypeError):
        calculate_square_root("16")

    assert calculate_square_root(0) == 0.0

    assert calculate_square_root(10000) == 100.0
