import pytest
from Btask import max_of_three

def test_max_of_three():
    assert max_of_three(1, 2, 3) == 3
    assert max_of_three(3, 2, 1) == 3
    assert max_of_three(1, 3, 2) == 3
    assert max_of_three(2, 3, 1) == 3
    assert max_of_three(3, 1, 2) == 3
    assert max_of_three(2, 1, 3) == 3
    assert max_of_three(3, 3, 1) == 3
    assert max_of_three(1, 3, 3) == 3
    assert max_of_three(3, 1, 3) == 3
    assert max_of_three(3, 3, 3) == 3
    assert max_of_three(-1, -2, -3) == -1
    assert max_of_three(-3, -1, -2) == -1
    assert max_of_three(-2, -3, -1) == -1
    assert max_of_three(-2, 3, -1) == 3
    assert max_of_three(2, -3, -1) == 2