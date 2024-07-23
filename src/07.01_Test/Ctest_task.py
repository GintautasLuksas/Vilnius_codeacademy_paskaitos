import pytest
from Ctask import factorial

def test_factorial():
    assert factorial(0) == 1
    assert factorial(1) == 1
    assert factorial(2) == 2
    assert factorial(3) == 6
    assert factorial(4) == 24
    assert factorial(5) == 120
    assert factorial(6) == 720
    assert factorial(10) == 3628800

def test_negative_factorial():
    with pytest.raises(ValueError):
        factorial(-1)

def test_large_factorial():
    assert factorial(20) == 2432902008176640000
