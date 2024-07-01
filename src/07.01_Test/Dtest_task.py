import pytest
from Dtask import fibonacci

def test_fibonacci_small_values():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1
    assert fibonacci(2) == 1
    assert fibonacci(3) == 2
    assert fibonacci(4) == 3
    assert fibonacci(5) == 5

def test_fibonacci_edge_cases():
    assert fibonacci(0) == 0
    assert fibonacci(1) == 1

def test_fibonacci_large_values():
    assert fibonacci(10) == 55
    assert fibonacci(20) == 6765
    assert fibonacci(30) == 832040
    assert fibonacci(50) == 12586269025

def test_negative_fibonacci():
    with pytest.raises(ValueError):
        fibonacci(-1)
