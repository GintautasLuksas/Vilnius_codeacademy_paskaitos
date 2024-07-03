import pytest

@pytest.fixture
def operation_params():
    return 7, 9
def addition(x, y):
    return x + y

def subtraction(x, y):
    return x - y

def multiplication(x, y):
    return x * y

def division(x, y):
    return x / y


def test_math_operations(operation_params):
    x, y = operation_params

    assert addition(x, y) == 16
    assert subtraction(x, y) == -2
    assert multiplication(x, y) == 63
    assert division(x, y) == 7 / 9

