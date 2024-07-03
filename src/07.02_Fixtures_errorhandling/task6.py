import pytest


def get_element(lst, index):
    if index < 0 or index >= len(lst):
        raise IndexError("Index is out of range")
    return lst[index]


def test_get_element_index_error():
    lst = [1, 2, 3, 4, 5]

    assert get_element(lst, 2) == 3

    with pytest.raises(IndexError):
        get_element(lst, 10)
