import pytest


def get_value_from_dict(d, key):
    if key not in d:
        raise KeyError(f"Key '{key}' does not exist in the dictionary")
    return d[key]


def test_get_value_from_dict_key_error():
    dictionary = {'a': 1, 'b': 2, 'c': 3}

    assert get_value_from_dict(dictionary, 'b') == 2

    with pytest.raises(KeyError):
        get_value_from_dict(dictionary, 'x')
