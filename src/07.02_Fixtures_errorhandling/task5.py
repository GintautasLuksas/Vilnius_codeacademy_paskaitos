import pytest


# Function to get value from dictionary by key
def get_value_dict(d, key):
    return d.get(key, None)


# Function to remove key-value pair from dictionary
def remove_value_dict(d, key):
    if key in d:
        del d[key]


@pytest.fixture
def dict_data():
    # Updated sample dictionary and key for testing
    dictionary = {'a': 1, 'c': 3}
    key = 'b'  # Note: 'b' key does not exist in dictionary
    return dictionary, key



def test_get_value_dict(dict_data):
    dictionary, key = dict_data
    assert get_value_dict(dictionary, key) == None  # Ensure key 'b' does not exist



def test_remove_value_dict(dict_data):
    dictionary, key = dict_data
    remove_value_dict(dictionary, key)
    assert key not in dictionary
    assert len(dictionary) == 2

def test_remove_value_dict_does_not_change_other_values(dict_data):
    dictionary, key = dict_data
    original_values = list(dictionary.values())  # Capture original values
    remove_value_dict(dictionary, key)
    assert list(dictionary.values()) == original_values
