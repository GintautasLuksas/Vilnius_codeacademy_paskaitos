import pytest
def remove_element(lst, elem):
    lst.remove(elem)
    return lst

def add_element(lst, elem):
    lst.append(elem)
    return lst

def unique_elements(lst):
    return list(set(lst))
@pytest.fixture
def list_fixture():
    return [10, 20, 20, 30, 40, 40, 50]

def test_remove_element(list_fixture):
    assert remove_element(list_fixture, 20) == [10, 20, 30, 40, 40, 50]

def test_add_element(list_fixture):
    assert add_element(list_fixture, 60) == [10, 20, 20, 30, 40, 40, 50, 60]

def test_unique_elements(list_fixture):
    assert unique_elements(list_fixture) == [10, 20, 30, 40, 50]
