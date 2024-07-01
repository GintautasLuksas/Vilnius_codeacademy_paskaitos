import pytest
from Atask import is_palindrome

def test_palindrome():
    assert is_palindrome('madam') == True
    assert is_palindrome('racecar') == True
    assert is_palindrome('12321') == True

def test_non_palindrome():
    assert is_palindrome('hello') == False
    assert is_palindrome('world') == False
    assert is_palindrome('12345') == False

def test_empty_string():
    assert is_palindrome('') == True