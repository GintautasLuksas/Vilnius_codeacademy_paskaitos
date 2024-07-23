import pytest
from tasks import reverse_string, is_prime, gcd, merge_sorted_lists, count_vowels
from tasks import transpose_matrix, word_frequency, remove_duplicates, common_elements
from tasks import string_to_int, find_missing_number

# Test for Task 5: reverse_string(s)
def test_reverse_string():
    assert reverse_string('hello') == 'olleh'
    assert reverse_string('') == ''
    assert reverse_string('a') == 'a'
    assert reverse_string('12345') == '54321'

# Test for Task 6: is_prime(n)
def test_is_prime():
    assert is_prime(1) == False
    assert is_prime(2) == True
    assert is_prime(3) == True
    assert is_prime(4) == False
    assert is_prime(29) == True
    assert is_prime(49) == False

# Test for Task 7: gcd(a, b)
def test_gcd():
    assert gcd(48, 18) == 6
    assert gcd(48, 0) == 48
    assert gcd(0, 18) == 18
    assert gcd(7, 3) == 1
    assert gcd(100, 100) == 100

# Test for Task 8: merge_sorted_lists(list1, list2)
def test_merge_sorted_lists():
    assert merge_sorted_lists([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]
    assert merge_sorted_lists([], [1, 2, 3]) == [1, 2, 3]
    assert merge_sorted_lists([4, 5, 6], []) == [4, 5, 6]
    assert merge_sorted_lists([], []) == []

# Test for Task 9: count_vowels(s)
def test_count_vowels():
    assert count_vowels('hello') == 2
    assert count_vowels('bcdfghjkl') == 0
    assert count_vowels('AEIOUaeiou') == 10
    assert count_vowels('') == 0

# Test for Task 10: transpose_matrix(matrix)
def test_transpose_matrix():
    assert transpose_matrix([[1, 2, 3], [4, 5, 6]]) == [[1, 4], [2, 5], [3, 6]]
    assert transpose_matrix([[1, 2], [3, 4], [5, 6]]) == [[1, 3, 5], [2, 4, 6]]
    assert transpose_matrix([[1]]) == [[1]]
    assert transpose_matrix([]) == []

# Test for Task 11: word_frequency(text)
def test_word_frequency():
    assert word_frequency('hello world') == {'hello': 1, 'world': 1}
    assert word_frequency('hello hello world') == {'hello': 2, 'world': 1}
    assert word_frequency('') == {}
    assert word_frequency('one two three two three three') == {'one': 1, 'two': 2, 'three': 3}

# Test for Task 12: remove_duplicates(lst)
def test_remove_duplicates():
    assert remove_duplicates([1, 2, 3, 2, 1]) == [1, 2, 3]
    assert remove_duplicates([1, 1, 1, 1]) == [1]
    assert remove_duplicates([]) == []
    assert remove_duplicates([1, 2, 3, 4, 5]) == [1, 2, 3, 4, 5]

# Test for Task 13: common_elements(list1, list2)
def test_common_elements():
    assert common_elements([1, 2, 3], [2, 3, 4]) == [2, 3]
    assert common_elements([1, 2, 3], [4, 5, 6]) == []
    assert common_elements([], [1, 2, 3]) == []
    assert common_elements([1, 2, 3], []) == []

# Test for Task 14: string_to_int(s)
def test_string_to_int():
    assert string_to_int('123') == 123
    assert string_to_int('-123') == -123
    with pytest.raises(ValueError):
        string_to_int('')
    with pytest.raises(ValueError):
        string_to_int('abc')

# Test for Task 15: find_missing_number(lst)
def test_find_missing_number():
    assert find_missing_number([1, 2, 4, 5, 6]) == 3
    assert find_missing_number([2, 3, 4, 5, 6]) == 1
    assert find_missing_number([1, 2, 3, 4, 5]) == 6
    assert find_missing_number([1]) == 2
