# Task 5: reverse_string(s)
def reverse_string(s):
    return s[::-1]

# Task 6: is_prime(n)
def is_prime(n):
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

# Task 7: gcd(a, b)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Task 8: merge_sorted_lists(list1, list2)
def merge_sorted_lists(list1, list2):
    return sorted(list1 + list2)

# Task 9: count_vowels(s)
def count_vowels(s):
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

# Task 10: transpose_matrix(matrix)
def transpose_matrix(matrix):
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]

# Task 11: word_frequency(text)
from collections import Counter

def word_frequency(text):
    words = text.split()
    return dict(Counter(words))

# Task 12: remove_duplicates(lst)
def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result

# Task 13: common_elements(list1, list2)
def common_elements(list1, list2):
    return list(set(list1) & set(list2))

# Task 14: string_to_int(s)
def string_to_int(s):
    try:
        return int(s)
    except ValueError:
        raise ValueError("Invalid input for conversion to int")

# Task 15: find_missing_number(lst)
def find_missing_number(lst):
    n = len(lst) + 1
    total = n * (n + 1) // 2
    return total - sum(lst)
