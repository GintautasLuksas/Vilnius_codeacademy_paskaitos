# Užduotis 1: Parašykite funkciją is_palindrome(s), kuri tikrina, ar eilutė s yra palindromas.
# 	Testai: Patikrinkite palindromines eilutes, nepalindromines eilutes ir tuščias eilutes.

def is_palindrome(my_word):
    return my_word == my_word[::-1]


