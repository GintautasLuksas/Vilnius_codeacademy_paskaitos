# Užduotis 2: Parašykite funkciją max_of_three(a, b, c), kuri grąžina trijų skaičių maksimumą.
# 	Testai: Atlikite testą su teigiamais skaičiais, neigiamais skaičiais ir teigiamų bei neigiamų skaičių mišiniu.

def max_of_three(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c