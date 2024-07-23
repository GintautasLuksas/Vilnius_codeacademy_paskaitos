# Užduotis 3: Parašykite funkciją factorial(n), kuri grąžina nelyginio sveikojo skaičiaus n faktorialą.
# 	Testai: Atlikite testą su nuliu, teigiamais sveikaisiais skaičiais ir kraštutiniais atvejais, pavyzdžiui kai, n = 1.
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
