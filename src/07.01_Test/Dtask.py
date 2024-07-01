# Užduotis 4: Parašykite funkciją fibonacci(n), kuri grąžina n-ąjį Fibonačio skaičių.
# 	Testai: Atlikite testą su mažomis n reikšmėmis, kraštutiniais atvejais, pavyzdžiui, n = 0, ir didelėmis n reikšmėmis.

def fibonacci(n):
    if n < 0:
        raise ValueError("Fibonacci number is not defined for negative numbers")
    elif n == 0:
        return 0
    elif n == 1:
        return 1

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
