import numpy as np
import random
# 1. Sukurti ir spausdinti masyvus
# 	1.1 Sukurkite 1D NumPy masyvą su reikšmėmis nuo 1 iki 10 ir jį atspausdinkite.
# 	1.2 Sukurkite 2D NumPy masyvą, kurio forma (3, 3) užpildyta reikšme 7, ir jį išspausdinkite.

fromzerotohero = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
fromherototwo = np.array([[7, 7, 7], [7, 7, 7], [7, 7, 7]])

# 2. Pagrindinės aritmetinės operacijos
# 	2.1 Sukurkite du 1D masyvus su reikšmėmis nuo 1 iki 5. Sudėkite, atimkite, padauginkite ir padalykite elementus iš 2.

# arrtask1 = np.array([1, 3, 4, 2, 5])
# arrtask2 = np.array([[2, 3, 4, 5, 1]])
# print(arrtask2 + arrtask1)
# print(arrtask2 - arrtask1)
# print(arrtask2 * arrtask1)
# print(arrtask2 / arrtask1)

# 3. Masyvo indeksavimas ir pjaustymas
# 	3.1 Sukurkite 1D masyvą su reikšmėmis nuo 10 iki 20. Išspausdinkite pirmuosius 5 elementus ir paskutinius 5 elementus.

# arr = np.arange(10, 21)
# print(arr[:5])
# print(arr[-5:])

# 4. Masyvo pertvarkymas
# 	4.1 Sukurkite 1D masyvą su 12 elementų ir pertvarkykite jį į 2D masyvą, kurio forma (3, 4).
# arr2 = np.random.randint(1, 100, 12)
# print(arr2)
# sorted = arr2.reshape(3, 4)
# print(sorted)

# 5. Masyvų generavimas naudojant integruotas funkcijas
# 	5.1 Naudodami linspace sukurkite 10 tolygiai išdėstytų reikšmių nuo 0 iki 1 masyvą.
# 	5.2 Sukurkite 1D masyvą su reikšmėmis nuo 1 iki 10. Apskaičiuokite ir išspausdinkite sumą, vidurkį ir standartinį nuokrypį.

