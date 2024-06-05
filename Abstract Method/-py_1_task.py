#1 Užduotis
from abc import ABC, abstractmethod

class Shape(ABC):

    @abstractmethod
    def plotas(self, a):
        pass

    @abstractmethod
    def perimetras(self,a):
        pass


class Apskritimas(Shape):
    def plotas(self, a):
        return 3.14 * a **2

    def perimetras(self, a):
        return 2 *3.14 * a


class Staciakampis(Shape):
    def plotas(self, a, b):
        return a * b

    def perimetras(self, a, b):
        return 2 * (a + b)


class Trikampis(Shape):
    def plotas(self, a, b):
        return 0.5 * a * b
    def perimetras(self, a, b, c):
        return a + b + c

mano_apskritimas = Apskritimas()
mano_staciakampis = Staciakampis()
mano_trikampis = Trikampis()

print(mano_apskritimas.plotas(5))
print(mano_apskritimas.perimetras(5))
print(mano_staciakampis.plotas(5, 4))
print(mano_staciakampis.perimetras(5, 6))
print(mano_trikampis.plotas(5, 2))
print(mano_trikampis.perimetras(5, 4, 3))



#2. Užduotis
class Employee(ABC):
    @abstractmethod
    def calculate_pay(self, hours, rate):
        pass

    def get_role(self):
        pass


class Vadybininkas(Employee):
    def calculate_pay(self, hours, rate):
        salary = 1500
        addition = hours * rate * 0.1
        return f'Vadybininko atlygis: {salary + addition}.'

    def get_role(self):
        return 'Vadybininkas'


class Inzinierius(Employee):
    def calculate_pay(self, hours, rate):
        return f'Inžinieriaus atlygis: {hours * rate}.'

    def get_role(self):
        return 'Inzinierius'


class Stazuotojas(Employee):
    def calculate_pay(self, hours, rate, mistakes):
        salary = hours * rate
        return f'Inžinieriaus atlyginimas: {salary - mistakes}.'

    def get_role(self):
        return 'Stazuotojas'

vad = Vadybininkas()
inz = Inzinierius()
staz = Stazuotojas()

print(vad.calculate_pay(30, 11))
print(inz.calculate_pay(30, 13))
print(staz.calculate_pay(40, 6, 100))

print(vad.get_role())
print(inz.get_role())
print(staz.get_role())