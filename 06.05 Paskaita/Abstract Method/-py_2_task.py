from abc import ABC, abstractmethod
class Employee(ABC):

    @abstractmethod
    def calculate_pay(self, hours, rate):
        pass

    @abstractmethod
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