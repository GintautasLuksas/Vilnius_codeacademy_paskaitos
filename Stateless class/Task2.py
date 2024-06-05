#2. Sukurkite statefull klasę Skaitiklis, kuri saugoja skaičių.
# Joje turėtų būti metodai, skirti skaičiui padidinti, sumažinti, atstatyti ir dabartiniam skaičiui gauti.

class Skaitiklis:
    def __init__(self):
        self.number = 0

    def padidinti(self, increment):
        self.number += increment

    def sumazinti(self, decrement):
        self.number -= decrement

    def atstatyti(self, reset_value):
        self.number = reset_value

    def gauti_dabartini(self):
        return self.number

skaitiklis = Skaitiklis()
skaitiklis.padidinti(5)
print(skaitiklis.gauti_dabartini())

skaitiklis.sumazinti(3)
print(skaitiklis.gauti_dabartini())

skaitiklis.atstatyti(10)
print(skaitiklis.gauti_dabartini())


