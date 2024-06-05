#7. Sukurkite klasę Order (užsakymas), kuriame būtų saugoma užsakymo informacija (prekės, kiekiai, kainos).
#Įdiekite prekių pridėjimo, pašalinimo, bendros kainos apskaičiavimo ir nuolaidų taikymo metodus.

class Order():
    def __init__(self, product):
        self.product = product
        self.products = []

    def add_product(self, addition):
        if addition not in self.products
            self.products.append(addition)

    def remove_product(self, remove):
        self.products.remove(remove)

class Product():
    def __init__(self, name: str, amount: int, price: float):
        self.name = name
        self.amount = amount
        self.price = price
