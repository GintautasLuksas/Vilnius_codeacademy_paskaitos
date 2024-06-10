#7. Sukurkite klasę Order (užsakymas), kuriame būtų saugoma užsakymo informacija (prekės, kiekiai, kainos).
#Įdiekite prekių pridėjimo, pašalinimo, bendros kainos apskaičiavimo ir nuolaidų taikymo metodus.

class Product:
    def __init__(self, name: str, amount: int, price: float):
        self.name = name
        self.amount = amount
        self.price = price

    def get_total_price(self) -> float:
        return self.amount * self.price

    def display_product(self):
        print(f"Product: {self.name}, Amount: {self.amount}, Price per unit: {self.price:.2f}, Total: {self.get_total_price():.2f}")

class Order:
    def __init__(self):
        self.products = []

    def add_product(self, product: Product):
        self.products.append(product)

    def remove_product(self, product: Product):
        try:
            self.products.remove(product)
            print(f"Removed {product.name}")
        except ValueError:
            print(f"Product {product.name} not found in the order.")

    def total_price(self) -> float:
        return sum(product.get_total_price() for product in self.products)

    def apply_discount(self, discount: float) -> float:
        total = self.total_price()
        discount_amount = total * (discount / 100)
        return total - discount_amount

    def display_order(self):
        print("Order Details:")
        for product in self.products:
            product.display_product()
        print(f"Total price: {self.total_price():.2f}")

product1 = Product('Banana',5, 6.6)
product2 = Product('Pineapple', 3, 5.4)
product3 = Product('Lemon', 4, 5.6)

my_order = Order()
my_order.add_product(product1)
my_order.add_product(product2)
my_order.add_product(product3)

my_order.display_order()

my_order.total_price()

my_order.remove_product(product1)
my_order.display_order()

my_order.apply_discount(50)
my_order.display_order()

