#5. Sukurkite klasę ShoppingCart, kuriame būtų prekių sąrašas.
#Joje turėtų būti metodai: pridėti prekes, pašalinti prekes, peržiūrėti krepšelį ir apskaičiuoti bendrą kainą.

class ShoppingCart:
    def __init__(self):
        self.goods = []

    def add_goods(self, item: str, price: int):
        self.goods.append((item, price))

    def remove_item(self, item: str):
        for item_tuple in self.goods:
            if item_tuple[0] == item:
                self.goods.remove(item_tuple)
                break

    def show_goods(self):
        return self.goods

    def show_price(self):
        total_price = sum(price for _, price in self.goods)  # Calculate total price
        return total_price

cart = ShoppingCart()
cart.add_goods("Apple", 2)
cart.add_goods("Banana", 3)
cart.add_goods("Orange", 4)
print("Cart Contents:", cart.show_goods())

cart.remove_item("Banana")
print("Cart Contents after removal:", cart.show_goods())

print("Total Price:", cart.show_price())


