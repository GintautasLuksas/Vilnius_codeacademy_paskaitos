class ShoppingCart:
    def __init__(self, _items: dict):
        self._items = _items

    def _calculate_total(self):
        amount = 0
        for key, values in self._items.items():
            amount += values
        return f'Total of your items in car: {amount}.'

    def add_item(self, addition, amount):
        if addition in self._items:
            self._items[addition] += amount
        else:
            self._items[addition] = amount
        return self._calculate_total()

    def remove_item(self, removal):
        if removal in self._items:
            del self._items[removal]
        return self._calculate_total()


my_cart = ShoppingCart({'Banana': 4, 'Cucumber': 6})


print(my_cart._calculate_total())

print(my_cart.add_item('Peach', 6))

print(my_cart.remove_item('Banana'))
