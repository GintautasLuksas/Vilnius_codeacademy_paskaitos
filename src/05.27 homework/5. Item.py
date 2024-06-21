class Item:
    def __init__(self, name: str, price: float):
        self.name = name
        self.price = price


class Order:
    def __init__(self, order_id: int, items: dict):
        self.order_id = order_id
        self.items = items

    def add_element(self, element, _quantity=1):
        if element in self.items:
            self.items[element] += _quantity
        else:
            self.items[element] = _quantity

    def remove_element(self, element, _quantity=1):
        if element in self.items:
            if self.items[element] > _quantity:
                self.items[element] -= _quantity
            else:
                del self.items[element]

    def total_p(self):
        total_sum = 0
        for item, _quantity in self.items.items():
            total_sum += item.price * quantity
        return total_sum


watermelon = Item('Watermelon', 6.1)
melon = Item('Melon', 2.1)

my_order = Order(1, {})

my_order.add_element(watermelon, 2)
my_order.add_element(melon, 3)


my_order.remove_element(melon, 1)


total_price = my_order.total_p()


print(f"Order ID: {my_order.order_id}")
for item, quantity in my_order.items.items():
    print(f"Item: {item.name}, Quantity: {quantity}, Price per item: {item.price}")
print(f"Total price: {total_price}")
