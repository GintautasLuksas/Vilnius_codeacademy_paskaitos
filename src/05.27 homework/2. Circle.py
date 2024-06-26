class Circle:
    def __init__(self, radius: int):
        self.radius = radius

    def area(self):
        return (self.radius ** 2) * 3.14

    def circumference(self):
        return 2 * 3.14 * self.radius


my_circle = Circle(radius=6)

print(my_circle.area())

print(my_circle.circumference())
