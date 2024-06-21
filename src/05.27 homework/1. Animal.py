class Animal:
    def __init__(self, name: str, species: str):
        self.name = name
        self.species = species

    def describe(self):
        print(f'Animal name is: {self.name} it might bite it might not. It is {self.species} {self.name} indeed.')


shark = Animal('Shark', 'Great White')

shark.describe()
