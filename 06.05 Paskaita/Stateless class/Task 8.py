#8. Sukurkite klasę DataTransformer su statiniais metodais, kad atliktumėte
# duomenų transformavimo operacijas: skaičių sąrašo normalizavimą,
#eilučių kodavimą/dekodavimą ir duomenų filtravimą pagal kriterijus.

class DataTransformer:
    def __init__(self):
        self.data = []

    @staticmethod
    def min_max_normalize(data):
        min_val = min(data)
        max_val = max(data)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        return normalized_data

    @staticmethod
    def code(data):
        coded = []
        for new in data:
            coded.append(new+2)
        return coded
    @staticmethod
    def decode(data):
        decoded = []
        for new in data:
            decoded.append(new-2)
        return decoded
    @staticmethod
    def filter(data):
        filtered = sorted(data)
        print(f'Numbers filtered: {filtered}.')




my_thing = DataTransformer()
print(my_thing.min_max_normalize([10, 20, 30, 40, 50]))

print(my_thing.code([10, 50, 300, 440, 250]))
print(my_thing.decode([10, 220, 130, 40, 520]))
my_thing.filter([10, 210, 30, 420, 50])
