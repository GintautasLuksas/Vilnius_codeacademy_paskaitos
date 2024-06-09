#4.
#4.1 Įdiekite pickle pakeltą.
#4.2 Sukurkite klasę pavadinimu DataSerializer.
#4.3 Implementuokite metodus, kurie išsaugotų python objektą į .pkl formatą ir nuskaitytų iš jo.

import pickle

class DataSerializer:
    @staticmethod
    def save_object(obj, file_path):
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(obj, file)
            print("Objektas sėkmingai išsaugotas į", file_path)
        except Exception as e:
            print("Klaida saugant objektą:", e)

    @staticmethod
    def load_object(file_path):
        try:
            with open(file_path, 'rb') as file:
                obj = pickle.load(file)
            print("Objektas sėkmingai užkrautas iš", file_path)
            return obj
        except Exception as e:
            print("Klaida įkeliant objektą:", e)

data = {'name': 'John', 'age': 30, 'city': 'New York'}

DataSerializer.save_object(data, "data.pkl")

loaded_data = DataSerializer.load_object("data.pkl")
print(loaded_data)