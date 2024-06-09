import pandas as pd

class DataFrameHandler:
    def __init__(self):
        self.data = None

    def load_dataframe(self, file_path):
        try:
            self.data = pd.read_csv(file_path)
            self.data.columns = self.data.columns.str.strip()
            print("CSV failas įkeltas sėkmingai.")
            print("Stulpelių pavadinimai:", self.data.columns)
        except Exception as e:
            print("Klaida įkeliant CSV failą:", e)

    def filter_data(self, condition):
        if self.data is not None:
            try:
                filtered_data = self.data[condition].copy()
                print("Duomenys sėkmingai sufiltravę.")
                return filtered_data
            except Exception as e:
                print("Klaida filtruojant duomenis:", e)
        else:
            print("Duomenų rinkinys tuščias.")
            return None

    def group_data(self, column):
        if self.data is not None:
            try:
                grouped_data = self.data.groupby(column).size().reset_index(name='count').copy()
                print("Duomenys sėkmingai sugrupuoti.")
                return grouped_data
            except Exception as e:
                print("Klaida grupuojant duomenis:", e)
        else:
            print("Duomenų rinkinys tuščias.")
            return None

handler = DataFrameHandler()
handler.load_dataframe("C:/Users/MrComputer/PycharmProjects/Vilnius_codeacademy_paskaitos/06.06/file.csv")
filtered_data = handler.filter_data(handler.data['Age'] > 25)
print(filtered_data)
