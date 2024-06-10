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
                print("Duomenys sėkmingai sufiltravę:")
                print(filtered_data)
                return filtered_data
            except Exception as e:
                print("Klaida filtruojant duomenis:", e)
        else:
            print("Duomenų rinkinys tuščias.")
            return None

handler = DataFrameHandler()
handler.load_dataframe("C:/Users/BossJore/PycharmProjects/06.05_paskaita/src/06.06/file.csv")
handler.filter_data(handler.data['Age'] > 25)