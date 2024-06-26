import pandas as pd

pd.read_csv(r"C:\Users\BossJore\PycharmProjects\06.05_paskaita\src\06.06\countries of the world.csv")
data = pd.head("C:\Users\BossJore\PycharmProjects\06.05_paskaita\src\06.06\countries of the world.csv")
with open('countries of the world.csv', 'r') as file:
    data = file.read()
print(data)
