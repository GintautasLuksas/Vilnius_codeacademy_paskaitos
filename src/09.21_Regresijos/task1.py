import numpy as np

# Duomenys: namų dydis (kvadratiniai metrai) ir jų kainos (tūkstančiai eurų)
X = np.array([50, 80, 120, 160, 200])
Y = np.array([150, 220, 320, 410, 480])

# Apskaičiuojame X ir Y vidurkius
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Apskaičiuojame B1 (nuolydį)
numerator = np.sum((X - X_mean) * (Y - Y_mean))
denominator = np.sum((X - X_mean) ** 2)
B1 = numerator / denominator

# Apskaičiuojame B0 (susikirtimą su Y ašimi)
B0 = Y_mean - B1 * X_mean

# Sukuriame prognozavimo funkciją
def predict(x):
    return B1 * x + B0

# Apskaičiuojame prognozuotas kainas
Y_pred = predict(X)

# Apskaičiuojame vidutinę kvadratinę klaidą (MSE)
MSE = np.mean((Y - Y_pred) ** 2)

# Rezultatai
print(f"B1 (nuolydis): {B1}")
print(f"B0 (susikirtimas): {B0}")
print(f"Vidutinė kvadratinė klaida (MSE): {MSE}")

# Prognozė 150 kv. m. namui
prediction = predict(150)
print(f"Prognozuojama kaina už 150 kv. m. namą: {prediction} tūkst. eurų")      l0,
