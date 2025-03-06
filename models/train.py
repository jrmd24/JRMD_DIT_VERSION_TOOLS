import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle

file_path = "data/employer.csv"
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

if "anciennete" in data.columns and "salaire" in data.columns:
    data = data.dropna(subset=["anciennete", "salaire"]) 
    data["anciennete"] = pd.to_numeric(data["anciennete"], errors="coerce")
    data["salaire"] = pd.to_numeric(data["salaire"], errors="coerce")
else:
    raise ValueError("Les colonnes 'anciennete' et 'salaire' sont introuvables dans le fichier CSV.")

print(data.head())

X = data[["anciennete"]]
y = data["salaire"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Coefficient de détermination (R²): {r2:.4f}")
print(f"Équation de la droite : Salaire = {model.coef_[0]:.2f} * Ancienneté + {model.intercept_:.2f}")

plt.scatter(X, y, color="blue", label="Données réelles") 
plt.plot(X, model.predict(X), color="red", label="Régression")
plt.xlabel("Ancienneté (années)")
plt.ylabel("Salaire")
plt.title("Régression Linéaire - Ancienneté vs Salaire")
plt.legend()
plt.show()

filename = 'linear_model.pkl'
with open(filename, 'wb') as file: pickle.dump(model, file)