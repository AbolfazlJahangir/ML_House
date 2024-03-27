from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def sigmoid(x, beta1, beta2):
    y = 1 / (1 + np.exp(-beta1 * (x - beta2)))
    return y

df = pd.read_csv("house.csv")

# Handle NaN values more systematically (e.g., using imputation)
df["Area"] = pd.to_numeric(df["Area"], errors='coerce')
df = df.dropna()

df["Address"] = df["Address"].map(df["Address"].value_counts())

msk = np.random.rand(len(df)) < 0.8

cdf = df[["Area", "Room", "Parking", "Warehouse", "Elevator", "Address", "Price"]]

cdf["Parking"] = cdf["Parking"].astype(int)
cdf["Warehouse"] = cdf["Warehouse"].astype(int)
cdf["Elevator"] = cdf["Elevator"].astype(int)

scaler = MinMaxScaler()
numeric_columns = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', 'Price']

cdf[numeric_columns] = scaler.fit_transform(cdf[numeric_columns])

train = cdf[msk]
test = cdf[~msk]

x_train = np.asanyarray(train[["Area", "Room", "Parking", "Warehouse", "Elevator", "Address"]])
x_test = np.asanyarray(test[["Area", "Room", "Parking", "Warehouse", "Elevator", "Address"]])

y_train = np.asanyarray(train[["Price"]])
y_test = np.asanyarray(test[["Price"]])

'''degree = 2
poly = PolynomialFeatures(degree=degree)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)

y_pred_poly = poly_model.predict(x_test_poly)

R2_poly = r2_score(y_test, y_pred_poly)
MSE_poly = mean_squared_error(y_test, y_pred_poly)

print(f"R2 Score (Polynomial Regression): {R2_poly}")
print(f"Mean Squared Error (Polynomial Regression): {MSE_poly}")'''

model = LinearRegression()
model.fit(x_train, y_train)
predict = model.predict(x_test)

R2 = r2_score(y_test, predict)
MSE = mean_squared_error(y_test, predict)

print(f"R2 Score: {R2}")
print(f"Mean Squared Error: {MSE}")

plt.scatter(x_test[:, 0], y_test, label="Original Data", color="blue")

# Sort the x_test values for a smoother plot
sorted_indices = np.argsort(x_test[:, 0])
x_test_sorted = x_test[sorted_indices, 0]
y_pred_poly_sorted = predict[sorted_indices]

# Plot the polynomial regression curve
plt.plot(x_test_sorted, y_pred_poly_sorted, color="red", label="Linear-Regression")

plt.xlabel("Area")
plt.ylabel("Price")
plt.legend()
plt.show()

sample = np.array([[0.03670 ,0.2, 1.0, 1.0, 1.0, 0.80625]])

min_price = df["Price"].min()
max_price = df["Price"].max()

predicted = model.predict(sample)

unnormalized_price = predicted[0][0] * (max_price - min_price) + min_price

print(unnormalized_price)