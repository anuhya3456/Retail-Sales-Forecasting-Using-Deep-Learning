import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (example)
data = pd.read_csv("online_retail.csv")
data = data.dropna()
data["Total"] = data["Quantity"] * data["UnitPrice"]
X = data[["Quantity", "UnitPrice"]]
y = data["Total"]

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(2,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss = model.evaluate(X_test, y_test)
print("Test MSE:", loss)
