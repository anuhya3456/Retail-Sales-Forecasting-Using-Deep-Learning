import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

# Load dataset
df = pd.read_csv("OnlineRetail.csv")
df = df.dropna()
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df = df[df['TotalPrice'] > 0]
df = df[['Quantity', 'UnitPrice', 'TotalPrice']]

# Feature-target split
X = df[['Quantity', 'UnitPrice']]
y = df['TotalPrice']

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model definition
model = Sequential([
    Dense(64, activation='relu', input_shape=(2,)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# Compile and train
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss = model.evaluate(X_test, y_test)
print("Test MSE:", loss)
