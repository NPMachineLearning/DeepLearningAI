import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import numpy as np

# Create X, y data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Create model
model = Sequential([
    layers.Dense(1, input_shape=[1]),
])

# Compile model
model.compile(loss="mean_squared_error",
              optimizer="sgd")

# Fit model to data
model.fit(xs, ys, epochs=100)

# Make prediction
print(f"Preidction: {model.predict([10.0])}")