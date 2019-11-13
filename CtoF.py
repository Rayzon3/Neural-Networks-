'''
This is a simple neural network to convert C to F using one neural layer
'''

import tensorflow as tf
import numpy as np

# I don't give a damn for warnings!

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

celcius_q = np.array([-200, -90, -40,  14, 32, 46, 59, 72, 100],  dtype=float)
fahrenheit_a = np.array([-328, -130, -40, 57.2,  89.6,  114.8, 138.2, 161.6, 212],  dtype=float)

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])
model = tf.keras.Sequential([l0])

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))
history = model.fit(celcius_q, fahrenheit_a, epochs=10000, verbose=False)

print("23.4 (exp: 74.12)", model.predict([23.4]))
print("0 (exp: 32)", model.predict([0.0]))
print("10 (exp: 50)", model.predict([10.0]))
