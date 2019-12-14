import tensorflow as tf
import numpy  as np
import pandas as pd

### Model ###

class LinearModel:
    def __init__(self):
        # Important to set dtype as float64
        # otherwise, there is a type error
        # during training as __call__
        # would return a float32
        self.b = tf.Variable(1.0, dtype=tf.float64)
        self.a = tf.Variable(-2.0, dtype=tf.float64)
    
    def __call__(self, x):
        return self.a * x + self.b

### Dataset ###
# ! Important note
# You need to setup X values near 0
# otherwise the training will fail
a, b, N = 0.5, 3, 200
X = np.linspace(0,3,N)
Y = a * X + b + np.random.normal(0,1,N)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def train(model, x, y, loss_fn, l=0.1):
    with tf.GradientTape() as tape:
        loss = loss_fn(y, model(x))
    gradients = tape.gradient(loss, [model.a, model.b])
    # model.a.assign_sub(l * gradients[0])
    # model.b.assign_sub(l * gradients[1])
    optimizer.apply_gradients(zip(gradients, [model.a, model.b]))

if __name__ == "__main__":
    linear_model = LinearModel()
    epochs = 100
    for epoch in range(epochs):
        train(linear_model, X, Y, loss_fn)

    print([linear_model.a, linear_model.b])
    print("Error on a : {:2.2%}".format(abs(linear_model.a - a)/a))
    print("Error on b : {:2.2%}".format(abs(linear_model.b - b)/b))
