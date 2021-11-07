import tensorflow as tf
import numpy as np
import keras

from keras import Sequential,layers, metrics
from keras.utils.np_utils import to_categorical
from keras.layers import Dense


# 1. PRODUCE THE MODEL
# First technique
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Second technique
model2 = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(32),
        layers.Dense(10, activation='sigmoid'),
    ]
)
model2.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# 2. Play with the data
data = np.random.random((1000, 784))
labels = np.random.randint(10, size=(1000, 1))
labels = to_categorical(labels,10)

keras.
# 3. Fit the data to the model
model.fit(data, labels, epochs=10, batch_size=32)
model2.fit(data, labels, epochs=10, batch_size=32)

# 4. Either evaluate with the test set or predict a new value
pred = model.predict(np.random.random((1,784)))
pred2 = model2.predict(np.random.random((1,784)))

print(pred)
print(np.argmax(pred,axis=1))
print(pred2)
print(np.argmax(pred2,axis=1))


