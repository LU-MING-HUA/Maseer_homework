from keras.datasets import mnist
from keras.layers import Dense,Dropout,Input
from keras.models import Model,Sequential
from keras.layers import LeakyReLU
from keras.optimizers import Adam
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    (x_train, y_train),(_,_)=mnist.load_data()
    x_train=(x_train.astype(np.float32)-127.5)/127.5
    x_train=x_train.reshape(60000, -1)
    return (x_train, y_train)

x_train, y_train = load_data()
print(x_train.shape, y_train.shape)

def build_generator():
    model=Sequential()

    model.add(Dense(units=256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Dense(units=784, activation='tanh'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

generator=build_generator()
generator.summary()

def build_discriminator():
    model = Sequential()

    model.add(Dense(units=1024, input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))

    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(0.002, 0.5))
    return model

discriminator = build_discriminator()
discriminator.summary()