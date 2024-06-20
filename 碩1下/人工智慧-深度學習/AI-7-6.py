import os
import numpy as np
from PIL import Image
import np_utils
from keras.src.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

X_train = []
Y_train = []

dog_train_path = "E:\\作業\碩1下\\人工智慧-深度學習\\dog_save\\"
dog_train_filedir = os.listdir(dog_train_path)
for file in dog_train_filedir:
    im = Image.open(dog_train_path + file)
    data = np.array(im)
    X_train.append(data)
    Y_train.append(0)


cat_train_path = "E:\\作業\碩1下\\人工智慧-深度學習\\cat_save\\"
cat_train_filedir = os.listdir(cat_train_path)
for file in cat_train_filedir:
    im = Image.open(cat_train_path + file)
    data = np.array(im)
    X_train.append(data)
    Y_train.append(1)

X_train = np.array(X_train)
X_train_normalize = X_train.reshape(20, 784).astype("float32")
X_train_normalize = X_train_normalize/255
Y_train_onehot = to_categorical(Y_train)

model = Sequential()
model.add(Dense(units=256, input_dim=784, kernel_initializer="normal", activation="relu"))
model.add(Dense(units=2, kernel_initializer="normal", activation="softmax"))
# print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(x=X_train_normalize, y=Y_train_onehot, validation_split=0.2, epochs=20, batch_size=5, verbose=2)
model.save("dogcat.keras")


