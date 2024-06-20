# import np_utils
# import numpy as np
# from keras.datasets import mnist
# from keras.src.utils.np_utils import to_categorical

# (x_train_image, y_train_label),(x_test_image, y_test_label) = mnist.load_data()

# x_Train = x_train_image.reshape(60000, 28, 28, 1).astype("float32")
# x_Train_normalize = x_Train/255
# print(x_train_image[0])
# print(x_Train[0])

# y_Train_OneHot = to_categorical(y_train_label)
# print(y_train_label[0])
# print(y_Train_OneHot[0])

# #分隔~~~~~~~~~~~~
# x_Test = x_test_image.reshape(10000, 28, 28, 1).astype("float32")
# x_Test_normalize = x_Test/255

# y_Test_OneHot = to_categorical(y_test_label)

# '''
# from keras.models import Sequential
# from keras.layers import Dense
# model = Sequential()
# model.add(Dense(units=256, input_dim=784, kernel_initializer='normal', activation='relu')) #784輸入層 256隱藏層
# model.add(Dense(units=10,kernel_initializer='normal', activation='sigmoid')) #新增10層輸出層
# '''

# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
# model = Sequential()
# model.add(Conv2D(filters=16, kernel_size=(5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(filters=36, kernel_size=(5, 5), padding="same", activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation="relu"))
# model.add(Dense(10, activation="softmax"))
# print(model.summary())



# model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
# model.fit(x=x_Train_normalize, y=y_Train_OneHot, validation_split=0.3, epochs=5, batch_size=2000, verbose=2) 
# #訓練10次 每次100張圖片 #0.2是指所有資料中只拿2成來當驗證集 verbose=2是要把資料秀出來

# scores = model.evaluate(x_Test_normalize, y_Test_OneHot)
# print(scores[1])

# prediction = model.predict(x_Test_normalize)
# predicted_class = np.argmax(prediction, axis=1)


# import matplotlib.pyplot as plt
# fig = plt.gcf()
# plt.imshow(x_test_image[5], cmap="binary")
# plt.show()
# print(prediction[5])
# print(predicted_class[5])

# 辨識數字改辨識貓狗

import np_utils
import numpy as np
from keras.datasets import mnist
from keras.src.utils.np_utils import to_categorical
import os
from PIL import Image

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
X_train_normalize = X_train.reshape(20, 28, 28, 1).astype("float32")
X_train_normalize = X_train_normalize/255
Y_train_onehot = to_categorical(Y_train)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), padding="same", input_shape=(28, 28, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36, kernel_size=(5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(2, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=['accuracy'])
model.fit(x=X_train_normalize, y=Y_train_onehot, validation_split=0.2, epochs=20, batch_size=2, verbose=2) 
#訓練10次 每次100張圖片 #0.2是指所有資料中只拿2成來當驗證集 verbose=2是要把資料秀出來

scores = model.evaluate(X_train_normalize, Y_train_onehot)
print(scores[1])

prediction = model.predict(X_train_normalize)
predicted_class = np.argmax(prediction, axis=1)


import matplotlib.pyplot as plt
fig = plt.gcf()
plt.imshow(X_train[5], cmap="binary")
plt.show()
print(prediction[5])
print(predicted_class[5])
model.save("dogcat_CNN.keras")