import os
import numpy as np
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

print(X_train[19])
print(Y_train[19])