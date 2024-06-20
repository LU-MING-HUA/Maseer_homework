from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

img = Image.open("AI-7-7_dog.jpg")
reIm = img.resize((28, 28))
im1 = np.array(reIm.convert("L"))
plt.imshow(im1,cmap=plt.get_cmap("gray"))
plt.show()

im1 = im1.reshape(1, 28, 28, 1)
im1 = im1.astype("float32")/255

model = load_model("dogcat_CNN.keras")
# 预测输入数据的类别概率分布
predictions = model.predict(im1)

# 提取每个样本的类别
predicted_classes = np.argmax(predictions, axis=1)

# 显示预测的类别
print(predicted_classes)


#7-7跟6-9檔案一樣