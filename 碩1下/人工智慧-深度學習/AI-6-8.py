from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img = Image.open("數字8.jpg")
reIm = img.resize((28, 28))
im1 = np.array(reIm.convert("L"))
print(im1)