import os
from PIL import Image

dog_path = "E:\\作業\\碩1下\\人工智慧-深度學習\\dog\\"

save_dog_path = "E:\\作業\碩1下\\人工智慧-深度學習\\dog_save\\"
dogdir = os.listdir(dog_path)
saveeddogdir =  os.listdir(save_dog_path)

for file in dogdir:
    img = Image.open(dog_path + file)
    resizeImg = img.resize((28, 28))
    grayImg = resizeImg.convert("L")
    grayImg.save(save_dog_path + file)


cat_path = "E:\\作業\\碩1下\\人工智慧-深度學習\\cat\\"

save_cat_path = "E:\\作業\碩1下\\人工智慧-深度學習\\cat_save\\"
catdir = os.listdir(cat_path)
saveedcatdir =  os.listdir(save_cat_path)

for file in catdir:
    img = Image.open(cat_path + file)
    resizeImg = img.resize((28, 28))
    grayImg = resizeImg.convert("L")
    grayImg.save(save_cat_path + file)

