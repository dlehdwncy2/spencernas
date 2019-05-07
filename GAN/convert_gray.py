import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

path1="./test/Apple"
'''
imagePaths1= [os.path.join(path1,file_name) for file_name in os.listdir(path1)]
for imagePath in imagePaths1:
    img=Image.open(imagePath).convert('L')
    img_numpy=np.array(img,'uint8')

    cv2.imwrite(imagePath,img_numpy)

'''

fig = plt.figure()
rows = 1
cols = 3

img1 = cv2.imread(os.path.join(path1,"r_93_100.jpg"))
img2 = cv2.imread(os.path.join(path1,"r_94_100.jpg"))

ax1 = fig.add_subplot(rows, cols, 1)
ax1.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
ax1.set_title('Jumok community')
ax1.axis("off")

ax2 = fig.add_subplot(rows, cols, 2)
ax2.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
ax2.set_title('Withered trees')
ax2.axis("off")

plt.show()
