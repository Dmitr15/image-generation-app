import cv2
from PIL import Image
# import numpy as np
# img = cv2.imread('C:\\Users\\dovsy\\Downloads\\inpaint-example.png')
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2 = np.zeros_like(img)
# img2[:,:,0] = gray
# img2[:,:,1] = gray
# img2[:,:,2] = gray
# cv2.imwrite('10524.jpg', img2)
img =cv2.imread('C:\\Users\\dovsy\\Downloads\\back_change_1.jpg')
img =cv2.resize(img, (250, 500))
cv2.imwrite("pict1.png", img)