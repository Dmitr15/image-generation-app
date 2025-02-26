import cv2
import numpy as np
import matplotlib.pyplot as plt

coke_img = cv2.imread("C:\\Users\\dovsy\\Downloads\\new_zealand.jpg", 1)

# print("Image size is ", coke_img.shape)
# print("Data type of image is ", coke_img.dtype)
coke_img = cv2.resize(coke_img, (200, 200))

#b,g,r = cv2.split(coke_img)

# cv2.imshow('Img', coke_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#plt.figure(figsize=[20,5])

# plt.subplot(141);plt.imshow(r,cmap='gray');plt.title("Red Channel")
# plt.subplot(142);plt.imshow(g,cmap='gray');plt.title("Green Channel")
# plt.subplot(143);plt.imshow(b,cmap='gray');plt.title("Blue Channel")

#imgMerged = cv2.merge((b,g,r))
# img_NZ_rgb = cv2.cvtColor(coke_img, cv2.COLOR_BGR2RGB)
# plt.subplot(144)
#
# #plt.imshow(imgMerged[:,:,::-1])
# plt.imshow(img_NZ_rgb)
# plt.title("Merged Output")
# plt.show()

img_hsv = cv2.cvtColor(coke_img, cv2.COLOR_BGR2HSV)

h,s,v = cv2.split(img_hsv)

plt.figure(figsize=[20,5])
plt.subplot(141);plt.imshow(h,cmap='gray');plt.title("H Channel")
plt.subplot(142);plt.imshow(s,cmap='gray');plt.title("S Channel")
plt.subplot(143);plt.imshow(v,cmap='gray');plt.title("V Channel")
plt.subplot(144);plt.imshow(img_hsv)
plt.title("Original")

plt.show()