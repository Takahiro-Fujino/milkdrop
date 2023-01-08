import cv2
import os
import numpy as np
from PIL import Image

#パスの設定
cwd = os.getcwd()
path = cwd +'/image'
photo = '/milkdrop.bmp'
#二値化
img = cv2.imread(path + photo)
img_g = cv2.imread(path + photo, 0)
ret, img_t = cv2.threshold(img_g, 135, 255, cv2.THRESH_BINARY)

#輪郭抽出
contours, hierarchy = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for i, contour in enumerate(contours):
  # 輪郭の面積を求める
  area = cv2.contourArea(contour, True)
max_contour = max(contours, key=lambda x: cv2.contourArea(x))

#マスクの作成
mask = np.zeros_like(img_g)
cv2.drawContours(mask, [max_contour], -1, color=255, thickness=-1)
cv2.imwrite(path + '/mask.bmp', mask)

#合成
im1 = Image.open(path + '/milkdrop.bmp')
#黒背景の作成
im2 = Image.new(mode=im1.mode, size=im1.size, color=(0,0,0))
im_mask = Image.open(path + '/mask.bmp')
im = Image.composite(im1, im2, im_mask)
im.save(path + '/out.bmp')

#表示
img_out = cv2.imread(path + '/out.bmp')
cv2.imshow('out', img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()