#감마 보정 실험하기
import cv2 as cv
import numpy as np

img=cv.imread('soccer.jpg')
img=cv.resize(img,dsize=(0,0),fx=0.25,fy=0.25)
#영상을 여러 장 이어붙일 목적으로 영상을 축소

def gamma(f,gamma=1.0):
    f1=f/255.0  #L=256이라고 가정하고 255로 나누어 [0,1] 범위 영상을 정규화.
    return np.uint8(255*(f1**gamma))

gc=np.hstack((gamma(img,0.5),gamma(img,0.75),gamma(img,1.0),gamma(img,2.0),
              gamma(img,3.0)))
cv.imshow('gamma',gc)

cv.waitKey()
cv.destroyAllWindows()