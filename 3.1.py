#RGB 컬러 영상을 채널별로 구분해 디스플레이하기
import cv2 as cv
import sys
import os

print(os.getcwd())

img=cv.imread('soccer.jpg')

if img is None:
    sys.exit('파일을 찾을 수 없습니다.')

cv.imshow('img',img)
cv.imshow('R channel',img[:,:,2])
cv.imshow('G channel',img[:,:,1])
cv.imshow('B channel',img[:,:,0])

cv.waitKey()
cv.destroyAllWindows()

