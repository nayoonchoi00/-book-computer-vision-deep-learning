#보간을 이용해 영상의 기하 변환하기
import cv2 as cv

img=cv.imread('rose.png')
patch=img[250:350,170:270,:]

img=cv.rectangle(img, (170,250),(270,350),(255,0,0), 3)
#오려낸 곳을 파란 박스로 표시
patch1=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_NEAREST)
#최근접이웃
patch2=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_LINEAR)
#양선형 보간
patch3=cv.resize(patch,dsize=(0,0),fx=5,fy=5,interpolation=cv.INTER_CUBIC)
#양3차 보간

cv.imshow('Original',img)
cv.imshow('Resize nearest',patch1)
cv.imshow('Resize bilinear',patch2)
cv.imshow('Resize bicubic',patch3)

cv.waitKey()
cv.destroyAllWindows()