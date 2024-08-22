#모폴로지 연산 적용하기
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

img=cv.imread('JohnHancocksSignature.png',cv.IMREAD_UNCHANGED)

t,bin_img=cv.threshold(img[:,:,3],0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
#오츄 이진화를 적용한 결과를 bin_img에 저장.
plt.imshow(bin_img,cmap='gray'),plt.xticks([]),plt.yticks([])
#cmap을 gray로 설정해 명암영상으로 출력.
plt.show()

#모폴로지 효과 확인 목적으로 영상의 일부만 잘라 b에 저장, 자른 패치를 디스플레이.
b=bin_img[bin_img.shape[0]//2:bin_img.shape[0],0:bin_img.shape[0]//2+1]
plt.imshow(b,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

se=np.uint8([[0,0,1,0,0], #구조요소
            [0,1,1,1,0],
            [1,1,1,1,1],
            [0,1,1,1,0],
            [0,0,1,0,0],
            [0,0,0,0,0]])

b_dilation=cv.dilate(b,se,iterations=1) #팽창연산 적용
plt.imshow(b_dilation,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

b_erosion=cv.erode(b,se,iterations=1) #침식
plt.imshow(b_erosion,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()

b_closing=cv.erode(cv.dilate(b,se,iterations=1),se,iterations=1) #닫기
#팽창을 적용한 영상에 침식을 적용. 즉, 닫기.
plt.imshow(b_closing,cmap='gray'),plt.xticks([]),plt.yticks([])
plt.show()