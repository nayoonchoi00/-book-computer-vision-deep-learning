#우편번호 인식기 v.2(CNN 버전)구현하기
import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
import winsound

model=tf.keras.models.load_model('cnn_v2.h5')

#reset 함수를 통해 img 라는 영상을 만듦. 
def reset():
    global img
    
    img=np.ones((200,520,3),dtype=np.uint8)*255
    #np.ones함수로 200*520 크기의 3채널 컬러 영상을 저장할 배열 만듦.
    #1로 초기화된 배열에 255를 곱해 255, 즉 모든 화소가 흰색인 배열 만듦.
    for i in range(5):
        cv.rectangle(img,(10+i*100,50),(10+(i+1)*100,150),(0,0,255))
        #지정한 위치에 5개의 빨간 박스 그림
    cv.putText(img,'e:erase s:show r:recognition q:quit',(10,40),
               cv.FONT_HERSHEY_SIMPLEX,0.8,(255,0,0),1)

#grab_numerals 함수로 숫자 5개를 떼어낸다.    
def grab_numerals():
    numerals=[]
    for i in range(5):
        roi=img[51:149,11+i*100:9+(i+1)*100,0]
        #img에서 숫자를 떼어 내
        roi=255-cv.resize(roi,(28,28),interpolation=cv.INTER_CUBIC)
        #28*28크기로 변환
        numerals.append(roi)
        #리스트에 추가
    numerals=np.array(numerals)
    return numerals

def show():
    numerals=grab_numerals()
    #반환값을 받아 numerals에 저장.
    plt.figure(figsize=(25,5))
    for i in range(5):
        plt.subplot(1,5,i+1)
        plt.imshow(numerals[i],cmap='gray')
        plt.xticks([]); plt.yticks([])
    plt.show()
    #다섯개 숫자를 표시.
    
def recognition():
    numerals=grab_numerals()
    numerals=numerals.reshape(5, 28, 28, 1)
    numerals=numerals.astype(np.float32)/255.0
    res=model.predict(numerals) #신경망 모델로 예측
    class_id=np.argmax(res,axis=1)
    #최댓값을 갖는 인덱스를 찾아 calss_id에 저장.
    for i in range(5):
        cv.putText(img,str(class_id[i]),(50+i*100,180),
                   cv.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)
        #빨간색 박스 밑에 인식 결과를 표시
    winsound.Beep(1000, 500)
    #삑소리
    
BrushSiz=4
LColor=(0,0,0)

def writing(event,x,y,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
        #마우스 왼쪽 버튼을 클릭하거나 누른 채 이동하면 
        cv.circle(img,(x,y),BrushSiz,LColor,-1)
        #circle함수로 BrushSiz크기의 원을 검은색으로 그려 글씨를 씀
    elif event==cv.EVENT_MOUSEMOVE and flags==cv.EVENT_FLAG_LBUTTON:
        cv.circle(img,(x,y),BrushSiz,LColor,-1)
        
reset()
cv.namedWindow('Writing')
cv.setMouseCallback('Writing', writing)

while True:
    cv.imshow('Writing',img)
    key=cv.waitKey(1)
    if key==ord('e'):
        reset()
    elif key==ord('s'):
        show()
    elif key==ord('r'):
        recognition()
    elif key==ord('q'):
        break
    
cv.destroyAllWindows()
        