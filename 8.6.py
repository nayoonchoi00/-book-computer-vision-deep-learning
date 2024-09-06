#ResNet50으로 자연 영상 인식하기
import cv2 as cv
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

model=ResNet50(weights='imagenet')
#ImageNet으로 학습된 가중치를 읽어오라는 지시.


img=cv.imread('rabbit.jpg')
x=np.reshape(cv.resize(img,(224,224)),(1,224,224,3))
#resize 함수로 ResNet50 모델의 입력 크기인 224*224로 변환함
#reshape 함수로 224*224*3 텐서를 1*224*224*3 텐서로 변환.
x=preprocess_input(x) 
#ResNet50모델이 영상을 신경망에 입력하기 전에 수행하는 전처리를 적용

preds=model.predict(x)
top5=decode_predictions(preds, top=5)[0]
#1000개 확률 중에 가장 큰 5개의 확률을 취하고 그들의 부류 이름을 같이 제공할 것을 지시.
print('예측 결과:', top5)

for i in range(5):
    cv.putText(img,top5[i][1]+':'+str(top5[i][2]),(10,20+i*20),
               cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    
cv.imshow('Recognition result',img)

cv.waitKey()
cv.destroyAllWindows()