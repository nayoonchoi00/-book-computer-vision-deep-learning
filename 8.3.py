import numpy as np
import tensorflow as tf
import tensorflow.keras.datasets as ds

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.optimizers import Adam

#데이터 준비
(x_train, y_train), (x_test, y_test)=ds.mnist.load_data()
x_train=x_train.reshape(60000,28,28,1)
#2차원 구조(28*28*2)를 1차원 구조(28*28*1)로 변환한다. 
x_test=x_test.reshape(10000,28,28,1)
x_train=x_train.astype(np.float32)/255.0
x_test=x_test.astype(np.float32)/255.0
y_train=tf.keras.utils.to_categorical(y_train,10)
y_test=tf.keras.utils.to_categorical(y_test,10)

#모델 선택(신경망 구조 설계)
cnn = Sequential() #Sequential()을 사용하면 간단한 순차적인 구조를 가진 모델을 쉽게 구성할 수 있다.
cnn.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1))) #28*28 사이즈의 흑백 이미지
#ReLU(Rectified Linear Unit) 는 0보다 큰 입력이 들어오면 그대로 통과시키고 0보다 작은 입력이 들어오면 0을 출력하는 함수이다.
cnn.add(Conv2D(32,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
#데이터의 크기를 줄이면서 주요 특징을 추출하는 역할을 함.
cnn.add(Dropout(0.25))
cnn.add(Conv2D(64,(3,3),activation='relu')) #64개의 필터
cnn.add(Conv2D(64,(3,3),activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Dropout(0.25))
cnn.add(Flatten())
cnn.add(Dense(units=512, activation='relu'))#출력 뉴런의 수 : 512개
cnn.add(Dropout(0.5))
cnn.add(Dense(units=10,activation='softmax')) #출력 뉴런의 수 : 10개


cnn.compile(loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.001),metrics=['accuracy'])
hist=cnn.fit(x_train, y_train, batch_size=128, epochs=20, 
             validation_data=(x_test,y_test),verbose=2)

cnn.save('cnn_v2.h5')

res=cnn.evaluate(x_test,y_test,verbose=0)
print('정확률=',res[1]*100)

