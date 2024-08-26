#텐서플로로 데이터 확인하기
import tensorflow as tf
import tensorflow.keras.datasets as ds
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test)=ds.mnist.load_data()
#MNIST에는 70,000개의 필기 숫자 샘플이 있고 훈련집합과 테스트 집합으로 분할되어 있다.
#훈련집합을 읽어 x_train, y_train에 저장, 테스트 집합을 읽어 x_test, y_test에 저장.
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
plt.figure(figsize=(24,3))
plt.suptitle('MNIST',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i],cmap='gray')
    plt.xticks([]);plt.yticks([])
    plt.title(str(y_train[i]),fontsize=30)

(x_train,y_train),(x_test,y_test)=ds.cifar10.load_data()
#CIFAR-10 데이터셋을 읽어온다.
print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
class_names=['airplane','car','bird','cat','deer','dog','frog','horse','ship'
             ,'truck']
plt.figure(figsize=(24,3))
plt.suptitle('CIFAR-10',fontsize=30)
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(x_train[i])
    plt.xticks([]);plt.yticks([])
    plt.title(class_names[y_train[i,0]],fontsize=30)