import tensorflow as tf
from sklearn.cluster import  KMeans
import numpy as np
from pandas.io.parsers import read_csv
import seaborn as sb
import matplotlib .pyplot as plt



#선형 회귀 모델
X=tf.placeholder(tf.float32,shape=[None,4])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([4,1]),name="weight")
b=tf.Variable(tf.random_normal([1],name="bias"))

Hypothesis=tf.matmul(X,W)+b #선형 회귀는 행렬의 곱 연산을 이용하여 결과식 세움

saver=tf.train.Saver()
model=tf.global_variables_initializer()


#사용자로 부터 float32 값 받음
avg_temp=float(input("평균온도 : ")) #15.5
min_temp=float(input("최저온도: ")) #3.5
max_temp=float(input("최고온도: ")) #20.5
rain_fall=float(input("강수량 : ")) #5.0


with tf.Session() as sess:
    sess.run(model)
    save_path="./saved.cpkt"
    saver.restore(sess,save_path) #해당 세션에 저장된 학습 모델을 저장


    data=((avg_temp,min_temp,max_temp,rain_fall), ) #사용자의 입력 값을 이용하여 2차원 배열을 만든다. 기존에 학습된 데이터가 2차원 배열이기 때문

    arr=np.array(data,dtype=np.float32) #사용자가 입력한 데이터를 토대로 초기화

    x_data=arr[0:4]

    dict=sess.run(Hypothesis,feed_dict={X:x_data})

    print("배추가격 : ",dict[0]) #데이터는 한개만 들어갔으므로 하나만 출력