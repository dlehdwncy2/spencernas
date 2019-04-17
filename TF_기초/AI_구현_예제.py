import tensorflow as tf
from sklearn.cluster import  KMeans
import numpy as np
from pandas.io.parsers import read_csv
import seaborn as sb
import matplotlib .pyplot as plt



#모델 초기화
model=tf.global_variables_initializer()

#CSV 파일을 읽어온 뒤 data 객체에 저장
data=read_csv('price data.csv',sep=',') # ',' 를  기준으로 분리

xy=np.array(data,dtype=np.float32)


#x data는 4가지 데이터에 대하여 슬라이스 해서 가져옴
x_data=xy[:,1:-1]

#y data는 가격 정보를 슬라이스 해서 가져옴
y_data=xy[:,[-1]]

#선형 회귀 모델
X=tf.placeholder(tf.float32,shape=[None,4])
Y=tf.placeholder(tf.float32,shape=[None,1])
W=tf.Variable(tf.random_normal([4,1]),name="weight")
b=tf.Variable(tf.random_normal([1],name="bias"))

Hypothesis=tf.matmul(X,W)+b #선형 회귀는 행렬의 곱 연산을 이용하여 결과식 세움

cost=tf.reduce_mean(tf.square(Hypothesis-Y)) #비용함수

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.000005)  #최적화 함수 설정
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(10001):
    cost_, hypo_, _=sess.run([cost,Hypothesis,train],
                                                feed_dict={X:x_data,
                                                                  Y:y_data})

    if step%500==0:
        print("#",step," 손실비용 : ",cost_)
        print(" - 배추가격 : ",hypo_[0])


#학습한 모델 저장
saver=tf.train.Saver()
save_path=saver.save(sess,"./saved.cpkt")


