# -*- coding: utf-8 -*-

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import random

#그래프 중복 에러 방지를 위해 기존 그래프 모두 삭제
tf.reset_default_graph()

############################################################
#첫 번째 컨볼루셔널 계층 : 입력 데이터에 필터를 적용 후 특징 추출
##define first layer


#필터정의

#32개의 특징 추출
num_filters1 = 32
#x는 입력되는 이미지 데이타로, 2차원 행렬(28x28)이 아니라, 1차원 벡터(784)로 되어 있고, 데이타의 수는 무제한으로 정의하지 않았다.
x=tf.placeholder(tf.float32,[None,784])
#x 에대한 무한 개(-1) 이미지를 28x28x1 이미지의 무한개 행렬로 reshape 이용하여 변경
x_image=tf.reshape(x,[-1,28,28,1])
#5x5x1 필터를 사용하고 필터의 수는 32개 그리고, 초기 값은 [5,5,1,32] 차원을 갖는 난수를 생성하는 truncated_normal 사용하여 임의의 수 지정
W_conv1=tf.Variable(tf.truncated_normal([5,5,1,num_filters1],stddev=1))

#필터 정의 후 입력 데이터인 이미지 적용
#conv2d를 활용하여 필터 적용
#28x28x1 사이즈에 32개의 특징을 스트라이드 기법을 이용하여 스트라이드 기법을 통해 적용
#padding : 오버피팅 방지
h_conv1=tf.nn.conv2d(x_image,W_conv1,
                                        strides=[1,1,1,1],padding="SAME")


#활성함수 적용 필터 적용 후 필터링된 값 활성함수 적용
b_conv1=tf.Variable(tf.constant(0.1,shape=[num_filters1]))

#h_conv1 : y=Wx
#b_conv1 : b
#h_conv1_cutoff : y=Wx+b
h_conv1_cutoff=tf.nn.relu(h_conv1+b_conv1)

#strides 2를 적용하여 각 행렬의 크기가 반으로 줄어든 14*14*1 행렬 32개 리턴
h_pool1=tf.nn.max_pool(h_conv1_cutoff,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
############################################################

############################################################
##define second layer
num_filters2=64

#필터의 사이즈가 5*5 였고 32개였으며 64개의 필터를 적용
W_conv2=tf.Variable(tf.truncated_normal([5,5,num_filters1,num_filters2]))
#striders 기법 적용 대상은 추출된 h_pool1
h_conv2=tf.nn.conv2d(h_pool1,W_conv2,
                                    strides=[1,1,1,1],padding="SAME")
b_conv2=tf.Variable(tf.constant(0.1,shape=[num_filters2]))
h_conv2_cutoff=tf.nn.relu(h_conv2+b_conv2)
#32개의 결과 값을 한번더 반으로 더 쪼개서 64개가 리턴되도록 한다.
#14*14*1 에서 7*7*1 64개 행렬로 리턴
h_pool2=tf.nn.max_pool(h_conv2_cutoff,ksize=[1,2,2,1],
                                        strides=[1,2,2,1],padding="SAME")

############################################################
##Fully Connected layer
#입력된 64개의 7*7 행렬을 1차원 행렬로 변환한다.
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*num_filters2])
num_units1=7*7*num_filters2
num_units2=1024

#64*7*7 개의 벡터 입력 값
w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
#1024개 뉴런 이용 학습
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))

#뉴런 계산 한 후 relu 적용
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)


#드롭아웃을 통해 오버피팅 제한하여 과적합을 방지한다.
keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0=tf.Variable(tf.zeros([num_units2,10]))
b0=tf.Variable(tf.zeros([10]))
k=tf.matmul(hidden2_drop,w0)+b0
p=tf.nn.softmax(k)


t=tf.placeholder(tf.float32,[None,10])
loss = -tf.reduce_mean(t * tf.log(k) + (1 - t) * tf.log(1 - k))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(t, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# prepare session

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()



# start training

i = 0

for _ in range(1000):

    i += 1

    batch_xs, batch_ts = mnist.train.next_batch(50)

    sess.run(train_step,

             feed_dict={x:batch_xs, t:batch_ts, keep_prob:0.5})

    if i % 500 == 0:

        loss_vals, acc_vals = [], []

        for c in range(4):

            start = len(mnist.test.labels) / 4 * c

            end = len(mnist.test.labels) / 4 * (c+1)

            loss_val, acc_val = sess.run([loss, accuracy],

                feed_dict={x:mnist.test.images[start:end],

                           t:mnist.test.labels[start:end],

                           keep_prob:1.0})

            loss_vals.append(loss_val)

            acc_vals.append(acc_val)

        loss_val = np.sum(loss_vals)

        acc_val = np.mean(acc_vals)

        print ('Step: %d, Loss: %f, Accuracy: %f'

               % (i, loss_val, acc_val))



saver.save(sess, 'cnn_session')

sess.close()