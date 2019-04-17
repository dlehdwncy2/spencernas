# -*- coding: utf-8 -*-

# MNIST 데이터를 다운로드 한다.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# TensorFlow 라이브러리를 추가한다.
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import random



nb_classes=10

# 변수들을 설정한다.
#가로 세로 28*28 784개의 픽셀
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32,[None,10])

#가중치
W = tf.Variable(tf.random_normal([784, nb_classes]))
#편향값
b = tf.Variable(tf.zeros([nb_classes]))


# cross-entropy 모델을 설정한다. 특정 숫자로 제한된 결과 값을 도출하는 학습임으로 softmax classification 사용
hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#test models
#Calculate Accracy
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

#parameters
#train_data는 55,000개 이므로 이를 여러개로 나누어 학습을 진행
training_epochs=15
batch_size=100
#55,000 * 15 = 825,000개 데이터 학습하며 100번씩 나눠 550번 반복하여 학습 진행


with tf.Session() as sess:
  #initialize tensorflow variables
  sess.run(tf.global_variables_initializer())
  #Training Cycle
  for epoch in range(training_epochs):
      avg_cost=0
      total_batch=int(mnist.train.num_examples/batch_size)
      for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        c,_=sess.run([cost,optimizer], feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost+=c/total_batch
      print('Epoch: ', '%04d' % (epoch + 1), 'Cost: ', '{:.9f}'.format(avg_cost))

  ######## 밑부분의 코드는 수정하시마십시오 ##########
  # 학습된 모델이 얼마나 정확한지를 출력한다.
  is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
  accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
  print(accuracy.eval(session=sess, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
  ##################################################################################


#mnist 데이터 중 임의의 필기체 하나 선정
  r = random.randint(0, mnist.test.num_examples - 1)
  print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r+1], 1)))
  #학습한 데이터를 통해 예측한 결과 값 도출
  print('Prediction: ', sess.run (tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r+1]}))

#해당 이미지 show
  plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
  plt.show()

