# 과적합 방지 기법 중 하나인 Dropout 을 사용해봅니다.
#Dropout : 학습 데이터에만 너무 맞춰줘 있을 경우 그 외 데이터에는 잘 맞지 않는 상황을 말함.
#그래서 학습 단계마다 일부 뉴런을 제거(사용안함)하여 일부 특징이 특정 뉴런들에게 고정되는 것을 막아 과적합 방지

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 신경망 모델 구성
######
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))
# 텐서플로우에 내장된 함수를 이용하여 dropout 을 적용합니다.
# 함수에 적용할 레이어와 확률만 넣어주면 됩니다. 겁나 매직!!
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),b2))
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3),b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(30):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.8})  #keep_prob 플레이스 홀더 생성 후 0.8을 넣어 드롭아웃 사용
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료!')

#########
# 결과 확인
######

####드롭 아웃 기법은 학습 시에는 0.8을 넣어 신경망의 일부를 적용하지만, 학습할 땐 전체 1을 넣는다.

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={X: mnist.test.images, #X 데이터 값
                                   Y: mnist.test.labels, #X에 따른 Y라벨 값
                                   keep_prob: 1}))

#########
# 결과 확인 (matplot)
######
labels = sess.run(model, #테스트 데이터를 이용한 예측 모델 실행하여 결괏값 labels 저장
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels,
                             keep_prob: 1})#label는 0~9까지 배열 인덱스로 되있음



fig = plt.figure() #출력 그래프 준비
for i in range(10): #첫 번째부터 열 번째까지의 이미지
    subplot = fig.add_subplot(2, 5, i + 1) #2열 5행 그래프 만들고 i+1 번째 숫자 이미지 출력
    subplot.set_xticks([])
    subplot.set_yticks([]) #깨끗한 이미지를 위해 X와 Y의 눈금은 제거
    subplot.set_title('%d' % np.argmax(labels[i])) #해당 배열에서 가장 높은 값을 가지 인덱스를 예측한 숫자로 출력
    #np,argmax == tf.argmax랑 같음
    #i 번째 요소가 원한인코딩 방식으로 되어있으므로 해당 배열에서 가장 높은 값을 갖은 인덱스를 숫자로 출력함.
    subplot.imshow(mnist.test.images[i].reshape((28, 28)),
                   cmap=plt.cm.gray_r)
    #28*28 형식 2차원 배열로 변형하여 이미지 형태로 출력
    #cmap 파라미터를 통해 이미지를 그레이스케일로 출력




plt.show()