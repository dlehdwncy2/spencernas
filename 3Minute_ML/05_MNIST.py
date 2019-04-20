# 머신러닝 학습의 Hello World 와 같은 MNIST(손글씨 숫자 인식) 문제를 신경망으로 풀어봅니다.
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# 텐서플로우에 기본 내장된 mnist 모듈을 이용하여 데이터를 로드합니다.
# 지정한 폴더에 MNIST 데이터가 없는 경우 자동으로 데이터를 다운로드합니다.
# one_hot 옵션은 레이블을 동물 분류 예제에서 보았던 one_hot 방식의 데이터로 만들어줍니다.
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

#########
# 신경망 모델 구성
######
# 입력 값의 차원은 [배치크기, 특성값] 으로 되어 있습니다.
# 손글씨 이미지는 28x28 픽셀로 이루어져 있고, 이를 784개의 특성값으로 정합니다.
X = tf.placeholder(tf.float32, [None, 784])
# 결과는 0~9 의 10 가지 분류를 가집니다.
Y = tf.placeholder(tf.float32, [None, 10])

# 신경망의 레이어는 다음처럼 구성합니다.
# 784(입력 특성값)
#   -> 256 (히든레이어 뉴런 갯수) -> 256 (히든레이어 뉴런 갯수)
#   -> 10 (결과값 0~9 분류)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
# 입력값에 가중치를 곱하고 ReLU 함수를 이용하여 레이어를 만듭니다.
b1 = tf.Variable(tf.zeros([256]))
L1 = tf.nn.relu(tf.add(tf.matmul(X, W1),b1))

W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
# L1 레이어의 출력값에 가중치를 곱하고 ReLU 함수를 이용하여 레이어를 만듭니다.
b2 = tf.Variable(tf.zeros([256]))
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2),b2)) #relu를 통해서 신경망 계층을 만든다


W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
# 최종 모델의 출력값은 W3 변수를 곱해 10개의 분류를 가지게 됩니다.
b3 = tf.Variable(tf.zeros([10]))
model = tf.add(tf.matmul(L2, W3),b3) #model 텐서에 요소 10개 짜리 배열이 출력된다
#10개의 요소는 가장 큰 값을 가진 인덱스가 예측결과에 가까운 숫자다.
#argmax를 사용할 시 10개의 인덱스 중에서 가장 예측 결과에 가까운 인덱스 숫자를 반환함!

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)#미니배치 평균 손실 값 구함
#optimizer=train_op

#########
# 신경망 모델 학습
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size) #학습 데이터 총 갯수/batch_size
#미니배치가 총 몇 개인지 total_batch에 저장

for epoch in range(15): #epoch란 똑같은 학습을 총 15번한다!
    total_cost = 0

    for i in range(total_batch):
        # 텐서플로우의 mnist 모델의 next_batch 함수를 이용해
        # 지정한 크기만큼 학습할 데이터를 가져옵니다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        #batch_xs : 입력 값인 이미지 데이터
        #batch_ys : 출력 값인 레이블 데이터

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys}) #학습
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))
    #세대 별 학습 값 출력


print('최적화 완료!')

#########
# 결과 확인
######
# model 로 예측한 값과 실제 레이블인 Y의 값을 비교합니다.
# tf.argmax 함수를 이용해 예측한 값에서 가장 큰 값을 예측한 레이블이라고 평가합니다.
# 예) [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 3
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
#X_input 예측값을 넣었을 때랑, Y_input 실제값을 넣었을 때랑 비교
#equal 함수를 통해 예측 값과 실제 값이 '같은 지 확인'

accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#cast 함수를 통해서 is_correct 결과 값을 0 혹은 1로 변환하여 reduce_mean을 통해 평균을 구하면 정확도에 대한 확률이 나옴

print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images,
                                  Y: mnist.test.labels}))


##########출력 값########
'''
Epoch: 0001 Avg. cost = 0.415
Epoch: 0002 Avg. cost = 0.152
Epoch: 0003 Avg. cost = 0.094
Epoch: 0004 Avg. cost = 0.068
Epoch: 0005 Avg. cost = 0.050
Epoch: 0006 Avg. cost = 0.039
Epoch: 0007 Avg. cost = 0.030
Epoch: 0008 Avg. cost = 0.025
Epoch: 0009 Avg. cost = 0.018
Epoch: 0010 Avg. cost = 0.018
Epoch: 0011 Avg. cost = 0.014
Epoch: 0012 Avg. cost = 0.012
Epoch: 0013 Avg. cost = 0.014
Epoch: 0014 Avg. cost = 0.014
Epoch: 0015 Avg. cost = 0.010
최적화 완료!
IS Correct : Tensor("Equal:0", shape=(?,), dtype=bool) 
정확도: 0.9796
'''