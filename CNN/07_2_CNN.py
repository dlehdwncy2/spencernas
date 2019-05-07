import tensorflow as tf
import numpy as np
import random
import os
import matplotlib.pyplot as plt

train_list , test_list = [], []

with open('train.txt') as f:
    for line in f:
        tmp = line.strip().split()
        #[0] = jpg 파일이름
        #[1] = 과일 인덱스  ex) 바나나 = 1 사과 = 0
        train_list.append([tmp[0], tmp[1]])

with open('test.txt') as f:
    for line in f:
        tmp = line.strip().split()
        test_list.append([tmp[0], tmp[1]])


def readimg(path):
    img = plt.imread(path) #imread는 이미지를 다차원 Numpy 배열로 로딩한다.
    return img


def batch(path, batch_size):
    img, label, paths = [], [], []
    for i in range(batch_size):
        img.append(readimg(path[0][0]))
        label.append(int(path[0][1]))
        path.append(path.pop(0))

    return img, label


#이미지 크기
IMG_H = 100 #높이
IMG_W = 100 #너비
IMG_C = 3 #채널

num_class = 3


X = tf.placeholder(tf.float32, [None, IMG_H, IMG_W, IMG_C])
Y = tf.placeholder(tf.int32, [None]) #Y축 분류된 인덱스 값
keep_prob = tf.placeholder(tf.float32)

W_conv1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
b_conv1=tf.Variable(tf.constant(0.1,shape=[32]))
h_conv1=tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME')+b_conv1)
h_pool1=tf.nn.max_pool(h_conv1,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#100*100 -> 50*50

W_conv2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
b_conv2=tf.Variable(tf.constant(0.1,shape=[64]))
h_conv2=tf.nn.relu(tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding="SAME")+b_conv2)
h_pool2=tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#50*50 -> 25*25

W_conv3=tf.Variable(tf.random_normal([3,3,64,128],stddev=0.01))
b_conv3=tf.Variable(tf.constant(0.1,shape=[128]))
h_conv3=tf.nn.relu(tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding="SAME")+b_conv3)
#25*25 -> 5*5

# 네번째 convolutional layer
W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)


W_fc1 = tf.Variable(tf.truncated_normal(shape=[25 * 25 * 128, 384], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
h_conv5_flat = tf.reshape(h_conv4, [-1, 25* 25* 128])
h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)
# Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 3], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[3]))
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
model = tf.nn.softmax(logits)

# 학습에 직접적으로 사용하지 않고 학습 횟수에 따라 단순히 증가시킬 변수를 만듭니다.
global_step = tf.Variable(0, trainable=False, name='global_step')
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=logits))

#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y,logits=logits))
optimizer=tf.train.AdamOptimizer(learning_rate=0.01)
train_op =optimizer.minimize(cost, global_step=global_step)

#########
# 신경망 모델 학습
######

# 모델을 저장하고 불러오는 API를 초기화합니다.
# global_variables 함수를 통해 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정합니다.

sess=tf.Session()
# 모델을 저장하고 불러오는 API를 초기화합니다.
# global_variables 함수를 통해 앞서 정의하였던 변수들을 저장하거나 불러올 변수들로 설정합니다.
saver = tf.train.Saver(tf.global_variables())
batch_size = 1461
ckpt = tf.train.get_checkpoint_state('./model') #./model 디렉터리에 기존에 학습해둔 모델이 있는지 확인
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path): #기존 학습 모델 확인
    saver.restore(sess, ckpt.model_checkpoint_path) #retore를 통해 학습된 값 불러옴
    print("Load sess")
else:
    sess.run(tf.global_variables_initializer()) #없을 시 새로 초기화

    for epoch in range(3):

        for i in range(10):
            batch_data, batch_label = batch(train_list, batch_size) #train_list = numpy로 된 이미지, 라벨
            print(len(batch_label ))
            _, cost_val = sess.run([train_op, cost], feed_dict = {X: batch_data, Y: batch_label,keep_prob: 0.7})

        print('Epoch:', '%04d' % (epoch + 1),
              'cost =', '{:.3f}'.format(cost_val))

    print('최적화 완료!')

    # 최적화 및 학습이 끝난 뒤, 변수를 저장합니다.
    saver.save(sess, './model/dnn.ckpt', global_step=global_step)