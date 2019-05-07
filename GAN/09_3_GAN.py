import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#실제 이미지, 라벨 파싱
###############################################
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
###############################################


#image [다차원 배열화 | 인덱스 라벨]
###############################################
def readimg(path):
    img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    #img = plt.imread(path,cmap="gray") #imread는 이미지를 다차원 Numpy 배열로 로딩한다.
    img=np.reshape(img, [-1, 10000])
    return img

def batch(path, batch_size):
    img, label, paths = [], [], []
    for i in range(batch_size):
        img.append(readimg(path[0][0]))
        label_list= [0 for _ in range(n_class)]
        label_list[int(path[0][1])]=int(path[0][1])
        label.append(label_list)

        path.append(path.pop(0))
    return img, label
###############################################


#옵션
###############################################
n_input=100*100
n_class=3
n_noise=128
total_epoch = 10
batch_size = 1461
learning_rate = 0.0002
n_hidden = 256
###############################################


#신경만 모델 구성
###############################################
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])
###############################################

#생성자
###############################################
G_W1=tf.Variable(tf.random_normal([n_noise,n_hidden],stddev=0.01))
G_b1=tf.Variable(tf.zeros([n_hidden]))
G_W2=tf.Variable(tf.random_normal([n_hidden,n_input],stddev=0.01))
G_b2=tf.Variable(tf.zeros([n_input]))

#판별기 신경망 변수
D_W1=tf.Variable(tf.random_normal([n_input,n_hidden],stddev=0.01))
D_b1=tf.Variable(tf.zeros([n_hidden]))
D_W2=tf.Variable(tf.random_normal([n_hidden,1],stddev=0.01))
D_b2=tf.Variable(tf.zeros([1]))

#생성자
def generator(noise, labels):
    with tf.variable_scope('generator'):
        # noise 값에 labels 정보를 추가합니다.
        inputs = tf.concat([noise, labels], 1) #noise 값에 label 추가

        # TensorFlow 에서 제공하는 유틸리티 함수를 이용해 신경망을 매우 간단하게 구성할 수 있습니다.
        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input,
                                 activation=tf.nn.sigmoid)
    return output

#구분자
def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        # 노이즈에서 생성한 이미지와 실제 이미지를 판별하는 모델의 변수를 동일하게 하기 위해,
        # 이전에 사용되었던 변수를 재사용하도록 합니다.
        if reuse:
            scope.reuse_variables()

        inputs = tf.concat([inputs, labels], 1)

        hidden = tf.layers.dense(inputs, n_hidden,
                                 activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, # 판별기의 최종 결과값은 얼마나 진짜와 가깝냐를 판단하는 한 개의 스칼라값입니다.
                                 activation=None) #활성화 함수를 사용하지 않은 이유는 뒤에 나옴

    return output

#노이즈 생성
###############################################
def get_noise(batch_size, n_noise):
    return np.random.uniform(size=(batch_size, n_noise))#uniform : 균등 분포 -1과 1사이에 랜덤값 추출
###############################################
'''
# 노이즈를 이용해 랜덤한 이미지를 생성합니다.
G = generator(Z)
# 노이즈를 이용해 생성한 이미지가 진짜 이미지인지 판별한 값을 구합니다.
D_gene = discriminator(G)
# 진짜 이미지를 이용해 판별한 값을 구합니다.
D_real = discriminator(X)
'''
G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)

#손실함수 생성
###############################################
loss_D_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_real, labels=tf.ones_like(D_real)))  # ones_like = 1에 가깝게
loss_D_gene = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_gene, labels=tf.zeros_like(D_gene))) #zero_like = 0에 가깝게

# loss_D_real 과 loss_D_gene 을 더한 뒤 이 값을 최소화 하도록 최적화합니다.
loss_D = loss_D_real + loss_D_gene #loss_D가 1에 가까워질 수록 실제 이미지로 판별
# 가짜 이미지를 진짜에 가깝게 만들도록 생성망을 학습시키기 위해, D_gene 을 최대한 1에 가깝도록 만드는 손실함수입니다.
loss_G = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=D_gene, labels=tf.ones_like(D_gene)))
###############################################

#변수 생성
###############################################
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                           scope='generator')

train_D = tf.train.AdamOptimizer(learning_rate).minimize(loss_D,
                                                         var_list=vars_D)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(loss_G,
                                                         var_list=vars_G)
###############################################

#신경만 모델 학습
###############################################
sess=tf.Session()
saver = tf.train.Saver(tf.global_variables())
cpkt=tf.train.get_checkpoint_state('./GAN')
if cpkt and tf.train.checkpoint_exists(cpkt.model_checkpoint_path):
    saver.restore(sess,cpkt.model_checkpoint_path)
    print("Load sess")
else:
    sess.run(tf.global_variables_initializer())

    for epoch in range(12): #세대 학습
        for i in range(100):
            batch_xs,batch_ys=batch(train_list, batch_size)
            batch_xs=np.reshape(batch_xs,[-1,10000])
            noise=get_noise(batch_size,n_noise)


            _,loss_val_D=sess.run([train_D,loss_D],feed_dict={X:batch_xs,Y: batch_ys,Z:noise})
            _,loss_val_G=sess.run([train_G,loss_G],feed_dict={Y: batch_ys,Z:noise})

        print('Epoch:', '%04d' % epoch,
              'D loss: {:.4}'.format(loss_val_D),
              'G loss: {:.4}'.format(loss_val_G))

        if epoch==0 or (epoch)%2==0:
            sample_size=10
            noise = get_noise(sample_size, n_noise)
            samples=sess.run(G,feed_dict={Y: batch_ys[:sample_size],Z:noise})

            fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 1))

            for i in range(sample_size):
                ax[0][i].set_axis_off()
                ax[1][i].set_axis_off()

                ax[1][i].imshow(np.reshape(batch_xs[i], (100, 100)))
                ax[1][i].imshow(np.reshape(samples[i],(100,100)))

            plt.savefig('./samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

print("End")
