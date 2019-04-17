"GAN 모델 구현"
import tensorflow as tf
import matplotlib.pylot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


mnist=input_data.read_data_sets("../mnist/data/",one_hot=True)

#실제  숫자 이미지 크기는 28*28 = 784개의 특징을 가짐
X=tf.placeholder(tf.float32,[None,784])

#가짜 이미지를 위해 필요한 노이즈의 크기는 128이다. 노이즈를 생성자로 하여금 가짜 이미지로 생성한다!
Z=tf.placeholder(tf.float32, [None,128])


#b = bias
#W = Weight

#생성자 : 128(노이즈) -> 256 (은닉층) -> 784(입력층)
G_W1=tf.Variable(tf.random_normal([128,256],stddev=0.01))
G_b1= tf.Variable(tf.zeros([256]))
G_W2=tf.Variable(tf.random_normal([256,784],stddev=0.01))
G_b2=tf.Variable(tf.zeros([784]))


#구분자 : 784(입력층) -> 256(은닉층) -> 0~1 (일치도)
D_W1=tf.Variable(tf.random_normal([784,256],stddev=0.01))
D_b1=tf.Variable(tf.zeros([256]))
D_W2=tf.Variable(tf.random_normal([256,1],stddev=0.01))
D_b2=tf.Variable(tf.random_normal([1]))



#무작위 노이즈 생성 함수
def get_noise(batch_size,noise):
    return np.random.normal(size=(batch_size,noise))

#생성자 객체 생성하는 함수
def generator(noise):
    #신경망 활성화 함수
    hidden=tf.nn.relu(tf.matmul(noise, G_W1)+G_b1) #히든 생성
    output=tf.nn.sigmoid(tf.matmul(hidden,G_W2)+G_b2) #아웃풋 생성
    return output #가짜 이미지 생성


#구분자 객체 생성 함수
def discriminator(inputs):
    hidden=tf.nn.relu(tf.matmul(inputs,D_W1)+D_b1) #히든 생성
    output=tf.nn.sigmoid(tf.matmul(hidden,D_W2)+D_b2) #아웃풋 생성
    return output #일치도 값 생성



#가짜 이미지 생성자는 128크기의 노이즈에서 불러옴
G=generator(Z)

#가짜 이미지 구분자는 128크기의 노이즈가 생성한 784 크기의 이미지에서 불러옴
D_gene=discriminator(G)

#실제 이미지 구분자는 784 크기의 이미지에서 불러옴
D_real=discriminator(X)


#손실 함수
#구분자 손실 함수 : 진짜 이미지에 1에 가깝고, 가짜 이미지가 0에 가깝도록함 / 생성기로 만들어낸 가짜 이미지 판단하기 위함
loss_D=tf.reduce_mean(tf.log(D_real)+tf.log(1-D_gene))
#생성자의 손실 함수 : 가짜 이미지가 1에 가깝도록 함 / 실제 이미지로 판독하기 위함
loss_G=tf.reduce_mean(tf.log(D_gene))

#GAN 모델 학습 준비
#구분자는 구분자 가중치 및 바이어스만을 사용 / loss_D를 구할 때는 판별기 신경망에 사용되는 변수만 사용
D_var_list=[D_W1,D_b1,D_W2,D_b2]
#생성자는 생성자 가중치 및 바이어스만을 사용 / loss_G를 구할 때는 생성기 신경망에 사용되는 변수만 사용하여 최적화
G_var_list=[G_W1,G_b1,G_W2,G_b2]


#loss를 극대화 해야하지만, minimize 하는 함수이기 때문에, 최적화 하려는 loss_D, loss_G에 음수 부호를 붙인다.
#구분자 최적화 진행
train_D=tf.train.AdamOptimizer(0.001).minimize(-loss_D,var_list=D_var_list)
#생성자 최적화 진행
train_G=tf.train.AdamOptimizer(0.001).minimize(-loss_G,var_list=G_var_list)

#구분자 생성자 비용 변수 생성
loss_var_D,loss_val_G=0,0

#배치 크기 설정
batch_size=100
total_batch=int(mnist.train.num_examples/batch_size)

####신경망 학습 모델 및 이미지 결과 확인

#세션 생성
sess=tf.Session()
sess.run(tf.global_variables_initializer())

#100번 학습 진행
for epoch in range(total_batch):
    #전체 배치 크기 만큼 반복
    for i in range(total_batch):
        batch_x, batch_y=mnist.train.next_batch(batch_size)
        noise=get_noise(batch_size,128)
        #구분자는 실제 이미지 및 노이즈를 이용해 학습을 진행함.
        _, loss_val_D = sess.run([train_D,loss_D],feed_dict={X:batch_x,Z:noise})
        #생성자는 노이즈만을 이용해 학습 진행
        _, loss_val_G = sess.run([train_G,loss_G],feed_dict={Z:noise})

    #총 1번 돌 때마다 학습 상황 출력
    print("학습 : ","%04d"%epoch,
            "구분차 오차 : {:.4}".format(loss_val_D),
            "생성자 오차 : {:.4}".format(loss_val_G))

    #10 번 돌 때마다 결과를 그림으로 확인
    if epoch==0 or (epoch+1)%10==0:
        #샘플 이미지 크기 10
    size=10
    noise=get_noise(size,128)
    #생성자가 임의의 샘플 이미지를 생성하도록 한다.
    samples=sess.run(G,feed_dict={Z:noise})
    #만든 그림을 폴더에 출력
    fig, ax = plt.subplots(1, size, figsize=(size, 1))

    for i in range(size):
        ax[i].set_axis_off()
        #28*28 크기 이미지 생성
        ax[i].imshow(np.reshape(samples[i], (28, 28)))

    plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)
