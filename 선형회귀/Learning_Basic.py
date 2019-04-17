import tensorflow as tf


xData =[1,2,3,4,5,6,7]
yData =[25000,55000,75000,110000,128000,155000,180000]


W=tf.Variable(tf.random_uniform([1],-100,100))
#W = Weight
#random_uniform([1],-100,100  = -100부터100까지 특정 값을 1개 생성한다.

b=tf.Variable(tf.random_uniform([1],-100,100))
#b= bias


#X와 Y를 placeholder로 틀을 생성
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#가설 생성
H=W*X+b

#비용함수 생성
cost=tf.reduce_mean(tf.square(H-Y))
#square는 제곱이다. 즉 예측 값에 실제 값을 뺀 값을 제곱한다.
#reduce_mean : 평균값을 구한다.

#경사하강법에서 얼마만큼 점프할 지 정의 0.01은 스텝의 크기임
a=tf.Variable(0.01)

#텐서플로우에서 제공해주는 경사하강 라이브러리 (경사 하강 학습 라이브러리)
optimizer=tf.train.GradientDescentOptimizer(a)

#비용함수를 가장 적게 만드는 방향으로 학습
train=optimizer.minimize(cost)

#세션 생성 및 변수 초기화
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#학습 진행 5000번 반복
for i in range(5001):
    #실제 학습 진행
    #X,Y 데이터 매칭 시켜줌
    sess.run(train,feed_dict={X:xData,Y:yData})

    #500번에 한번씩 확인
    if  i%500==0:
        #인덱스와 각각의 변수들을 확인할 수 있도록 한다.
        #W=현재의 기울기
        #b=현재의 Y절편인 b값
        print(i,sess.run(cost,feed_dict={X:xData,Y:yData}),sess.run(W),sess.run(b))



#원하는 학습이 끝난 뒤 원하는 입력에 대한 결과 값을 출력해주도록 한다.
#8시간 일했을 때의 결과 값(매출값)
print("최종 결과 값 : {}" .format(sess.run(H,feed_dict={X:[8]})))
