import numpy as np


# MSE(Mean Squared Error) : (오차)값이 작을수록  정답에 가깝다.
# yi는 신경망의 출력, ti는 정답 레이블(One-Hot인코딩)
def mean_squared_error(y, t):
    return ((y - t) ** 2).mean(axis=None)


# with axis=0 the average is performed along the row, for each column, returning an array
# with axis=1 the average is performed along the column, for each row, returning an array
# with axis=None the average is performed element-wise along the array, returning a single value

# def mean_squared_error(y, t):     # same code
#     return np.sum((y-t)**2)/y.shape[0]


# CEE(Cross Entropy Error) : (오차)값이 작을수록  정답에 가깝다.
# yi는 신경망의 출력, ti는 정답 레이블(One-Hot인코딩)
def cross_entropy_error(y, t):
    if 1 == y.ndim:  # y의 차원수가 1이면 1차원이면 2차원으로 변환
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    delta = 1e-7
    return -np.sum(np.log(y + delta) * t) / y.shape[0]


y = [0.15, 0.30, 0.45]
t = [[1, 0, 2], [0, 1, 0], [0, 0, -1]]

S = []
for i in range(len(t)):
    S.append(mean_squared_error(np.array(y), np.array(t[i])))

print(S)
print("np.min(S)   : " + str(np.min(S)))
print("np.argmin(S): " + str(np.argmin(S)))
print("np.max(S)   : " + str(np.max(S)))
print("np.argmax(S): " + str(np.argmax(S)))

# [0.33833333333333332, 0.23833333333333331, 0.13833333333333334]

# np.min(S)   : 0.138333333333  --> 오차가 가장 작다.
# np.argmin(S): 2               --> t가 정답지 0,1,2라 가정하면 인덱스 2 정답지일때 오차가 가장 적다.
# np.max(S)   : 0.338333333333  --> 오차가 가장 크다.
# np.argmax(S): 0               --> t가 정답지 0,1,2라 가정하면 인덱스 0 정답지일때 오차가 가장 크다.

print("---------------------------------------------")

S = []
for i in range(len(t)):
    S.append(cross_entropy_error(np.array(y), np.array(t[i])))

print(S)
print("np.min(S)   : " + str(np.min(S)))
print("np.argmin(S): " + str(np.argmin(S)))
print("np.max(S)   : " + str(np.max(S)))
print("np.argmax(S): " + str(np.argmax(S)))

# [1.8971193182194368, 1.2039724709926583, 0.79850747399557409]
# np.min(S)   : 0.798507473996  --> 오차가 가장 작다.
# np.argmin(S): 2               --> t가 정답지 0,1,2라 가정하면 인덱스 2 정답지일때 오차가 가장 적다.
# np.max(S)   : 1.89711931822   --> 오차가 가장 크다.
# np.argmax(S): 0               --> t가 정답지 0,1,2라 가정하면 인덱스 0 정답지일때 오차가 가장 크다.