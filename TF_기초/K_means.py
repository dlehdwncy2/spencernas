import tensorflow as tf
from sklearn.cluster import  KMeans
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib .pyplot as plt

#데이터 프레임 생성 x,y 좌표에서 활용 되도록 셋팅
df = pd.DataFrame(columns=['x','y'])

#데이터 삽입 30개
df.loc[0] =[2,3]
df.loc[1]=[2,11]
df.loc[2]=[2,18]
df.loc[3]=[4,5]
df.loc[4]=[4,7]
df.loc[5]=[5,3]
df.loc[6]=[5,15]
df.loc[7]=[6,6]
df.loc[8]=[6,8]
df.loc[9]=[6,9]
df.loc[10]=[7,2]
df.loc[11]=[7,4]
df.loc[12]=[7,5]
df.loc[12]=[7,17]
df.loc[13]=[7,17]
df.loc[14]=[7,18]
df.loc[15]=[8,5]
df.loc[16]=[8,4]
df.loc[17]=[9,10]
df.loc[18]=[9,11]
df.loc[19]=[9,15]
df.loc[20]=[9,19]
df.loc[21]=[10,5]
df.loc[22]=[10,8]
df.loc[23]=[10,18]
df.loc[24]=[12,6]
df.loc[25]=[13,5]
df.loc[26]=[14,11]
df.loc[27]=[15,6]
df.loc[28]=[15,18]
df.loc[29]=[18,12]


#데이터 프레임 헤드에 표현 시켜줌
print(df.head(30))








point=df.values #데이터 프레임의 값들을 Numpy 객체로서 선언
kmeans=KMeans(n_clusters=4).fit(point)  #정해준 데이터를 기반으로 K-means 알고리즘을 수행해서 총 클러스터 4개를 생성한다.
print(kmeans.cluster_centers_) #중심 값 확인
print(kmeans.labels_) #데이터 프레임 내 데이터들 소속 확인

#데이터프레임에 클러스터 속성을 만들어 준뒤 각각의 데이터에 대하여 클러스터 소속을 넣어준다.
df['cluster']=kmeans.labels_
print(df.head(30))


#데이터 시각화 설정된 데이터를 그래프 형태로 표현
sb.lmplot('x','y',
                        data=df, #위에서 정의한 데이터 프레임을 데이터로 정의
                        fit_reg=False,
                        hue="cluster", #클러스터 속성으로 분류함.
                        scatter_kws={"s":150}) #시각화를 표현할때 그래프의 '점' 크기를 100정도로 표현


plt.title("K-Means Example") #그래프 타이틀
plt.xlabel('x') #X 범주
plt.ylabel('y') #Y 범주
plt.show()