import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns


iris=datasets.load_iris()
X_iris,y_iris,colum=iris.data, iris.target, iris.feature_names

#X : y의 데이터
#y : 속성 번호 [그룹 번호]
#colum :  속성 컬럼
#iris.target_names[y[0]] -> 속성에 따른
'''
print(X_iris)
print(y_iris)
print(colum)
'''

#Pandas 데이터 프레임으로 만들기 (컬럼별 데이터, 속성 컬럼)
df =pd.DataFrame(iris.data, columns=iris.feature_names)
#print(iris.feature_names)
#print(df[iris.feature_names])

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names],
                                                                            iris.target,
                                                                            test_size=0.25,
                                                                            stratify=iris.target,
                                                                            random_state=123456)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='entropy',
                                            n_estimators=100,
                                            n_jobs=4,
                                            oob_score=True,
                                            random_state=12).fit(X_train,y_train)


from sklearn.metrics import accuracy_score
predicted = rf.predict(X_test)
print("총 테스트 갯수 : %d, 오류 갯수 :%d"%(len(y_test),(y_test!=predicted).sum()))
print("정확도 : %.2f"%(accuracy_score(y_test,predicted)))


accuracy = accuracy_score(y_test, predicted)
print(f'Out-of-bag score estimate: {rf.oob_score_:.3}')



from sklearn.metrics import confusion_matrix
cm = pd.DataFrame(confusion_matrix(y_test, predicted),
                                                            columns=iris.target_names,
                                                            index=iris.target_names)
sns.heatmap(cm, annot=True)
plt.show()
