import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import graphviz
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import numpy as np
import matplotlib.pyplot as plt

wine = load_wine()
wine.keys()

wine_pd = pd.DataFrame(data=np.c_[wine['data'], wine['target']],
                       columns=wine['feature_names'] + ['target'])

#이 줄의 코드는 절대 수정하지마시오
train_set, test_set = train_test_split(wine_pd, test_size=0.2, random_state=42)
##########################################

tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=1)

features = list(train_set.columns[:13])

X_train = train_set[features]
Y_train= train_set['target']
tree_clf.fit(X_train, Y_train)


X_test = test_set[features]
y_test = test_set['target']


export_graphviz(tree_clf, out_file='tree.dot',

                class_names=wine.target_names,

                feature_names=wine.feature_names,

                impurity=False, # gini 미출력

                filled=True) # filled: node의 색깔을 다르게

with open('tree.dot') as file_reader:

    dot_graph = file_reader.read()

dot = graphviz.Source(dot_graph) # dot_graph의 source 저장

dot.render(filename='tree.png') # png로 저장

n_feature = wine.data.shape[1]

idx = np.arange(n_feature)


feature_imp = tree_clf.feature_importances_

print('{}'.format(feature_imp))
plt.barh(idx, feature_imp, align='center')

plt.yticks(idx, wine.feature_names)

plt.xlabel('feature importance', size=15)

plt.ylabel('feature', size=15)

plt.show()


score_tr = tree_clf.score(X_train, Y_train)
score_te = tree_clf.score(X_test, y_test)
print('{:.3f}'.format(score_tr))
print('{:.3f}'.format(score_te))

### 아래의 코드는 수정하지마시오 ###
y_pred = tree_clf.predict(X_test)
print('------이 라인에 학번과 이름을 출력하세요')
print(classification_report(y_test, y_pred))
