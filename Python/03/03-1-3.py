# 라이브러리 호출 및 데이터 준비
import pandas as pd
from torch import rand

# 판다스를 이용하여 train.csv 파일을 로드해서 df에 저장
df = pd.read_csv('Data/Chap3/titanic/train.csv', index_col='PassengerId')
print(df.head()) # train.csv 데이터의 상위 행 다섯 개를 출력

# 데이터 전처리
# 승객의 생존 여부를 예측하려고 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare' 사용
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
# 성별을 나타내는 'sex'를 0 또는 1의 정수값으로 변환
df['Sex'] = df['Sex'].map({'male' : 0, 'female' : 1})
df = df.dropna() # 값에 없는 데이터 삭제
X = df.drop('Survived', axis=1)
y = df['Survived'] # 'Survived' 값을 예측 레이블로 사용

# 훈련과 검증 데이터셋으로 분리
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 결정 트리 모델 생성
from sklearn import tree
model = tree.DecisionTreeClassifier()

# 모델 훈련
model.fit(X_train, y_train) # 모델을 훈련시킵니다.

# 모델 예측
y_predict = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("정확도 : ", accuracy_score(y_test, y_predict)) # 검증 데이터에 대한 예측 결과를 보여 줍니다

# 혼동 행렬을 이용한 성능 측정
from sklearn.metrics import confusion_matrix
confusion = pd.DataFrame(
    confusion_matrix(y_test, y_predict),
    columns=['Predicted Not Survival', 'Predicted Survival'],
    index = ['True Not Survival', 'True Survival']
)

print(confusion)