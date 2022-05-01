# 서포트 벡터 머신

# 라이브러리 호출
from sklearn import svm, metrics, datasets, model_selection
import tensorflow as tf
import os

# iris 데이터를 준비하고 훈련과 검증 데이터셋으로 분리
iris = datasets.load_iris() # 사이킷런에서 제공하는 iris 데이터 호출
# 사이킷런의 model_selection 패키지에서 제공하는 train_test_split 메서드를 활용하여
# 훈련셋과 검증셋으로 분리
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    iris.data, iris.target, test_size=0.6, random_state=42
)

# SVM 모델에 대한 정확도
svm = svm.SVC(kernel='linear', C=1.0, gamma=0.5) # SVM이 지원하는 선형, 비선형 분류 증 선형 커널을 사용
svm.fit(X_train, y_train) # 훈련 데이터를 사용하여 SVM 분류기를 훈련
predictions = svm.predict(X_test) # 훈련된 모델의 사용하여 검증 데이터에서 예측
score = metrics.accuracy_score(y_test, predictions)
print('정확도 : {0:f}'.format(score)) # 검증 데이터 (예측) 정확도 측정