# 로지스틱 회귀 분석

# 라이브러리 호출 및 데이터 준비
from sklearn.datasets import load_digits
from yaml import load
digits = load_digits() # 숫자 데이터셋은 사이킷런에서 제공

# digits 데이터셋의 형태(이미지가 1797개 있으며, 
# 8 × 8 이미지의 64차원을 가짐)
print("Image Data Shape", digits.data.shape)
# 레이블 이미지 1797개가 있음
print("Label Data Shape", digits.target.shape)

# digits 데이터셋의 시각화
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20,4))
# 예시로 이미지 다섯 개만 확인
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
    plt.subplot(1, 5, index+1)
    plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
    plt.title('Training : %i\n' % label, fontsize=20)
plt.show()

# 훈련과 검증 데이터셋 분리 및 로지스틱 회귀 모델 생성
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target,
                                                    test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression() # 로지스틱 회귀 모델의 인스턴스 생성
logisticRegr.fit(x_train, y_train) # 모델 훈련

# 일부 데이터를 사용한 모델 예측
# 새로운 이미지(검증 데이터)에 대한
# 예측 결과를 넘파이 배열로 출력
logisticRegr.predict(x_test[0].reshape(1, -1))
# 이미지 열 개에 대한 예측을 한 번에 배열로 출력
print("이미지 열 개에 대한 예측: ",logisticRegr.predict(x_test[0:10]))

# 전체 데이터를 사용한 모델 예측
predictions = logisticRegr.predict(x_test) # 전체 데이터셋에 대한 예측
score = logisticRegr.score(x_test, y_test) # 스코어 메서드를 사용한 성능 측정
print("score : ", score)

# 혼동 행렬 시각화
import numpy as np
import seaborn as sns
from sklearn import metrics

cm = metrics.confusion_matrix(y_test, predictions) # 혼동 행렬
plt.figure(figsize=(9, 9))
# heatmap으로 표현
sns.heatmap(cm, annot=True, fmt='.3f', linewidths=.5, square=True, cmap='Blues_r')
plt.ylabel('Actual label') # y축
plt.xlabel('Predicted label') # x축
all_sample_title = 'Accuracy Score : {0}'.format(score)
plt.title(all_sample_title, size=15)
plt.show()