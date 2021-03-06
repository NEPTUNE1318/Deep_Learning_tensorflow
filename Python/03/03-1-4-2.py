# 선형 회귀 분석

# 라이브러리 호출
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# weather.csv 파일 불러오기
dataset = pd.read_csv('Data/Chap3/weather.csv')

# 데이터 간 관계를 시각화로 표헌
dataset.plot(x='MinTemp', y='MaxTemp', style='o')
plt.title('MinTemp vs MaxTemp')
plt.xlabel('MinTemp')
plt.ylabel('MaxTemp')
plt.show()

# 데이터를 독립 변수와 종속 변수로 분리하고 선형 회귀 모델 작성
X = dataset['MinTemp'].values.reshape(-1, 1)
y = dataset['MaxTemp'].values.reshape(-1, 1)
# 데이터의 80%를 훈련 데이터셋으로 하고
# 데이터의 20%를 검증 데이터셋으로 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

regressor = LinearRegression() # 선형 회귀 클래스를 가져옴
regressor.fit(X_train, y_train) # fit() 메서드를 사용하여 모델 훈련

# 회귀 모델에 대한 예측
y_pred = regressor.predict(X_test)
df = pd.DataFrame({'Actual' : y_test.flatten(), 'Predicted': y_pred.flatten()})
print(df)

# 검증 데이터셋을 사용한 회귀선 표현
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()

# 선형 회귀 모델 평가
print('평균제곱법 : ', metrics.mean_squared_error(y_test, y_pred))
print('루트 평균제곱법 : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))