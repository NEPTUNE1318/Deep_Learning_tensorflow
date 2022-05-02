# K-평균 군집화

# 라이브러리 호출
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 상품에 대한 연 지출 데이터 호출
data = pd.read_csv('Data/Chap3/sales data.csv')
#print(data.head())

# 연속형 데이터와 명목형 데이터로 분류
categorical_features = ['Channel', 'Region'] # 명목형 데이터
continuous_features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper',
                       'Delicassen'] # 연속형 데이터

for col in categorical_features:
    # 명목형 데이터는 판다스의 get_dummies() 메서드를 사용하여 바이너리로 변환
    dummies = pd.get_dummies(data[col], prefix=col)
    data = pd.concat([data, dummies], axis=1)
    data.drop(col, axis=1, inplace=True)
print(data.head())

# 데이터 전처리(스케일링 적용)
mms = MinMaxScaler()
mms.fit(data)
data_transformed = mms.transform(data)

# 적당한 K 값 추출
Sum_of_squared_distances = []
K = range(1, 15) # K에 1부터 15까지 적용
for k in K:
    km = KMeans(n_clusters=k) # 1~15의 K 값 적용
    km = km.fit(data_transformed) # KMeans 모델 훈련
    Sum_of_squared_distances.append(km.inertia_)
    
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Optimal k')
plt.show()

