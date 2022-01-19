import numpy as np
import tensorflow as tf

# 선형 회귀 기초 - 텐서플로우 케라스 확률적 경사 하강법(SGD)을 이용해 일한 시간 대비 번돈 데이터를 통해 8시간 일했을 때 벌 수 있는 돈 예측 프로그램
xData = [1, 2, 3, 4, 5, 6, 7] # 일한 시간
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000] # 번 돈

# 케라스 주요 데이터 구조는 모델로 Layer를 조직하는 방식, 가장 간단한 종류의 모델인 Sequntial 모델은 레이어를 선형적으로 쌓음
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1, input_dim = 1))  # 레이어 추가
sgd = tf.keras.optimizers.SGD(learning_rate = 0.01) # ir = 학습률 경사 하강에서 얼마만큼 내려가는지
model.compile(loss = 'mean_squared_error', optimizer = sgd)
model.fit(xData, yData, epochs = 2000)  # epochs = 반복 횟수
print(model.predict(np.array([8])))
