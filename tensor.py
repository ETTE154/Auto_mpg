# -*- coding: utf-8 -*-
#%%
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 자동차 연비 예측 모델 생성
# 1. 데이터 준비
# 2. 모델 구성
# 3. 모델 학습과정 설정
# 4. 모델 학습
# 5. 학습과정 살펴보기
# 6. 모델 평가
# 7. 모델 사용
#%%
dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

dataset_path
#%%
column_names = ['MPG','Cylinders','Displacement','Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names, na_values = "?",
                          comment='\t', sep=" ", skipinitialspace=True)

raw_dataset.tail()
# %%
# 데이터 정제
dataset = raw_dataset.copy()
dataset.isna().sum()
# %%
dataset = dataset.dropna()
dataset.isna().sum()
#%%
# Origin 열은 범주형이므로 원-핫 인코딩으로 변환
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
dataset.tail()
#%%
# 데이터 분할
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
#%%
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
#%%
train_stats = train_dataset.describe()  # 통계치
train_stats.pop("MPG")  # MPG 열 삭제
train_stats = train_stats.transpose()   # 전치
train_stats # 통계치
#%%
# 특성과 레이블 분리하기
train_labels = train_dataset.pop('MPG') # MPG 열을 레이블로 분리
test_labels = test_dataset.pop('MPG')   # MPG 열을 레이블로 분리


#%%
print(test_dataset.tail(), train_labels.tail())
# %%
# 데이터 정규화
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']   # 정규화 함수
normed_train_data = norm(train_dataset)     # 정규화
normed_test_data = norm(test_dataset)    # 정규화
print(normed_train_data.tail(), normed_test_data.tail())    # 정규화된 데이터 확인
#%%

# train_dataset.keys().shape    # 9
#%%
model = keras.Sequential([  
                          layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
                          layers.Dense(64, activation='relu'),
                        layers.Dense(1),
])

model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(0.001),     # 평균 제곱 오차, RMSprop 옵티마이저
              metrics=['mae', 'mse'])   # 평균 절대 오차, 평균 제곱 오차

model.summary()     # 모델 구조 확인
#%%
example_batch = normed_train_data[:10]  # 10개의 샘플을 추출
example_result = model.predict(example_batch)   # 예측
example_result  # 예측 결과
#%%
# %%
class PrintDot(keras.callbacks.Callback): # 콜백함수 정의
    def on_epoch_end(self, epoch, logs):    # epoch마다 호출됨
        if epoch % 100 == 0: print('')  # 100번마다 줄바꿈
        print('.', end='')  # 100번마다 . 출력
# %%
history = model.fit(normed_train_data,  # 입력 데이터
                    train_labels,   # 정답
                    epochs=200,     # 200번 반복
                    validation_split=0.2,   # 20% 검증용 데이터
                    verbose=0,    # 학습과정 출력 안함
                    callbacks=[PrintDot()]) # 100번마다 . 출력
# %%

hist = pd.DataFrame(history.history)    # history를 DataFrame으로 변환
hist['epoch'] = history.epoch   # epoch 열 추가
hist.tail() # 마지막 5행 출력
# %%
import matplotlib.pyplot as plt

def plot_history(history):  # history를 그래프로 출력하는 함수
    hist = pd.DataFrame(history.history)    # history를 DataFrame으로 변환
    hist['epoch'] = history.epoch   # epoch 열 추가

    plt.figure()    # 그래프 출력
    plt.xlabel('Epoch') # x축 이름
    plt.ylabel('Mean Abs Error [MPG]')  # y축 이름
    plt.plot(hist['epoch'], hist['mae'], label='Train Error')   # 훈련 오차
    plt.plot(hist['epoch'], hist['val_mae'], label = 'Val Error')   # 검증 오차
    plt.ylim([0,5]) # y축 범위
    plt.legend()    # 범례 출력

    plt.figure()    # 그래프 출력
    plt.xlabel('Epoch') # x축 이름
    plt.ylabel('Mean Square Error [$MPG^2$]')  # y축 이름
    plt.plot(hist['epoch'], hist['mse'], label='Train Error')   # 훈련 오차
    plt.plot(hist['epoch'], hist['val_mse'], label = 'Val Error')   # 검증 오차
    plt.ylim([0,20])    # y축 범위
    plt.legend()    # 범례 출력
    plt.show()  # 그래프 출력

# %%

plot_history(history)   # history를 그래프로 출력   

# %%
# 100번 이상 반복하면 검증 오차가 증가함