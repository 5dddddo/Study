# Tensorflow 기초1 (07/10)

##  Gradient Descent (경사하강법) 

- 주위에서 경사가 가장 급한 곳(미분해서 기울기가 최대)으로 이동해서

  함수의 최솟값(해)을 찾을 때까지 앞의 과정을 반복하는 알고리즘

- cost 비용이 최소가 되도록 하는 최적의 parameter를 찾는 방법

- **learning_rate** : 경험적으로 적합한 값을 찾아야 함

  초기값으로 0.01을 주로 사용 

  경사가 급한 곳으로 얼마나 많이 이동할 것인지 정함

  learning_rate 작으면 조금 움직임 

  learning_rate 크면 많이 움직임

- **가중치 Weight** : 기울기

$$
W = W - \alpha \frac{d}{dW} cost(W)\\
\\
(\alpha = learning\;\;rate, 상수)
$$

![1562836838627](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1562836838627.png)

[^출처]: https://www.oreilly.com/library/view/learn-arcore-/9781788830409/e24a657a-a5c6-4ff2-b9ea-9418a7a5d24c.xhtml

- **Cost function , 손실 함수** 

  : 손실함수의 결과값(오차)을 가장 작게 만드는 것이 목표

$$
cost(W,b) = \frac{1}{n} \sum (H(x_1^i, x_2^i,x_3^i) - y^i)^2)
$$



- cost function이 2개 이상의 최솟값을 가지면

  local minima ( 지역 해 ) 에 빠져 

  global minima ( 진짜 해 ) 를 못 구하는 문제 발생할 수 있음

- 따라서, cost function이 convex function 형태일 때

  gradient descent algo 적용

- ##### 목표 : W가 1에 가깝고 b가 0에 가까워 지도록 W와 b를 결정하는 것!



## Linear regression (선형 회귀)

: 입력 데이터 x와 label화 된 y의 선형 상관관계를 모델링하는 회귀 분석 기법

- 하나의 입력 데이터를 가지면 single linear regression
- 여러 개의 입력 데이터를 가지면 multi linear regression

- 입력 data set X가 여러 개이이기 때문에 W의 값도 여러 개 

$$
H(x_1,x_2,x_3) = w_1x_1+w_2x_2+w_3x_3+b = Wx+b
$$

- matrix로 표현할 때는 관용적으로 X가 W보다 먼저 위치

$$
\begin{equation}
  \begin{bmatrix}
    x_1& x_2& x_3 \\ 
  \end{bmatrix}
    \begin{bmatrix}
    W_1\\W_2\\W_3\\ 
  \end{bmatrix}
\end{equation}
= (x_1w_1 + x_2w_2+x_3w_3)\\
H = XW + b
$$



- 일반적으로 최소제곱법 / 손실 함수를 최소화 하는 방식으로 모델링

- ##### training data에 가장 근접한 Hypothesis를 만드는 것이 목표!

  ##### W가 1에 가깝고 b가 0에 가까워 지도록 W와 b를 결정

  W와 b, H 모두 Node

  

### 선형회귀 모델링 순서

1. training data set 준비

2. Hypothesis에 이용할 변수들 정의 & 초기화

3. Hypothesis 정의

4. cost(loss) function 정의

5. cost가 최소가 되도록 돕는 node 생성 

   => tf.train.GradientDescentOptimizer

6. cost function을 최소화 하기 위해 정의
   => train = optimizer.minimize(cost)

7. 그래프를 실행 시키기 위한 Session(runner) 생성과

   global variable(W,b) 초기화  작업

8. 학습 수행

9. prediction 테스트

   

``` python
import tensorflow as tf

## 1. training data set 준비
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

# 실제 training data set
x_data = [1,2,3,4]
y_data = [4,7,10,13]

## 2. Hypothesis에 이용할 변수들 정의 & 초기화
# 1000, 5 ,... 등 수를 지정해 주면 train시 문제 발생
# 따라서, 0과 가까운 랜덤 값으로 초기화!
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

## 3. Hypothesis 정의
H = W * x + b

## 4. cost(loss) function 정의 
# 실제 데이터에 근접하는 Hypothesis를 찾아주는 함수 정의
cost = tf.reduce_mean(tf.square(H-y))

## 5. cost가 최소가 되도록 돕는 node 생성 
#     => tf.train.GradientDescentOptimizer
# tensorflow에서 복잡한 계산을 줄일 수 있게 node 제공

optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)

## 6.cost function을 최소화 하기 위해 정의
train = optimizer.minimize(cost)

# 그래프를 생성함
======================================================

## 7. 그래프를 실행 시키기 위한 runner 생성과
## global variable (W,b) 초기화  작업

sess = tf.Session()
sess.run(tf.global_variables_initializer())

======================================================
## 8. 학습 수행

for step in range(3000):
    # sess.run(train)만 해도 되지만
    # cost값을 최소화하려는 의도대로 
    # 학습이 잘 수행되고 있는지 확인하려고 같이 실행
    _, cost_val = sess.run([train,cost],
                           feed_dict={x:x_data,y:y_data})
    if step % 300 == 0:
        print("{}".format(cost_val))

======================================================
# 실제 test
## 9. prediction : 예측 모델
print(sess.run(H,feed_dict={x:[300]}))

20.295454025268555
0.016762500628829002
0.002773507498204708
0.0004588987212628126
7.593041664222255e-05
1.2565705219458323e-05
2.080663307424402e-06
3.450272743066307e-07
5.730441898776917e-08
9.680263701739023e-09
[900.9902]
```

``` python
import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
# 그래프 그려주는 python의 library
import matplotlib.pyplot as plt

warnings.filterwarnings(action = "ignore")
## 파일 불러오기 => pandas 이용하는 것이 가장 효율적

df = pd.read_csv("./data/ozone/ozone.csv",sep=",")
display(df)


## 데이터 정제/분류/생성 => Pandas에서 수행 + numpy

# 온도에 따른 오존량 예측
# 필요한 col 2개만 일단 추출 => fancy indexing

df2 = df[["Ozone","Temp"]]
# 결치값 제거
# dropna(how = "all") : Ozone과 Temp 행이 모두 NaN인 행 지우기
# dropna(how = "any") : Ozone과 Temp 행 중 하나라도 NaN이면 행 지우기

df3= df2.dropna(how = "any", inplace = False )
print(df2.shape)
print(df3.shape)

# (153, 2)
# (116, 2)

# 이렇게 준비한 데이터가 linear한 데이터인지 확인
# scatter : 데이터의 분포, 산전도 그려줌
plt.scatter(df3["Temp"],df3["Ozone"])
plt.show()
```

![1562839006127](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1562839006127.png)

```python
# placeholder
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

# training data set
x_data = df3["Temp"] # Series
y_data = df3["Ozone"]

# Weight & bias
W = tf.Variable(tf.random_normal([1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# Hypothesis
H = W*x +b

# cost function
cost = tf.reduce_mean(tf.square(H-y))

# train node 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.1)
train = optimizer.minimize(cost)

# 그래프 시
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 (train)
for step in range(3000):
    _, cost_val = sess.run([train, cost],
                          feed_dict = {x:x_data,y:y_data})
    if step % 300 == 0:
        print("cost : {}".format(cost_val))
        
## 문제
# cost 값이 발산함 => 해 X
# x_data(Ozone)와 y_data(Temp)의 값의 차가 커서 error
# cost가 0으로 가까워져야 함!

## 해결
# 정규화를 통해 x_data와 y_data의 차이값을 0~1 사이의 값으로 변환

'''
cost : 1414.1903076171875
cost : nan
cost : nan
cost : nan
cost : nan
cost : nan
cost : nan
cost : nan
cost : nan
cost : nan
'''
```

- 정규화

  데이터 정제 : shrink

  1. normalization : (요소값 - 최솟값) / (최댓값 - 최솟값)

  (요소값 - 최솟값) < (최댓값 - 최솟값)

  0 < (요소값 - 최솟값) / (최댓값 - 최솟값) < 1

  2. standardization : (요소값 - 평균) / 표준편차



``` python
# placeholder
x = tf.placeholder(dtype = tf.float32)
y = tf.placeholder(dtype = tf.float32)

# training data set
## 데이터 정제 : 정규화
x_data = (df3["Temp"] -  df3["Temp"].min())/ (df3["Temp"].max()-df3["Temp"].min())
y_data = (df3["Ozone"] -  df3["Ozone"].min())/ (df3["Ozone"].max()-df3["Ozone"].min())

# Weight & bias
W = tf.Variable(tf.random_normal([1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# Hypothesis
H = W*x +b

# cost function
cost = tf.reduce_mean(tf.square(H-y))

# train node 생성
optimizer = tf.train.GradientDescentOptimizer(learning_rate =0.1)
train = optimizer.minimize(cost)

# 그래프 실행 & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습 (train)
for step in range(3000):
    _, cost_val = sess.run([train, cost],
                          feed_dict = {x:x_data,y:y_data})
    if step % 300 == 0:
        print("cost : {}".format(cost_val))
        
## 문제
# cost 값이 발산함 => 해 X
# x_data(Ozone)와 y_data(Temp)의 값의 차가 커서 error
# cost가 0으로 가까워져야 함!

## 해결
# 정규화를 통해 x_data와 y_data의 차이값을 0~1 사이의 값으로 변환
import tensorflow as tf
# training data set
# x_data , y_data의 속성
# 데이터가 많아지면 행만 변화하고 열은 변하지 않음!
# shape = [None,3]
x_data = [[73,80,75],
          [93,88,93],
          [89,91,90],
          [96,98,100],
          [73,66,70]]
y_data = [[152],[185],[180],[196],[142]]

# placeholder
X = tf.placeholder(shape=[None,3],dtype=tf.float32)
Y = tf.placeholder(shape=[None,1],dtype=tf.float32)

# Weight & bias
# W = [3,1]

W = tf.Variable(tf.random_normal([3,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# Hypothesis
# H = XW + b => 행렬 곱
H = tf.matmul(X,W)+b

# cost function
cost = tf.reduce_mean(tf.square(H - Y))

# 학습 노드 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3000):
    _,cost_val=sess.run([train,cost],
                       feed_dict={X:x_data,Y:y_data})
    if step % 300 == 0:
        print(cost_val)
'''
24220.01
nan
nan
nan
nan
nan
nan
nan
nan
nan
'''

import tensorflow as tf
import numpy as np
import pandas as pd
import warnings
# pytho
# pip install sklearn
# 0~1 사이의 값으로 scaling해주는 library
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings(action="ignore")

## Data loading
df = pd.read_csv("./data/ozone/ozone.csv",sep=",")
## 필요한 컬럼만 추출 : 2개의 열 지우기
df.drop(["Month","Day"],axis = 1, inplace= True)
## 결측값 제거
df.dropna(how ="any",inplace=True)

## x 데이터 추출 : dataframe type
df_x = df.drop("Ozone",axis = 1,inplace =False)
## y 데이터 추출 : series type
df_y = df["Ozone"]

# training data set : 데이터 정제
# values : 2차원 numpy array로 추출
#  MinMaxScaler().fit_transform() : 0~1 사이의 값으로 scaling
x_data = MinMaxScaler().fit_transform(df_x.values)
y_data = MinMaxScaler().fit_transform(df_y.values.reshape(-1,1))

# --------------------------------------------------------------
# tensorflow

# placeholder
X = tf.placeholder(shape=[None,3],dtype=tf.float32)
Y = tf.placeholder(shape=[None,1],dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random_normal([3,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# Hypothesis
# H = XW + b => 행렬 곱
H = tf.matmul(X,W)+b

# cost function
cost = tf.reduce_mean(tf.square(H - Y))

# 학습 노드 생성
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(30000):
    _,cost_val=sess.run([train,cost],
                           feed_dict={X:x_data,Y:y_data})
    if step % 30000 == 0:
        print(cost_val)
        
# prediction
# X : 2차원 배열의 1행 3열로 넘겨줘야 함
sess.run(H,feed_dict = {X:[[190,7.4,67]]})

'''
0.4278693
array([[46.176838]], dtype=float32)
'''
```

