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


```

