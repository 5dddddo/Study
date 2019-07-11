# Tensorflow 기초1 (07/10)

Linux => 추후 하둡 시스템을 사용하기 위해

​				간단한 형태로 시스템 명령어 위주로 학습

python => 기본 언어 특성

numpy => ndarray 자료구조와 여러 가지 함수 사용

pandas => 탐색적 데이터 분석을 하기 위한 python module

machine learning (기계 학습) => 데이터 분석의 또 다른 방법



## 머신러닝 분류
#### Supervised Learning 
- training set 이라고 불리는 **lable화 된 데이터**를 통해 학습 후 

  **예측 모델**을 생성 => 예측모델을 이용해 실제 데이터의 예측값 추출

- Linear Hypothesis : 2차원에서 Linear Hypothesis 를 정의하는 것은 training data set에 잘 맞는 linear한 선을 긋는 것으로 생각할 수 있음 ->  Linear Hypothesis을 수정해 나가면서 데이터에 가장 근접한 선이 되도록 직선의 기울기와 절편을 바꾸는 과정이 "**학습**" 
  

    - Linear regressing( 선형 회귀 ) 

      : 추측한 결과값이 선형값 ( 값의 분포에 제한이 없음 )

       ex) 시험 성적 : 0~100점

      

  - Logistic regression 

    추측한 결과값이 논리값 ( 0 / 1 )

    ex) 시험 성적 : 합격 / 불합격  

    

  - Multinomial Classification 

    : 추측한 결과값이 정해져 있는 몇 개의 boundary 중 1개

    ex) 시험 성적 : A, B, C, D, F 학점



#### Unsupervised Learning
- **lable화 되지 않은 데이터**를 통해 학습 후

  비슷한 데이터끼리 모아 **Clustering** 모델을 생성

- News Grouping 같은 경우

- 데이터를 통해 스스로 학습



## Tensorflow 
- google이 만든 machine library로 open source library
  
- 수학적 계산을 하기 위한 library
  
- data flow **graph**를 이용
  
- tensorflow를 이용해 그릴 수 있는 data flow graph는 

  **Node와 Edge로 구성된 방향성 있는 그래프**
- **Node** : 데이터의 입출력과 수학적 연산 담당
- **Edge** : 데이터를 Node로 실어 나르는 역할
- **Tensor** : 동적 크기의 다차원 배열을 지칭



### Tensorflow의 자료형
#### 1. constant : 상수 tensor를 생성하는 node
- decode() : byte -> str로 casting
   global variable(W,b) 초기화  작업

``` python
# constant() : 상수 Node 생성
my_node = tf.constant("Hello World")
sess = tf.Session()

print(sess.run(my_node)) 
# b'Hello World'
# byte type
print(sess.run(my_node).decode()) 
# Hello World
# str type
# decode() : byte -> str로 casting
```



#### Session : 그래프를 실행시키기 위한 runner 역할

- constant() 시 그래프는 생성되어 메모리에 올라갔지만

  연산이 수행되기 전 상태

  => 실행시켜야 연산이 수행됨

  => 그래프를 실행시키기 위해 runner 역할을 하는 Session 객체가 필요함

  => sess = tf.Session() 을 생성

  => sess.run(해당 노드/해당 그래프)시 **해당 노드, 해당 그래프만** 실행함



- ##### 복수 개의 node를 실행하려면 배열 형태로 입력해야 함

  => 결과값도 배열 형태
  
  

``` python
node1 = tf.constant(10, dtype = tf.float32)
node2 = tf.constant(20, dtype = tf.float32)

## 현재 그래프만 만들어진 상태이고 메모리에 올라가기만 함
## 실행 전에는 node3의 값 X 
## 실행시켜야 연산이 수행되면서 node3 의 값이 30이 됨
node3 = node1 + node2

## 그래프를 실행시키기 위해 runner 역할을 하는
## Session 객체가 있어야 함
## tf.Session이 있어야 위의 그래프를 실행할 수 있음
sess = tf.Session()  

## 해당 노드, 해당 그래프만 실행됨
## sess.run(해당노드/그래프)

## node1만 실행 됨 
#sess.run(node1)
## node2만 실행 됨 
#sess.run(node2)

## 연산이 수행되며 node3의 값이 들어감
print(sess.run(node3)) # 30.0

## 복수개의 node를 실행시키려면 
## 배열 형태로 입력해야 함 -> 결과도 배열 형태로 출력
print(sess.run([node1,node2,node3])) # [10.0, 20.0, 30.0]
```



- **casting (변수, dtype = 바꾸려는 type)** : 변수의 type 바꾸는 함수
  

``` python
import tensorflow as tf
node1= tf.constant([10,20,30],dtype = tf.int32)
print(node1) 
# [10,20,30]
# Tensor("Const_33:0", shape=(3,), dtype=int32)

node1= tf.cast(node1,dtype = tf.float32)
print(node2) 
# [10. 20. 30.]
# Tensor("Cast_9:0", shape=(3,), dtype=float32)
```



#### 2. Placeholder : 입력을 받아들이는 저장 공간 node

- 입력을 받아들이는 저장공간

  x = tf.placeholder(dtype = tf.float32)
  y = tf.placeholder(dtype = tf.float32)

- **실행 시 sess.run(변수, feed_dict = 값 ) 으로 반드시 값 주입해야 함**

- 2개의 수를 입력으로 받아서 더하는 프로그램

``` python
import tensorflow as tf
node1 = tf.placeholder(dtype=tf.float32)
node2 = tf.placeholder(dtype=tf.float32)
node3 = node1 + node2

sess=  tf.Session()
# node1과 node2의 값 dictionary 형태로 주입
result = sess.run(node3,feed_dict = {node1:10,node2:20})
print(result)
```



#### 3. Variable : 계속 변하는 값을 받아들여 변수처럼 동작하는 node

- Variable은 연산을 실행하기 전에 반드시 명시적으로 초기화해야 함

  => **tf.global_variables_initializer()**



###  선형 회귀 ( linear regression )

-  가장 큰 목표는 가설의 완성

  => 가설(hypothesis) : H(x) = Wx+b

``` python
import tensorflow as tf

# training data set
# x값이 1 일 때 y값이 1 이라는 label 값 가짐
x = [1,2,3]
y = [1,2,3]  # label

# 선형 회귀(linear regression)
# 가장 큰 목표는 가설의 완성
# 가설(hypothesis) = Wx+b
# W와 b 
# Weight & bias 정의
# Variable () : 계속 다른 값을 받아들이는, 변하는 값을 
# 받아들여 변수처럼 동작하는 node
# (tf.random_normal([1]), name="weight"):
# 정규 분포 내에서 난수 1차원 1개
# name="weight" : tensorflow가 사용할 이름 지정 
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# Hypothesis(가설)
# 우리의 최종 목적은 training data에 가장
# 근접한 hypothesis를 만드는 것(W와 b를 결정)
# 잘 만들어진 가설은 W가 1에 가깝고 b가 0에 가까워야 함
H = W*x + b
# W와 b, H 모두 Node


# cost(loss) function :
# 우리의 목적은 cost함수를 최소로 만드는 W,b 구하기
# tf.reduce_mean() : 평균 구하는 함수
cost = tf.reduce_mean(tf.square(H-y))

## cost function minimize
## python이 제공하는 GradientDescentOptimizer를 이용해
## 최소가 되는 값 찾기
## 기존 cost보다 더 작은 cost를 얻음

optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)


## 그래프 실행하기 위한 runner 생성
sess = tf.Session()
## global variable을 반드시 초기화 해줘야 함 (W,b...)
sess.run(tf.global_variables_initializer())

# 학습 진행
for step in range(3000):
    _, w_val, b_val, cost_val = sess.run([train,W,b,cost])
    if step%300 == 0:
        print("{},{},{}".format(w_val,b_val,cost_val))

[-0.00958644],[-0.51523536],8.985591888427734
[1.0127325],[-0.02894408],0.00012076189887011424
[1.0061849],[-0.01405989],2.8494876460172236e-05
[1.0030046],[-0.00682983],6.7243545345263556e-06
[1.0014597],[-0.00331826],1.5872219591983594e-06
[1.0007098],[-0.0016131],3.750964481241681e-07
[1.0003458],[-0.00078522],8.892008196426104e-08
[1.0001684],[-0.0003826],2.1092347424200852e-08
[1.000082],[-0.0001869],5.030217575807683e-09
[1.00004],[-9.1983624e-05],1.2182314046427223e-09        
        
```