# Tensorflow 기초3 (07/12)

## Logistic regression

- y의 label이 논리값 (0 / 1)으로 산출

- Linear regression은 발산하는 입력 데이터를 처리하는데 부적합

  

#### Linear regression => Logistic regression 일 때,

#### 변화하는 3가지

1. Hypothesis 재정의

   binary (0~1 사이 값) data 형태의 label은

   linear regression으로 학습하면 반드시 Error 발생 

   => 범위가 너무 넓기 때문에 발산함

   => data에 따라 그래프가 변하고 

   ​     data에 따라 1보다 크거나 0보다 작은 값이 도출될 수 있음

   => 이런 data set은 Hypothesis를 직선으로 정할 수 없음

   => Hypothesis를 곡선으로 정해야 함

   => sigmoid 함수 이용 : H(가정)이 0~1 사이의 값으로 binding 되게 함

$$
H(x), sigmoid\;function = \frac{1}{1+e^{-z}}
$$



2. cost function 재정의

   => linear regression의 cost func은 지수 형태이기 때문에

   ​	 여러 local minimum을 가짐

   ![1563605464403](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1563605464403.png)

   => 문제) local minimum에 빠질 수 있음
   
   => 해결) 지수 형태의 함수에 log를 취하면 1개의 최솟값을 가지게 됨
   $$
   cost(H(x),y) = -y \log(H(x))-(1-y)\log(1-H(x))
$$
   
=> 목표) minimize cost 
   
   ​	 cost를 최소화하기 위해 gradient descent algo 이용



3. Accuracy 측정

   Linear regression에서는 accuracy 측정 불가

   Logistic regression부터 accuracy 측정이 가능해짐

------------------------------------------------------------------------------



### Logistic regression example - 시험 P/F 1

``` python
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
warnings.filterwarnings(action = "ignore")
# training data set
x_data = [1,2,5,8,10]
y_data = [0,0,0,1,1]

# placeholder
x = tf.placeholder(dtype=tf.float32)
y = tf.placeholder(dtype=tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Hypothesis
H = W*x+ b

# cost function
cost = tf.reduce_mean(tf.square(H-y))

# train node 생성
train = tf.train.GradientDescentOptimizer(learning_rate =0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train,cost], feed_dict={x:x_data,y:y_data})
    if step % 3000 == 0:
        print(cost_val)
    
# prediction
print(sess.run(H, feed_dict={x:[6]}))

```



### Logistic regression example - 시험 P/F 2

``` python
import tensorflow as tf

# training data set
x_data = [[30,0],[10,0],[8,1],[3,3],[2,3],[5,1],[2,0],[1,0]]
y_data = [[1],[1],[1],[1],[1],[0],[0],[0]]


# placeholder
X = tf.placeholder(shape = [None,2], dtype=tf.float32)
Y = tf.placeholder(shape = [None,1], dtype=tf.float32)

# Weight & bias
W = tf.Variable(tf.random_normal([2,1]),name="weight")
b = tf.Variable(tf.random_normal([1]),name="bias")


# Hypothesis
logits = tf.matmul(X,W)+b
# 지수 함수 형태의 H 정의
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train,cost],feed_dict = {X:x_data,Y:y_data})
    if step % 3000 == 0:
        print(cost_val)
        
## Accuracy
# H > 0.5 == 1 : True / H <= 0.5 == 0 : False
predict = tf.cast(H>0.5,dtype=tf.float32)    

## correct node
correct = tf.equal(predict,Y)
accuracy = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
print("정확도 : {}".format(sess.run(accuracy,feed_dict = {X:x_data,Y:y_data})))

# prediction
print(sess.run(H, feed_dict = {X:[[4,2]]}))

# plot
plt.scatter(x_data,y_data)
# w와 b가 node이기 때문에 sess.run()으로 실행시켜야
# node에 값이 할당 됨!
plt.plot(x_data,x_data*sess.run(W)+sess.run(b),"r")
plt.show()

```



### Logistic regression example - Titanic

```python
import tensorflow as tf
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings(action="ignore")

df = pd.read_csv("./data/titanic/train.csv")
data = df[["Survived","Pclass","Sex","Age","Fare"]]
data.dropna(how ="any",inplace=True)

data.loc[data["Sex"]=="male","Sex"] = 1
data.loc[data["Sex"]=="female","Sex"] = 2
data["Age"] = data["Age"]//10
data["Fare"] = data["Fare"] // 1

df_x = data.drop("Survived",axis = 1,inplace=False)
df_y = data["Survived"]

x_data = MinMaxScaler().fit_transform(df_x.values)
y_data = MinMaxScaler().fit_transform(df_y.values.reshape(-1,1))

# placeholder 
X = tf.placeholder(shape = [None,4],dtype = tf.float32)
Y = tf.placeholder(shape = [None,1],dtype = tf.float32)

# weight & bias
W = tf.Variable(tf.random_normal([4,1]),name = "weight")
b = tf.Variable(tf.random_normal([1]),name = "bias")

# Hypothesis
logits = tf.matmul(X,W)+b
# 지수 함수 형태의 H 정의
H = tf.sigmoid(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(3000):
    _, cost_val = sess.run([train,cost],feed_dict = {X:x_data,Y:y_data})
    if step % 300 == 0:
        print(cost_val)
        
## Accuracy
# H > 0.5 == 1 : True / H <= 0.5 == 0 : False
predict = tf.cast(H>0.5,dtype=tf.float32)    

## correct node
correct = tf.equal(predict,Y)
accuracy = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
print("정확도 : {}".format(sess.run(accuracy,feed_dict = {X:x_data,Y:y_data})))

df = pd.read_csv("./data/titanic/test.csv")
test = df[["Pclass","Sex","Age","Fare"]]
test.dropna(how ="any",inplace=True)

test.loc[test["Sex"]=="male","Sex"] = 1
test.loc[test["Sex"]=="female","Sex"] = 2
test["Age"] = test["Age"]//10
test["Fare"] = test["Fare"] // 1
x_test = MinMaxScaler().fit_transform(test.values)

print(sess.run(H, feed_dict = {X:x_test}) > 0.5 )
```



## Multinomial classification

- y의 label이 정해진 몇 가지 경우 중 1개로 산출

- 여러 개의 Binary classification 이용

  

#### Logistic regression => Multinomial regression 일 때,

#### 변화하는 요소 2가지

1.  Softmax 함수

    logistic regression처럼 0~1 사이의 값으로 sigmoid 취하는 것이 아니라

    softmax를 취해서 probability ( 확률 값 )을 도출해 냄

   ​								 => 0 ~ 1사이의 값이고 모두 더하면 1

   ``` python
   cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
       logits=logits, labels=y_data)
   ```

   

2. binary classification 이용 => 행렬 곱

   ONE-HOT Encoding : y 레이블을 spread로 펼쳐서 표현 

   => 2차원 matrix로 변환 됨 => 행렬 계산이 가능해짐

   ``` python
   logits = tf.matmul(X,W)+b
   H = tf.nn.softmax(logits)
   ```

### Multinomial classification example - 학점 부여

``` python
import tensorflow as tf
x_data = [[10,7,8,5],
          [8,8,9,4],
          [7,8,2,3],
          [6,3,9,3],
          [7,5,7,4],
          [3,5,6,2],
          [2,4,3,1]]

# ONE-HOT Encoding
y_data = [
    [1,0,0],
    [1,0,0],
    [0,1,0],
    [0,1,0],
    [0,1,0],
    [0,0,1],
    [0,0,1]]

# placeholder
X = tf.placeholder(shape=[None,4],dtype=tf.float32)
Y = tf.placeholder(shape=[None,3],dtype=tf.float32)

# weight & bias
# logistic 3개가 모여 있음
# W와 b 모두 3개씩!
W = tf.Variable(tf.random_normal([4,3]),name = "weight")
b = tf.Variable(tf.random_normal([3]),name = "bias")
     
# hypothesis
logits= tf.matmul(X,W)+b
H = tf.nn.softmax(logits)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))

# training node 생성
# learning rate가 작으면 local minimum에 빠져 해를 못 구할 수도 있음
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(30000):
    _, cost_val = sess.run([train,cost],feed_dict = {X:x_data,Y:y_data})
    if step % 3000 == 0:
        print(cost_val)
        
#  accuracy
# logistic = > H가 0~1 사이의 실수로 값 산출
# multinomial => H가 (확률,확률,확률)로 산출

# argmax(node, axis = 1 : 열 방향)
# axis 방향에서 가장 큰 값의 index return
# ex) (0.4,0.5,0.1) => 1
predict = tf.argmax(H,1)
correct = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,dtype = tf.float32))
print("Accuracy : {}".format(sess.run(accuracy,feed_dict = {X :x_data,Y:y_data})))
```



### Multinomial classification example - BMI

```python
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import MinMaxScaler
warnings.filterwarnings(action = "ignore")

df = pd.read_csv("./data/bmi/bmi.csv",sep=",")
df.dropna(how="any", inplace=True)
split_count = int(df.shape[0] * 0.7)
train_df = df.loc[:split_count,:]
test_df = df.loc[split_count:,:]

test_x_data = MinMaxScaler().fit_transform(test_df.drop("label" ,axis=1,inplace=False))
test_y_data = tf.one_hot(test_df["label"],3)
sess = tf.Session()
test_y_data = sess.run(test_y_data)

df_x = train_df.drop("label",axis = 1, inplace = False)
# ONE-HOT Encoding
df_y = train_df["label"]

x_data = MinMaxScaler().fit_transform(df_x.values)
y_data = tf.one_hot(df_y,3).eval(session=tf.Session())
# placeholder
X = tf.placeholder(shape=[None,2],dtype=tf.float32)
Y = tf.placeholder(shape=[None,3],dtype=tf.float32)

# weight & bias
# logistic 3개가 모여있다~!
# W와 b 모두 3개씩!
W = tf.Variable(tf.random_normal([2,3]),name = "weight")
b = tf.Variable(tf.random_normal([3]),name = "bias")
     
# hypothesis
logits= tf.matmul(X,W)+b
H = tf.nn.softmax(logits)

# cost
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))

# training node 생성
train = tf.train.GradientDescentOptimizer(learning_rate=1).minimize(cost)

# session & 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 학습
for step in range(10000):
    _, cost_val = sess.run([train,cost],feed_dict = {X:x_data,Y:y_data})
    if step % 1000 == 0 :
        print(cost_val)
        
predict = tf.argmax(H,1)
correct = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct,dtype = tf.float32))
print("Accuracy : {}".format(sess.run(accuracy,feed_dict = {X :x_data,Y:y_data})))


res = sess.run(tf.argmax(sess.run(H, feed_dict = {X:test_x_data}),1))
# for idx,i in enumerate(res):
#     if i == 0 : 
#         print(idx,"저체중")
#     elif i == 1 :
#         print(idx,"표준")
#     else : 
#         print(idx,"과체중")

print("테스트 정확도 :{}".format(sess.run(accuracy, feed_dict={X:test_x_data, Y:test_y_data})))  #정확도 노드를 출력  
```