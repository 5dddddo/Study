# Deep Learning (07/20)

- Single layer의 문제점
  - Single layer로 AND/OR problem ->  해결 가능

    But, Single layer로 XOR problem ->  해결 불가능

- 해결방법 : Deep Learning
  - 하나의 Logistic regression으로는 해결 불가능
  - Multiple Logistic regression으로는 해결 가능
  - Hidden layer를 추가하여 XOR problem을 해결 함



### Deep Learning example - MNIST

``` python
# 기본 MNIST(multinomial classification)
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data Loading
train_data = pd.read_csv("./data/digitrecognizer/train.csv")
train_x_data = train_data.drop('label', axis = 1)
train_y_data = tf.one_hot(train_data["label"], depth=10).eval(session = tf.Session())
test_x_data = pd.read_csv("./data/digitrecognizer/test.csv")

# Tensorflow Graph Initialization
tf.reset_default_graph()

X = tf.placeholder(shape = [None, 784], dtype = tf.float32)
Y = tf.placeholder(shape = [None, 10], dtype = tf.float32)
keep_prob = tf.placeholder(dtype=tf.float32)

# Weight & bias
W1 = tf.get_variable("weight1", shape = [784,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name = "bias1")
_layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)
layer1 = tf.nn.dropout(_layer1, keep_prob = keep_prob)

W2 = tf.get_variable("weight2", shape = [256,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name = "bias2")
_layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
layer2 = tf.nn.dropout(_layer2, keep_prob = keep_prob)

W3 = tf.get_variable("weight3", shape = [256,10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]), name = "bias3")

# Hypothesis
logits = tf.matmul(layer2,W3) + b3
H = tf.nn.relu(logits)

# cost function
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = Y))

# train node
train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# session object & initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# epoch & batch size
training_epoch = 10
batch_size = 100

# training
for step in range(training_epoch):
    num_of_iteration = int(train_data.shape[0] / batch_size)
    cost_val = 0
    
    for i in range(num_of_iteration):
        batch_x, batch_y = train_x_data[i*batch_size:(i+1)*batch_size],train_y_data[i*batch_size:(i+1)*batch_size]
        _, cost_val = sess.run([train, cost], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})

    if step %5 == 0:
        print(cost_val)
        
#predict check
predict = tf.argmax(H,1)
result = sess.run(predict, feed_dict={X:test_x_data, keep_prob: 1.0})
df = pd.DataFrame({
    'ImageId': [i for i in range(1,28001)],
    'Label': result
})
df.to_csv('./data/digitrecognizer/submission.csv', index=False)

correct = tf.equal(predict, tf.math.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))
print("Accuracy: {}".format(sess.run(accuracy, feed_dict = {X: train_x_data, Y: train_y_data, keep_prob: 1.0})))
```



### Deep Learning example - Titanic

``` python
# Kaggle - Titanic 구현 (NN)

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import warnings
from random import randrange

warnings.filterwarnings(action="ignore")

df = pd.read_csv("./data/titanic/train.csv")
train_df = df[["Survived","Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]]
train_df.fillna(method="ffill",inplace = True)
train_df.loc[train_df["Sex"]=="male","Sex"] = 1
train_df.loc[train_df["Sex"]=="female","Sex"] = 2
train_df["Age"] = train_df["Age"]//10
train_df["Fare"] = pd.qcut(train_df["Fare"],5,labels=[1,2,3,4,5])
embarked_mapping = {"S":1,"C":2,"Q":3}
train_df["Embarked"] = train_df["Embarked"].map(embarked_mapping)

train_x_data = train_df.drop("Survived",axis = 1,inplace=False)
train_y_data = train_df["Survived"]

train_x_data = MinMaxScaler().fit_transform(train_x_data.values)
train_y_data = MinMaxScaler().fit_transform(train_y_data.values.reshape(-1,1))

df = pd.read_csv("./data/titanic/test.csv")
test_df = df[["Pclass","Sex","Age","Fare","SibSp", "Parch", "Embarked"]]
test_df.fillna(method="ffill",inplace = True)

test_df.loc[test_df["Sex"]=="male","Sex"] = 1
test_df.loc[test_df["Sex"]=="female","Sex"] = 2
test_df["Age"] = test_df["Age"]//10
test_df["Fare"] = test_df["Fare"] = pd.qcut(test_df["Fare"],5,labels=[1,2,3,4,5])
test_df["Embarked"] = test_df["Embarked"].map(embarked_mapping)
test_x_data = MinMaxScaler().fit_transform(test_df.values)

# Tensorflow Graph Initialization
tf.reset_default_graph()

# placeholder 
X = tf.placeholder(shape = [None,7],dtype = tf.float32)
Y = tf.placeholder(shape = [None,1],dtype = tf.float32)
keep_prob = tf.placeholder(dtype=tf.float32)

# Weight & bias
W1 = tf.get_variable("weight1", shape = [7,256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]), name = "bias1")
_layer1 = tf.nn.relu(tf.matmul(X,W1)+b1)
layer1 = tf.nn.dropout(_layer1, keep_prob = keep_prob)

W2 = tf.get_variable("weight2", shape = [256,256], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([256]), name = "bias2")
_layer2 = tf.nn.relu(tf.matmul(layer1,W2)+b2)
layer2 = tf.nn.dropout(_layer2, keep_prob = keep_prob)

W3 = tf.get_variable("weight3", shape = [256,1], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([1]), name = "bias3")

# Hypothesis
logits = tf.matmul(layer2,W3) + b3
H = tf.nn.relu(logits)

# cost function
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = Y))

# train node
train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

# session object & initialization
sess = tf.Session()
sess.run(tf.global_variables_initializer())

training_epoch = 10
# training
for step in range(training_epoch):
    num_of_iteration = int(train_x_data.shape[0])
    cost_val = 0
    
    for i in range(num_of_iteration): 
        _, cost_val = sess.run([train, cost], feed_dict={X: train_x_data, Y: train_y_data, keep_prob: 1.0})

    if step %5 == 0:
        print(cost_val)
        

## Accuracy
# H > 0.5 == 1 : True / H <= 0.5 == 0 : False
predict = tf.cast(H>0.5,dtype=tf.float32)    
correct = tf.equal(predict,Y)
accuracy = tf.reduce_mean(tf.cast(correct,dtype=tf.float32))
print("Accuracy: {}".format(sess.run(accuracy, feed_dict = {X: train_x_data, Y: train_y_data, keep_prob: 1.0})))

# prediction
result = sess.run(tf.cast(sess.run(H, feed_dict = {X:test_x_data,keep_prob:1.0}) > 0.5,dtype = tf.int32))
result = result[:,0]
(result.shape)
data = pd.DataFrame({
    'PassengerId': [i for i in range(892,892+test_x_data.shape[0])],
    'Survived': result
})
data.to_csv('./data/titanic/submission.csv', index = False)
```

