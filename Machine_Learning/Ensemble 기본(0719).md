# Ensemble (07/19)

## Class

- 객체 모델링의 수단

- ADT (Abstract data type) : 추상 데이터 타입

- class안에 정의되는 내용은 크게 3가지

  1. 상태값 저장을 위한 변수(field, member variable, property)

  2. 수행하는 작업을 위한 함수(method, member function, method)

  3. 정의된 class 정보를 바탕으로 일정 메모리를 확보

     - 확보된 메모리 공간 : 인스턴스, 객체

     - 이런 객체를 생성하기 위해 생성자가 호출되어야 함

- 형식

  ``` python
  class Class명:
      # constructor & property 생성
  	def __init__(self,매개변수 list):
      	self.property명
      # constructor
      def method명:
  ```

  - property를 생성하려면 self를 이용
  - self를 붙이지 않으면 지역변수로 사용 -> 생성자 호출 후 사라짐 

  

- Student 예제

```python
class Student:
    # constructor
    def __init__(self,s_name,s_kor,s_eng,s_math):
        self.student_name = s_name
        self.kor = s_kor
    	self.eng = s_eng
		self.math = s_math
    
    # method
    def get_total(self):
        # self를 붙여야 property 지칭
        self.total = self.kor+self.eng+self.math   
        return self.total
    
stu1 = Student("홍길동",10,20,30)
```



## Ensemble을 이용한 MNIST

``` PYTHON
## Ensemble을 이용한 MNIST
import tensorflow as tf
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data

## Graph 초기화
tf.reset_default_graph()

## Class Define
## cnn 모델을 생성하는 class
class CnnModel:
        # constructor
        def __init__(self,sess,name,m,test):
            self.sess = sess
            self.name = name
            self.mnist = m
            self.test_x_data = test
            self.build_net()
         
        # tensorflow model graph(node) 생성하는 method    
        def build_net(self):    
            with tf.variable_scope(self.name):
                # tensorflow graph 초기화
                self.X = tf.placeholder(shape = [None,784], dtype = tf.float32)
                self.Y = tf.placeholder(shape = [None,10], dtype = tf.float32)
                self.drop_rate = tf.placeholder(dtype = tf.float32)
                X_img = tf.reshape(self.X,[-1,28,28,1])

                L1 = tf.layers.conv2d(inputs=X_img,
                                      filters = 32,
                                      kernel_size=[3,3],
                                      padding= "SAME",
                                      strides=1,
                                      activation=tf.nn.relu)

                L1 = tf.layers.max_pooling2d(inputs= L1,
                                             pool_size=[2,2],
                                             strides = 2,
                                             padding = "SAME")

                L2 = tf.layers.conv2d(inputs=L1,
                                      filters = 64,
                                      kernel_size=[3,3],
                                      padding= "SAME",
                                      strides=1,
                                      activation=tf.nn.relu)

                L2 = tf.layers.max_pooling2d(inputs= L2,
                                             pool_size=[2,2],
                                             strides = 2,
                                             padding = "SAME")

                L2 = tf.reshape(L2, [-1,7*7*64])

                W1 = tf.get_variable("weight1",shape = [7*7*64,256],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b1 = tf.Variable(tf.random_normal([256]),name = "bias1")
                _layer1 = tf.nn.relu(tf.matmul(L2,W1)+ b1)
                layer1 = tf.layers.dropout (_layer1, rate = self.drop_rate)

                W2 = tf.get_variable("weight2",shape = [256,10],
                                     initializer=tf.contrib.layers.xavier_initializer())
                b2 = tf.Variable(tf.random_normal([10]),name = "bias2")

                self.logits = tf.matmul(layer1,W2)+ b2
                self.H = tf.nn.relu(self.logits)

                self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.logits, labels=self.Y))

                self.train_net(self.mnist.images,self.mnist.labels)
                self.get_accuracy(self.mnist.images,self.mnist.labels) 
                self.get_prediction(self.test_x_data)

        # model 학습
        def train_net(self, train_x_data, train_y_data):
            
            optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
            self.train = optimizer.minimize(self.cost)
            
            self.sess.run(tf.global_variables_initializer())
            
            training_epoch = 30
            batch_size = 100 
            print("학습")
            for step in range(training_epoch):
                num_of_iter = int(self.mnist.num_examples / batch_size)
                cost_val = 0
                for i in range(num_of_iter):
                    batch_x, batch_y = self.mnist.next_batch(batch_size)
                    _,cost_val = sess.run([self.train, self.cost],
                                         feed_dict = {self.X : batch_x,
                                                      self.Y: batch_y,
                                                      self.drop_rate:0.7})
                print(cost_val)
                    
        # model의 Accuracy 측정
        def get_accuracy(self,train_x_data, train_y_data):
            predict = tf.argmax(self.H,1)
            correct = tf.equal(predict,tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, dtype = tf.float32))
            self.result = sess.run(self.accuracy , feed_dict = {self.X:train_x_data,
                                                           self.Y: train_y_data,
                                                           self.drop_rate:0.7})
            print("정확도 : {}".format(self.result))

        # model의 prediction
        def get_prediction(self,x_data):
            sess.run(self.H,feed_dict={self.X:x_data,
                                       self.drop_rate:0.7})
            
## 1.Data loading
mnist= input_data.read_data_sets("./data/mnist", one_hot=True)
mnist= mnist.train
test_x_data = pd.read_csv("./data/digitrecognizer/test.csv")

## 2. Model의 개수 지정 & 생성
sess = tf.Session()
num_of_model = 10
cnn_models = [CnnModel(sess,
                       "Model"+str(x),
                       mnist,test_x_data) for x in range(num_of_model)]


```


        
    