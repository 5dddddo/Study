# Python 기본3 (07/01)

### if문

``` python
area = ["seoul", "suwon","pusan"]

if "suwon" in area:
    # 수행할 코드 작성
    # 비워두기
    pass 
elif "pusan1" in area:
    print("부산")
else:
    print("마지막입니다.")
```

### 날짜

``` python
# python data type - 날짜(date, time, datetime)
from datetime import date, time, datetime

# 현재 날짜 출력
today = date.today()
print("오늘의 날짜 : " + str(today))
print("연도 : {0}, 월 : {1}, 일 : {2}".format(today.year, today.month, today.day))

today = datetime.today()
print(today)

# 날짜 연산
from datetime import date, time, datetime, timedelta
from dateutil.relativedelta import relativedelta

today = date.today() # 오늘의 날짜
# days = timedelta(days=3) # + : 이후 - : 이전 날짜 간격

# days = timedelta(months=-2) : timedelta에서는 months, years는 사용할 수 없음
# 일, 시간은 timedelta / 월, 년은 relativedelta에서 처리

days = relativedelta(months=-2) # relativedelta에서는 months, years는 사용할 수 있음
print(today+days)

today = datetime.today() # 오늘의 날짜와 시간
delta = timedelta(hours=3) # + : 이후 - : 이전 날짜 간격
print(today+delta)
```



### Print 문

``` python
for tmp in range(10):
    print("tmp : {}".format(tmp), end = ",")

# print 함수는 기본적으로 출력한 다음에 개행
# print 함수의 마지막 인자로 출력에 대한 제어 가능


```

### for 문

``` python
my_list =[1,2,3,4,5]
my_sum = 0

for tmp in my_list:
    my_sum += tmp
print("합계 : {}".format(my_sum))

my_sum= 0
for i in range(len(my_list)):
    my_sum += my_list[i]
print("합계 : {}".format(my_sum))

my_list =[1,2,3,4,5]
new_list = [tmp * 2 for tmp in my_list]
print(new_list)

my_list =[1,2,3,4,5]
new_list = [tmp * 2 for tmp in my_list if tmp%2 == 0]
print(new_list)
```



### python의 기본 자료구조 (자료형)

#### 1. 내장함수

- 내장함수의 종류
  - len() : 문자열의 길이, list/set의 갯수
  - abs() : 숫자에 대한 절대값
  - all() : 반복가능한 자료형에 대해 모두가 참이면 return True
  - any() : 반복가능한 자료형에 대해 단 한개라도 참이면 return True
  - eval() : 수치연산에 대한 문자열을 입력받아서 수치연산 수행
  - int() : 정수로 변환
  - list() : 리스트로 변환
  - tuple() , str()

``` python
my_list = [1,2,True,"Hello"]
print(all(my_list))
print(any(my_list))
print(eval("3+4*2"))

my_list = [1,2,3,4,5]
# tuple () 생략 가능 -> idx,item
# enumerate() 
# : 반복 가능한 자료형에서 index와 값을 값이 뽑아낼 수 있음
for idx,item in enumerate(my_list):
    print("idx : {0} item : {1}".format(idx,item))
    
# 정렬에 대한 instance가 가진 method : sort()
# 정렬이 안 된 리스트 준비
my_list = [7,3,9,2,8,5]
# instance의 method를 호출
# 원본 instance를 처리, return : None
my_list.sort()
print(my_list)

# 정렬에 대한 내장 함수 : sorted()
# 원본 객체 변화 X , 정렬 결과 return 해줌
my_list = [7,3,9,2,8,5]
result = sorted(my_list) 
print(result)
```

#### 2. 사용자 정의 함수

- 함수 정의

``` python
def my_sum(a,b,c):
    return a+b+c
    
result = my_sum(1,2,3)
print(result)

# 인자의 개수를 알지 못할 때 : *
def my_sum1(*args):
    result = 0
    for tmp in args:
        result += tmp
    return result 
    
result = my_sum1(1,2,3,4,5)
print(result)
```

- 사용자 정의 함수의 scope
  - 전역변수(global variable)와 지역변수(local variable)

``` python
tmp = 100 # 전역변수
def my_func(x):
    # tmp # 지역변수
    global tmp # 전역변수를 지칭
    tmp += x
    return tmp

print(my_func(3))
```



## Python의 객체지향

### Class
1. 현실세계의 개체를 프로그래밍적으로 모델링하기 위해서 사용하는 수단

   => 객체 모델링의 수단

2. 새로운 데이터 타입을 만드는 수단

``` python
class Student:
    # property(field)
    s_nation = "국적" # class variable : instance를 생성해도 static처럼 1개의 변수 공유 가능
    # instance : 클래스를 사용하기 위해 메모리에 띄운 상태 -> 생성자 호출
    
    # Constructor
    def __init__(self,n,nation): # self => this 와 같은 변수
        Student.s_nation = nation # class variable
        self.s_name = n           # instance variable : 생성자 내에서 self 키워드와 같이 변수 선언하면 instance 변수
    
    # Method
    def display(self):
        print("국적 : {0} 이름 : {1}".format(self.s_nation, self.s_name))
    
# instance 생성
stu1 = Student("홍길동","한국")
print(stu1.s_name)
print(stu1.s_nation)
stu1.display()

```



## Python의 Module

- **Module** : 함수, 변수, class를 모아놓은 파일

  ​				다른 python 프로그램에서 불러와 사용할 수 있도록

  ​				만들어진 python 파일을 지칭

- 만들어진 module을 불러와 사용하는 keyword가 **import**

``` python
from my_sum import my_sum
print(my_sum(1,4))

import my_package.my_module
print(my_package.my_module.my_sum(10,20))

# from 뒤에는 package import 뒤에는 사용하려는 함수의 파일명
from my_package import my_module
print(my_module.my_sum(10,20))
```

