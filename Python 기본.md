# Python 기본 (06/28)

#### 주석 

```python
# 한 줄 주석

'''
여러 줄 주석
'''
```

#### 내장상수

- True , False , None( 값이 존재하지 않음 )

```python
a = True
b = False
c = None
```



## built-in data type ( 내장형 )

: 계산할 때 실수형으로 처리

###  Numeric type( 숫자형 )

``` python
a = 123 # 정수
b = 3.1415926535 # 실수
c = 3.14E10 # 실수( 지수형태 )
div = 3/4
print(div)  # java에서는 0 출력, python은 계산 결과 0.75 출력

a = 123 # 정수
b = 3.1415926535 # 실수
c = 3.14E10 # 실수( 지수형태 )
d = 0xEE # 16진수
e = 0o54 # 8진수

```

코드를 실행하려면 ctrl + enter

```python
result = 3 ** 4 # 3의 4제곱 ( 제곱 : ** )

result = 10 %  # 나머지

result = 10 // 3 # 몫
```



### Text Sequence type( 문자열 ) : str

- #### 문자열 생성 방법

  java는 문자열(" ")과 문자(' ')

  ##### python는 문자 개념 X , 구분없이 문자열로 취급 ("" , ' ')

``` python
first = "이것은"
second = "소리없는"
third = "아우성"
print(first+second+third)
# 이것은소리없는아우성

num = 100
print(first+num) # ERROR
print(first+str(num))
# 이것은100
```

- ##### str() : 숫자를 문자열로 변환해주는 함수

``` python
text = 'python'
print(text*3) # 문자열 3번 찍어줌

#sample text
sample_text = "Show me the money"

str(문자열) => Text Sequence Type

print(sample_text[0]) # 배열처럼 동작
print(sample_text[-2]) # 끝에서 두번째


# Slicing할 때 앞은 포함, 뒤는 불포함
print(sample_text[1:3]) # index 1~2까지 출력
print(sample_text[:3]) # 0번 인덱스부터 2까지 출력
print(sample_text[3:]) # 3부터 끝까지 출력
print(sample_text[:]) #문자열 전체 출력

# in 연산자 : 문자열 내에 특정 문자열이 있는지 결과를 논리값으로 반환
print("sample" in sample_text) # 
print("sample" not in sample_text)
```



- #### 문자열 formatting

``` PYTHON
apple = 30
my_text = "나는 땅콩을 %d개 가지고 있어요" %0
print(my_text)

apple = 5
banana = "여섯"
my_text= "나는 사과 %d개와 바나나 %s개를 가지고 있어요" %(apple,banana)

my_text = "나는 사과를 {0}개 가지고 있어요!".format(10)

my_text= "나는 사과 {0}개와 바나나 {1}개를 가지고 있어요".format(5,3)


```



- #### 문자열 함수

```python
sample_text = "cocacola"
print(len(sample_text)); # 문자열 길이 내장 함수

# 문자열이 몇 번 나오는지 count하는 함수
print(sample_text.count("c")) 

# 특정 문자열이 처음 나오는 index 반환
# -1 : 찾는 문자열 없음
print(sample_text.find("o")) 

# => a:b:c:d로 만들기
a = ":"
b = "abcd"
print(a.join(b));

a= "       hobby     "
print(a.upper()) # 대문자 변환
print(a.lower()) # 소문자 변환
print(a.strip()) # 문자열의 앞,뒤 공백 제거
```

