# Python 기본2 (07/01)

### Python Sequence Type : list

- java의 arraylist와 상당히 유사

``` python
#list 생성
a = list()    # 공백 리스트 생성1
a = []        # 공백 리스트 생성2
a = [1,2,3]   # 일반적인 리스트 생성

# 여러 타입을 리스트 안에 담을 수 있음
a = [1,2,3,"안녕하세요",False , 3.15]  
a = [1,2,[3,4,5]]  # list 안에 list도 가능
```



- indexing과 slicing 사용 가능

``` python
print(a[0])
print(a[-2])
print(a[1:3])

# list의 연산
a = [1,2,3]
b = [5,6,7]
print(a+b)
# [1, 2, 3, 4, 5, 6, 7]

print(a*3)
# [1, 2, 3, 1, 2, 3, 1, 2, 3] 

a = [1,2,3]
a[0] = [9,9,9]
print(a)
# [[9,9,9],2,3]

# 0~1번까지의 list를 다른 list로 대체해라
# a[0]을 지칭하지만 인덱싱과 슬라이싱의 결과가 다름
a[0:1] = [9,9,9]
print(a)
# [9,9,9,2,3]

# a = [1,2,6,7]로 바꾸려면
a = [1,2,3,4,5,6,7]
a[2:5] = []
print(a)


# list의 사용 함수
my_list = [1,2,3]

#맨 마지막 index에 값을 추가
my_list.append(4)
print(my_list)
# 1,2,3,4

my_list.append([5,6,7])
print(my_list)
# 1,2,3,4,[5,6,7]

### 정렬 함수
# sort() : 기본적으로 오름차순 형태로 정렬
# 정렬시 원본이 바뀜 , 결과를 return 하지 않음
my_list = [6,3,8,2,1,9]
my_list.sort()

# 순서를 역으로 바뀜. sort 후 사용시 내림차순 정렬
my_list.reverse()
print(my_list)

# index() : 1의 값을 가진 index 값을 return 해주는 함수
my_list = [7,9,4,1,3]
print(my_list.index(1))
```



## Python Sequence Type : Tuple

- list와 거의 유사

- 표현 방법과 수정, 삭제가 불가능 , readonly

- 리스트 : [] , 튜플 : ()

``` python
a = ()
a = (1,2,3) # [1,2,3]
a = (1) # 튜플이 아니라 숫자 1을 의미함

# 요소가 1개만 있을 때 튜플로 표현하려면 , 찍어줘야 함
a = (1,)

# 튜플은 괄호 생략 가능
a = (1,2,3,4)
a = 1,2,3,4

# a = 10 b = 20 c = 30
a,b,c = 10,20,30

# indexing과 slicing 둘 다 사용할 수 있음
a = 1,2,3,4
print(a[1]) # 2 출력
print(a[2:4]) # slicing은 원본은 그대로 두고 원하는 구간을 메모리에 복사해서 쓰는 것

#list와 마찬가지로 +, * 연산이 가능
a = (1,2,3)
b = (5,6,7)
print(a+b)

#list와 tuple 간의 변환
my_list = [1,2,3]
my_tuple = tuple(my_list)

my_tuple = 10,20,30,40
my_list = list(my_tuple)
print(my_list)
```



## Python Sequence Type : range

- range는 숫자 sequence로 주로 for문에서 사용

``` python
# range의 인자가 1개이면 0부터 시작해서 9까지1씩 증가
my_range = range(10)
print(my_range)

# range의 인자가 2개이면 시작, 끝
my_range = range(10,20)
print(my_range)

# range의 인자가 3개이면 시작,끝,증감을 의미
my_range = range(10,20,3)
print(my_range)

#my_range 는 10,13,16,19 가짐
print(12 in my_range)

#range도 list나 tuple처럼 indexing과 slicing 가능
my_range = range(10,20,3)
print(my_range[-1])
print(my_range[:2])

#range를 이용한 for문
for tmp in range(10,20,2):
    print(tmp)
```



## Python Mapping Type : dict

- 표현법은 JSON 표현과 유사

  { "name" : "홍길동" , "age" : 30}

``` python
my_dict = {"name" : "홍길동" , "age" : 30}
print(type(my_dict))

# 새로운 dict 추가하려면 key:value로 추가
my_dict[100] = "홍길동"
print(my_dict)

del my_dict["age"]
print(my_dict)

# key값이 중복되는 경우
# 어떤 값이 할당될지 알 수 없음
my_dict={"name":"홍길동", "age":30, "age":40}

# keys() 함수
my_dict = {"name": "홍길동" , "age" : 30, "address":"서울"}
# return 값들의 리스트처럼 생긴 객체
# list와 유사하지만 list의 함수는 사용할 수 없음
print(my_dict.keys())

#values() 함수 : dict의 value들만 뽑아냄
#items() 함수 : (key,value)의 형태인 tuple로 리스트처럼 생긴 객체를 return 

my_dict = {"name": "홍길동" , "age" : 30, "address":"서울"}
for tmp in my_dict.keys():
    print("{0}, {1}".format(tmp,my_dict[tmp]))
```



## Python set Type : set
- set의 가장 큰 특징 : 중복이 없는 저장 장소, 순서가 없는 저장 장소

``` python

# set 생성 => {1,2,3}
my_set = set([1,2,3])
print(my_set)
my_set = set("hello")
print(my_set)

# 기본적인 set 연산
# 합집합, 교집합, 차집합
s1 = {1,2,3,4,5}
s2 = {5,6,7,8,9}

# 교집합 (intersection)
print(s1 & s2)

# 합집합 (union)
print(s1 | s2)

# 차집합 ( differences )
print(s1 - s2)

# 기타 사용가능한 method
my_set = {1,2,3,4,5}
# set에 새로운 요소를 추가하려면
my_set.add(10)

# set에 여러개를 추가하려면
my_set.update([7,8,9])

# set에서 삭제할 경우
my_set.remove(1)
```

## Python Data Type - bool

- 논리 상수인 True, False를 사용

- 다음과 같은 경우는 false로 간주

1. 빈 문자열은 논리 연산시 False("")
2. 빈 리스트([]), 빈 튜플(()), 빈 dict ({})
3. 숫자 0 False 간주, 나머지 다른 숫자 True 간주
4. None => False 간주



## Python의 console 입출력
- 입력받은 값은 무조건 문자열

``` python
input_value = input("숫자를 입력하세요!")
result = input_value * 3
print(result)
# 출력 : 입력값입력값입력값

# eval() : 문자열을 숫자 연산 처리
result = eval(input_value) * 3
print(result)
# 출력 : 입력값*3 된 값
```



