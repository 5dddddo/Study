# Pandas 기본1 (07/04)

#### pandas : python data 분석의 핵심 module(package)

- Pandas는 고유한 자료구조를 2개 사용하는데

  - **Series** : numpy의 1차원 배열과 상당히 유사

  ​			 동일한 데이터 타입의 복수 개의 요소로 구성

  - **DataFrame** : table 형식으로 구성된 자료구조

    ​					   2차원 배열과 상당히 유사

    ​					  데이터 베이스 테이블과 상당히 유사

- 데이터 분석

1. EDA ( 탐색적 데이터 분석 ) : 예) Excel로 데이터 분석

   python 언어로 pandas를 이용해서 EDA 수행

2. 통계적 데이터 분석 : 통계학 이론을 이용한 분석

3. 머신러닝 : 기존 데이터를 이용해서 프로그램을 학습

   (딥러닝)    이 학습된 결과를 이용해 예측



- jupyter에서 pandas를 import해서 사용하기 위해

  Anaconda prompt를 관리자 모드로 실행하여

  conda install pandas 수행 

``` python
import numpy as np
import pandas as pd

print("=" * 30)
arr = np.array([-1,10,50,99],dtype=np.float64)
print(arr)
print("=" * 30)

s = pd.Series([-1,10,50,99],dtype=np.float64)
display(s)
print(s.values)  # numpy 1차원 array로 리턴
print(s.index)
print(s.dtype)

# 출력
# numpy.array
==============================
[-1. 10. 50. 99.]
==============================

# pandas.Series
0    -1.0
1    10.0
2    50.0
3    99.0
dtype: float64
[-1. 10. 50. 99.]
RangeIndex(start=0, stop=4, step=1)
float64

import numpy as np
import pandas as pd

# index를 문자로 쓸 수 있음
s = pd.Series([-1,10,50,99], index =['c','a','k','tt'])
display(s)

# 다른 형식의 인덱스를 사용할 수 있음
print(s['a'])     # 10
print(s[1])       # 10
print(s['a':'k']) # 문자로 slicing 가능 => 둘 다 포함함
                  # slicing 범위 조심!
print(s[1:3])     # 일반적인 slicing 사용 가능
print(s.sum())
```



## 공장 예제

``` python
## A 공장의 2019-01-01부터 10일간 제품 생산량을 Series에 저장
## 단, 생산량의 평균은 50이고 표준편차는 5인 정규분포에서
## 생산량을 랜덤하게 결정

## B 공장의 2019-01-01부터 10일간 제품 생산량을 Series에 저장
## 단, 생산량의 평균은 70이고 표준편차는 8인 정규분포에서
## 생산량을 랜덤하게 결정

## 2019-01-01부터 10일 간 모든 공장(A,B)의 생산량을 날짜별로 구하기
import numpy as np
import pandas as pd
# 날짜 연산을 위한 import
from datetime import date,timedelta
# 문자열을 date type으로 parsing하기 위해 import
from dateutil.parser import parse

start_day = parse("2019-01-01")
factory_a = pd.Series([int(x) for x in np.random.normal(50,5,(10,))],
                      index = [start_day + timedelta(days=x) for x in range(10)])
print(factory_a)
factory_b = pd.Series([int(x) for x in np.random.normal(70,8,(10,))],
                      index = [start_day + timedelta(days=x) for x in range(10)])

print(factory_b)


# Series의 덧셈 연산은 같은 index끼리 수행
print(factory_a+factory_b)
```



### Series를 dictionary 이용해 생성

``` python
##이전에는 series라는 자료구조를 만들때 python의
## list를 이용해서 만들었는데 이번에는
## dictionary를 이용

my_dict = {"서울":3000,"부산":2000,"제주":1000}
s = pd.Series(my_dict)
s.name = "지역별 가격 데이터"
s.index.name = "지역"

display(s)

# pandas의 두번째 자류구조는 DataFrame을 살펴보아요
## 거의 대부분의 경우 DataFrame을 이용해서 데이터 분석
## dictionary를 이용해서 생성

import numpy as np
import pandas as pd

data = {"name": ["kim","lee","park","moon","kim"],
        "year":[2015,2016,2019,2019,2015],
        "point" : [3.1,4.3,1.2,2.3,3.9] }

df = pd.DataFrame(data)
display(df)

print("DataFrame의 shape : {}".format(df.shape))
print("DataFrame의 요소 개수 : {}".format(df.size))
print("DataFrame의 차원 : {}".format(df.ndim))
print("DataFrame의 인덱스 : {}".format(df.index))
print("DataFrame의 컬럼 : {}".format(df.columns))
print("DataFrame의 데이터 : {}".format(df.values))
```
