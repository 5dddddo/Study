

# Pandas 기본2 (07/05)

## 데이터 읽어서 DataFrame으로 생성

- #### Database에서 읽어오기

  - mysql 사용
  - anaconda prompt에서 conda install pymysql
  - table 생성 후 jupyter에서 읽어오기

``` python
# Database로부터 데이터를 얻어내서 DataFrame을 생성
# mysql을 이용해서 처리함
# python에서 mysql을 사용할 수 있도록 도와주는 module을 설치
import pymysql.cursors
import numpy as np
import pandas as pd

# mysql을 다운로드 받아서 간단하게 설치/설정
# 데이터베이스를 연결하기 위한 변수를 하나 생성
conn = pymysql.connect(host="localhost",
                      user="python",
                      password="python",
                      db="library",
                      charset="utf8")
sql = "select bisbn,btitle,bauthor,bprice from books"

df = pd.read_sql(sql,con=conn)
display(df)
```



- #### CSV 파일 읽어오기

  - pd.read_csv(파일경로)
  - head() : 상위 5개 읽기
  - tail() : 하위 5개 읽기
  - to_json(파일경로) : dataframe에 들어있는 데이터를 json으로 저장

``` python
# CSV 파일을 읽어들여서 Pandas의 DataFrame으로 표현
## 일반적으로 CSV 파일의 첫번째 행은 컬럼을 명시
import numpy as np
import pandas as pd

df = pd.read_csv("./data/MovieLens/ratings.csv")
# 상위 5개
display(df.head())

# 하위 5개
display(df.tail())

# 3가지 형태의 데이터 파일을이용
# CSV 파일을 읽어들여서 PANDAS의 DataFrame으로 표현
import numpy as np
import pandas as pd

# dataframe에 들어있는 데이터를 json으로 저장하고 싶음
df.to_json("./data/MovieLens/ratings.json")
```



- #### JSON 파일 읽어오기 

  - json은 파이썬의 내장 package로 txt 기반
  - file = open(파일경로) : .json 파일 열기
  - json.load(file) : JSON string을 파싱해서 dictionary 타입으로 return
  - file.close() : 파일 닫기

``` python
import numpy as np
import pandas as pd
import json

file = open("./data/MovieLens/ratings.json")
my_dict = json.load(file)
file.close()

df = pd.DataFrame(my_dict)
print(df.index)
# json 파일에서 가져온 index도 문자열 type
# 따라서, index가 사전순으로 정렬되어 1들이 2보다 앞에 나옴
display(df)

```

![1562316164313](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1562316164313.png)





## Pandas의 DataFrame

### DataFrame 생성

- index를 자유롭게 설정할 수 있음

- DB의 table 형식과 비슷

- display() : table 구조로 출력

- describe() : 기본적인 분석함수 제공

  ​					count, mean, std, ...

``` python
import numpy as np
import pandas as pd

data = {"이름": ["홍길동","강감찬","이순신","신사임당"],
        "학과": ["컴퓨터","경영","철학","미술"],
        "학년":[1,3,2,4],
        "학점":[3.1,2.9,4.5,3.9]
       }

df = pd.DataFrame(data,
                 columns=["학과","이름","학년"],
                 index = ["one","two","three","four"])

## dataframe은 기본적인 분석함수 제공
display(df)
display(df.describe());
```

![1562396814988](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1562396814988.png)

![1562396678258](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1562396678258.png)



### DataFrame의 제어 / 처리

#### col 데이터 추출

- DataFrame의 col 1개 추출
  - col 1개는 Series type : numpy array view로 가져옴, 복사본 X

    ex) type(df["이름"]))  -> pandas.core.series.Series

    

- DataFrame의 col 2개 추출
  - col 2개는 DataFrame type

  - index로 추출 X

    ex) (df["이름","학년"]) => Error!

  - **Fancy indexing** 이용 => 배열 리스트로 추출

  ​										           indexing 하는 부분에 index 배열을 이용

  ​		ex) df[["학과","이름"]]

``` python
import numpy as np
import pandas as pd
import warnings

# warning을 출력하지 않기 위한 설정
# warnings.filterwarnings(action="ignore")  ## 출력 X
  warnings.filterwarnings(action="default") ## 출력 O

# Dataframe에서 특정 col만 추출
year=df["학년"] 
# df["학년"]은 [index,"학년"] 형식의 Series => view로 가져옴
# view가 아닌 복사본을 이용하려면 copy()
year[0] = 100

# col 2개 이상 추출하려면
# display(df["이름","학년"]) => Error

# Fancy indexing 이용 => 배열 리스트로 추출
# 인덱싱하는 부분에 인덱스 배열을 이용하는 indexing 방법
display(df[["학과","이름"]])  
```



#### col 데이터 수정

``` python
# col 값 수정 
df["학년"] = np.array([1,2,2,1])
df["학년"] = 3
df["나이"] = pd.Series([20,21,23,25])

# 위에서 인덱스를 one,...,four 로 지정해서
# default index인 1,2,3,4로 안 들어감
# 인덱스 지정해줘야함

df["나이"] = pd.Series([20,21,23,25],
                     index = ["one","two","three","four"])

df["장학금여부"]= df["학점"]>3.0
display(df)

# del df["학점"]; # 학점 컬럼 삭제, 실제 사용 X

## 컬럼 삭제한 일반적인 방법
# axis = 열방향인지 행방향인지
# inplace = 원본에서 제거할지 여부
#         = True : 원본 삭제시 return X 
#         = False : 원본에서 삭제하지 말고 복사본 만들어서 return
df.drop("학점",axis=1,inplace = True)
#display(df)


################################################
# dataFrame의 컬ㄹ럼을 제어하기 위한 CRUD 방법 #
################################################

## row indexing
#display(df)
#display(df[0])     ## index 번호로 row 선택할 수 없음 => 
                    ## df[" ~~~ "]  => 이건 col 지정 방식
#display(df["one"]) ## 이 방식은 컬럼을 지정할때 쓰는 방식이어서 error
                    ## index 번호로 특정 row를 선택하는건 안돼요!
                    ## 슬라이싱은 가능, 슬라이싱 결과는 dataframe
        
# display(df[0:1])
# display(df[:-1])
# display(df["one":"three"])

# 일반적으로 행을 제어할때
# df.loc[0]  # 0번째 행을 indexing 하려는 의도지만 사용할 수 없음 => error

# 숫자 index말고 사용자가 문자 index로 지정한 경우에는 사용 가능
# 1개의 행을 선택하는 것이기 때문에 Series로 return
df.loc["one"]
print(df.loc[["one","three"]]) # fancy indexing


# 컬럼은 []으로 row는 loc()로 제어
# loc 사용 시 컬럼에 대한 indexing도 할 수 있음
display(df.loc["one":"three", ["이름","학년"]]) # slicing해서 행 가져오고
                                                # fancy indexing해서 열 가져옴

    
# 새로운 행을 추가하려면
df.loc["five",:] = ["물리","유관순",2,3,4]
display(df)

## row를 삭제하려면
df.drop(["one","three"],axis=0,inplace=True)
display(df)

import numpy as np
import pandas as pd
import warnings

# warning을 출려하지 않기 위한 설정
# warnings.filterwarnings(action="ignore") ## off
warnings.filterwarnings(action="default") ## on


data = {"이름": ["홍길동","강감찬","이순신","신사임당"],
        "학과": ["컴퓨터","경영","철학","미술"],
        "학년":[1,3,2,4],
        "학점":[3.1,2.9,4.5,3.9]
       }

df = pd.DataFrame(data,
                 columns=["학과","이름","학년","학점"],
                 index = ["one","two","three","four"])

# 1. 이름이 강감찬인 사람을 찾아서
# 이름과 학점을 Dataframe으로 출력하세요
df["이름"] == "강감찬" ## boolean mask 생성
display(df.loc[df["이름"] == "강감찬",["이름","학점"]])


# 2. 학점이 2.5 초과 3.5 미만인 사람을 찾아서 학과와 이름 출력하세요
display(df.loc[(df["학점"] > 2.5) & (df["학점"] < 3.5),["학과","이름"]])

# 3. Grade라는 컬럼을 추가한 후 학점이 4.0이상인 사람을 찾아
# 해당 사람만 grade를 'A'로 설정
df.loc[df["학점"] >4.0,"grade"] = 'A'
display(df)

## dataFrame 조작을 위해 간단한 Dataframe을 생성
# 사용하는 dataframe의 value 값은 [0,10) 범위의 난수형 정수
# 균등 분포에서 추출해서 사용
# 6행 4열짜리 dataframe 생성
import numpy as np
import pandas as pd


np.random.seed(0)
df = pd.DataFrame(np.random.randint(0,10,(6,4)))

# 컬럼 : ["A","B","C","D"]
df.columns = ["A","B","C","D"]
# index : 날짜를 이용 (2019-01-01부터 하루씩 증가)
df.index=pd.date_range("20190101",periods = 6)

# NaN을 포함하는 새로운 컬럼 "E"를 추가
# [7,np.nan, 3, np.nan, 2, np.nan] 데이터 추가

df["E"] = [7,np.nan, 3, np.nan, 2, np.nan]
display(df)

##########################
# 결측값 처리
#########################
# 결측값을 제거(NaN이 포함된 row를 제거)

# df.dropna(how="any",inplace=True)
# display(df)

# 결측값을 다른 값으로 대체
df["E"] = [7,np.nan, 3, np.nan, 2, np.nan]
# df.fillna(value=0,inplace=True)
display(df)

# 결측값을 찾기위한 mask
display(df.isnull())

# 간단 예제
## "E" 컬럼의 값이 NaN인 모든 row를 찾고
## 해당 row의 모든 columns의 값을 출력

display(df.loc[df["E"].isnull(),:])

```

# 이론

- 평균(mean) : 수학적 확률의 기댓값

  ​					   어떤 사건을 무한히 반복했을 때

  ​					   얻을 수 있는 평균으로서 기대할 수 있는 값

- 편차 (deviation): 확률변수 X와 평균값의 차이

  ​							 데이터의 흩어진 정도를 수치화 하기에는 적합하지 X

  ​							 (편차의 합이 0이기 때문)

- 분산 (Variance) : 

  데이터의 흩어진 정도를 알기 위해 사용되는 편차의 제곱의 평균

​       제곱을 사용했기 때문에 단위표현이 애매해지는 경우 존재

- 표준편차 (standard deviation) : 분산의 제곱근

- 공분산 (covariance ) : 두개의 확률변수에 대한 관계를 보여주는 값

​       두개의 확률 변수에 대한 공분산 값이 양수 : 비례관계 

​       하나의 확률 변수가 증가할 때 다른 확률변수도 증가하는 경향

​       두개의 확률 변수에 대한 공분산 값이 음수 : 반비례 관계 

​       하나의 확률 변수가 증가할 때 다른 확률변수는 감소하는 경향

​       공분산 값이 0 이면 두 변수가 서로에게 영향을 끼치지 않는 독립적인 상태

​       두 관계의 연관성은 공분산으로 알 수 없음

그래서 나온 개념 : 상관관계

- 상관관계(correlation) : 두 대상이 서로 연관성이 있다고 추측되는 관계

  성적 vs 자존감 

  온라인 게임 vs 폭력성

- 상관계수 (correlation coefficient) : -1 과 1 사이의 실수

  0에 가까울 수록 연관성이 없다고 판단

  절댓값이 1에 가까울수록 밀접한 연관이 있다고 판단

  상관관계는 인과관계를 설명할 수 없음

