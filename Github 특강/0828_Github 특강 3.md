# 0828_Github 특강 3

MSA ( Micro Service Architecture ) <-> Monolithic

![1566988201309](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566988201309.png)

python 설치

![1566956479133](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566956479133.png)

환경변수 설정 체크 

install now

![1566956664248](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566956664248.png)

- warning 해결

![1566956699432](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566956699432.png)

- 초경량 Framework : fla

![1566956750599](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566956750599.png)



![1566957126208](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566957126208.png)

![1566957110312](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566957110312.png)



![1566957197863](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566957197863.png)

server run

![1566958312123](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958312123.png)

![1566958320212](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958320212.png)



![1566958405378](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958405378.png)

![1566958417634](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958417634.png)

![1566958555726](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958555726.png)



![1566958701677](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958701677.png)



![1566958664293](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958664293.png)



![1566958684424](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958684424.png)



![1566958731944](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566958731944.png)







![1566959930902](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566959930902.png)

글자를 제곱할 수 없기 때문에 str -> int 형변환

flask 함수에서 return은 무조건 str이어야 함

![1566959113667](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566959113667.png)





로또







![1566970338593](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566970338593.png)

![1566966078511](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566966078511.png)







.gitignore 파일 추가해서 추적하지 않을 파일명 추가

commit msg : 명령형으로 쓰자



- git tagging

  - git tag : tag 조회하기
  - git tag  -a (annotated) TAG_NAME -m (msg) "설명" (커밋해시,이름표)
  - git tag -d TAG_NAME : TAG 삭제
  - git checkout TAG_NAME
    - v1

  ![1566969354791](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566969354791.png)
  
  - ​	v2

![1566969368606](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566969368606.png)

![1566969854068](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566969854068.png)





로또

- python을 대신해서 요청을 보내줌

  ![1566970189510](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566970189510.png)









![1566970302365](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566970302365.png)

```python
'''
requests를 통해 동행 복권 API예 요청을 뽀냬어
1등 번호를 가져와 python list로 만듦
'''

import requests

# 1. requests 통해 요청 보내기
url = 'https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=873'
response = requests.get(url)
print(response.text)
```



![1566970571077](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566970571077.png)

dictionary ( {key:value,... } )형태로 출력 

``` python
# .json() : dictionary로 결과값 return
res_dict = response.json()
print(res_dict)
```

![1566970808246](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566970808246.png)



``` python
# 1등 번호 6개 담긴 result list 출력
result = []
result.append (res_dict['drwtNo1'])
result.append (res_dict['drwtNo2'])
result.append (res_dict['drwtNo3'])
result.append (res_dict['drwtNo4'])
result.append (res_dict['drwtNo5'])
result.append (res_dict['drwtNo6'])

for i in range(6):
    result.append (res_dict[f'drwtNo{i+1}'])

    
print(result)
```



![1566971080265](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566971080265.png)

-----------------------

비트코인 현재가

``` python
import requests

# 1. requests 통해 요청 보내기
url = 'https://api.bithumb.com/public/ticker/'
response = requests.get(url)
# print(response.text)

# .json() : dictionary로 결과값 return
res_dict = response.json()
print(res_dict)

# data 안의 opening_price 변수 추출
result = res_dict['data']['opening_price']
print(result)
```



![1566973582994](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566973582994.png)



---------------

![1566973636546](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566973636546.png)

commit 을 순서대로 하고싶을때 , 스냅샷 따로 찍고픔!

-> add를 따로 따로



![1566973953353](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1566973953353.png)



### telegram 예제

1. 텔레그램가입

2. BotFather

   /newbot

   user이름 입력

   bot이름 입력

3. user이름으로 검색

4. 사용할 method 

   getMe

   getUpdates

   sendMessage

5. <https://api.telegram.org/bot<token\>/getMe>

https://api.telegram.org/bot\<token\>/sendMessage?chat_id=939575516&text=고통없쉬

## chatbot 예제

mkdir chatbot









#### git 공부하기 유용한 사이트

- [https://git-scm.com](https://git-scm.com/)

- gitignore 파일 생성해주는 사이트 

  <http://gitignore.io/api/java>

  | 동행로또 API          | <https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=1> |
  | --------------------- | ------------------------------------------------------------ |
  | JSON Viewer           | <https://chrome.google.com/webstore/detail/json-viewer/gbmdgpbipfallnflgajpaliibnhdgobh?hl=ko&hc_location=ufi> |
  | Bithumb Open API      | <https://apidocs.bithumb.com/docs/ticker>                    |
  | 논산훈련소 Tip Github | <https://github.com/krta2/awesome-nonsan>                    |
  | telegram web          | <https://web.telegram.org>                                   |
  | telegram api          | https://api.telegram.org/bot<token>/METHOD_NAME              |
  | telegram sendMessage  | https://api.telegram.org/bot<토큰>/sendMessage?chat_id=<나의chat_id>&text=<내용> |
  | chatbot 코드          | <https://github.com/edu-john/t4ircc_chatbot>                 |