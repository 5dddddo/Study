HTML : 문서의 내용 (data)

CSS : 문서의 스타일 (UI)

javascript : 문서의 동작 (control)



#  CSS 

##### (cascading style sheets)

- 우선 순위 : 외부 파일 >  head 내 정의 (내부) > 인라인 > 사용자 제공



### 선택자 : HTML 요소를 선택하는 부분

- 타입 선택자 : tag명

- 전체 선택자 : *

- ID 선택자 : #

- class 선택자 : .

- 속성 선택자 : [title, class, value ... = " " ] 

- 자손 선택자 

  : 선택자 a 선택자 b : a의 후손 중 b 모두 선택

  - table 내의 선택자를 자손 선택자로 사용할 때 주의!

  - 웹 브라우저 내에서 table 내에 tbody 생성하기 때문에

    table>tbody>tr>th 반드시 요렇게 적어야 함

- 자식, 형제 결합자 

  : 선택자 a > 선택자 b : a의 자식 중 b만 선택

  - 후손 모두 선택되기 때문에 tbody 없어도 상관 없음

- 반응 선택자 

  - :active - 특정 태그를 마우스로 클릭했을 때 수행
  - :hover - 특정 태그 위에 마우스 커서를 올리면 수행

- 구조 선택자 : 형제 관계에서 n번째 태그 선택

  - :first-child

  - :last-child

  - nth-child(수열) : 앞에서 수열번째

  - nth-last-child(수열) : 뒤에서 수열번째

    

### 속성

- font: font-family 속성을 이용해 설정

  ​				   @font-face

- text style : 
  - text-align : center,  justify (양쪽 정렬) , left, right
  - text-decoration
  - text-transform : capitalize, uppercase, lowercase
  - word-wrap : break-word : 영역에 맞게 자동 줄 바꿈

----------------------------------------------------

- ## Box

![1559130424970](C:\Users\student\AppData\Roaming\Typora\typora-user-images\1559130424970.png)

- < border >

  - border-style :  선 종류

  - border-radius : 1 2 3 4 반지름 - 둥근 경계 설정

    순서 : 1    2 

    ​		   4    3

  - 축약 

  - border : 두께 스타일(필수) 색상

  - resize : both horizontal, vertical, none - 박스 크기 조정 가능



- margin : 외부 영역에 공백 설정

- 가운데 정렬

  - ##### 인라인 요소 - text-align : center

    ##### 블록 요소 - margin : auto

    ​					width를 지정해야 함

- padding : 내부 영역에 공백 설정

  - padding : 상하 좌우 / 상 하 좌 우

- background 

  - background-color 
  - background-image : url('파일경로');
  - background-repeat :  image 여러 개 반복

  - background-attachment : scroll(default) fixed local
  - background-position : 배경 이미지 위치 설정



### 링크 스타일

- :link - 방문되지 않은 링크 스타일
- :visited - 방문된 링크 스타일
- :hover - 마우스가 위에 있을 때 링크 스타일
- :active - 마우스로 클릭될 때 링크 스타일

- 위치 순서 : link / visited - hover - active



### Table