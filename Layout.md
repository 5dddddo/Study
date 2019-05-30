# Layout

### Display

- 블럭 요소 : 브라우저의 한 줄을 모두 차지 
  - <h> , <p> , <ul> ,<li>, <table>, <pre>, <div>, <form>
  - display : inline 으로 인라인 요소처럼 표시 가능
- 인라인 요소 : 한 줄 안에 차례대로 배치
  - <a> , <img> , <strong> , <span>, <input> , <br>
  - display : block 으로 블럭 요소처럼 표시 가능
- display : none  은 해당 요소를 완전히 배제시킴
- display : hidden 은 요소는 존재하지만 보이지 않을 뿐



### 요소 위치 정하기

#### Position 

- top, bottom, left, right 속성들은 position 속성이 먼저 설정되지 않으면 동작하지 않음

- 위치 설정 방법 

  - 정적 static : 정상적인 흐름 (default)

    ​					 블럭은 상하로 인라인 좌우로 배치

    ​					 top, bottom, left, right 속성 영향 받지 X

  - 상대 relative : 정상적인 위치가 기준

    ​						 상대 위치로 설정된 요소는 다른 요소와 겹칠 수 있음

    ​						 컨네이너는 상대적으로 배치해야 함. static X

  - 절대 absolute : 컨테이너의 원점이 기준, 전체 페이지를 기준

    ​							top, bottom, left, right 속성을 offset으로 사용

    ​							ex ) right : 2px - 오른쪽으로 2px 이동 X

    ​								   					  페이지의 오른쪽 경계선에서 2px 이동

  - 고정 fixed : 브라우저에 상대적으로 위치, 스크롤해도 움직이지 않음



#### float

