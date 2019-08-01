# Database (08/01)

- Database : 데이터의 집합체
- 초창기 : 계층형 DB

​					  네트워크 DB 등장 -> 이론적인 측면은 좋았지만 실제로는 거의 사용되지 않음

- 중기 : 관계형 DB

​				  수학자가 논문 발표

​				  데이터를 테이블 형태로 저장  -> 관계해석, 관계대수 

​				  IBM이 해당 논문을 근간으로 DBMS를 구축 -> DB2의 시초가 되는 DBMS가 탄생

​				  이를 시작으로 모든 DBMS는 관계형 DBMS로 전환 됨

​				  1990년대 후반까지 잘 사용되다가 객체지향의 패러다임이 시작 

​						-> DB에서는 프로그램 언어와 다르게  객체 지향의 중요한 특성만 받아들여 

​							객체-관계형 DB로 발전 시킴



- 빅데이터 시대 : 3V (Volume : 양, Variety : 비정형, Veracity : 속도)

  ​						   비정형 데이터를 저장, 관리 할 때는 관계형 DB가 효율적이지 않음

  ​						   NoSQL 계열의 DB (몽고 DB)가 사용되기 시작

  

- DBMS(Database Management System) : DB를 관리, 사용하기 위한 SW의 집합

  ex) Oracle, DB2, MySQL, ...



- Transaction : 작업(일)의 논리적인 최소 단위

  - 특정한 단위 작업의 묶음을 Transaction으로 설정할 수 있음

    ex) 은행의 이체 업무는 Transaction으로 설정할 수 있음

    A 계좌 -> B 계좌 : 2000원 이체

    1. A 계좌에 돈이 충분한지 잔액 select
    2. A 계좌 잔액에서 2000원을 뺀 값으로 update
    3. B 계좌의 잔액을 selet
    4. B 계좌 잔액에 2000원을 더한 값으로 update

  

- 왜 Transaction을 설정할까? 

  => 프로그램적으로 구현/수행하기 힘든 ACID 기능을 DBMS로부터 제공받기 위해서 설정함

  - Atomicity (원자성) : Transaction으로 지정된 작업은 모두 성공하거나 하나도 하지 않은 상태로 관리되어야 함

  - Consistency (일치성) : Transaction이 종료된 후에 데이터의 일관성이 유지 되어야 함

  - Isolation (독립성) : Transaction이 걸려있는 resource에 대해서 Transaction이 종료될 때까지

    ​								 데이터에 대해 다른 동작/접근을 제한함

  - Durability (영속성) : Transaction의 처리 결과는 2차 저장소에 안전하게 저장되는 것을 보장함

