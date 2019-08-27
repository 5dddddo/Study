# 0827_Github 특강 1

## Install

- vscode Install
  1. <https://code.visualstudio.com/Download> - [System Installer 64bit] Download
  2. 추가 작업 선택

    ![1566890322138](https://user-images.githubusercontent.com/50972986/63769800-a7981d80-c90e-11e9-92a1-b8ff5bda2e88.png)

<br>

- Git Install

<br>

## 실습

## 1. Add/Commit/push하기

![1566906433895](https://user-images.githubusercontent.com/50972986/63769797-a36c0000-c90e-11e9-9827-5d65ea508cae.png)

- Git bash 실행

  1. 실습을 위한 dir 생성

     `$ mkdir NEW_DIR_NAME`

  2. 새 dir로 이동

     `$ cd NEW_DIR_NAME`

  3.  Code Editor 실행

     `$ code .`

  <br>

- vscode 실행 됨

  1. 새 파일 생성

     ![1566891231073](https://user-images.githubusercontent.com/50972986/63769579-23459a80-c90e-11e9-9b93-609e5117f62e.png)

  2. hello.py 작성

     ![1566891347730](https://user-images.githubusercontent.com/50972986/63769586-2771b800-c90e-11e9-820d-2283f8e09549.png)

<br>

- github에 commit 해보자!

  1. git 시작하기

     `$ git init`

  2. git의 상태 확인

     `$ git status` 

     <br>

     - 현재 hello.py는 추적이 불가능

      ![1566892762767](https://user-images.githubusercontent.com/50972986/63769776-93ecb700-c90e-11e9-8766-e5d8d9f51f9a.png)

  <br>

  3. 파일 추가하기

     `$ git add hello.py`

     ![1566892938631](https://user-images.githubusercontent.com/50972986/63769594-2d679900-c90e-11e9-9440-94220b9d098a.png)

  <br>

  4. commit

     `$ git commit -m "커밋 내용"`

      ![1566893013120](https://user-images.githubusercontent.com/50972986/63769644-4bcd9480-c90e-11e9-8ba4-9973288161b1.png)

  <br>

- Github에서 새 Repositoriy 생성

  1. [New repository] 클릭

     ![1566893155377](https://user-images.githubusercontent.com/50972986/63769648-4d975800-c90e-11e9-91b7-0ea2c313ffde.png)

  <br>

  2. 새 Repositoriy 이름만 지정하고 [Create repository] 클릭

  <br>

     ![1566893220127](https://user-images.githubusercontent.com/50972986/63770181-8b48b080-c90f-11e9-90f0-d590a7b12bd2.png)

  <br>

  3. Repository가 생성되면 뜨는 화면

     ![1566893405746](https://user-images.githubusercontent.com/50972986/63770258-b8955e80-c90f-11e9-8e2d-683dbe560a3e.png)

  4. Git bash에서 새로 만든 repository를 origin이란 이름을 가진 원격 저장소로 추가하기

     ` $ git remote add origin https://github.com/5dddddo/Example.git `

     <br>

  5. Repository에 push하기

     `$ git push -u origin master`

     ![1566893811441](https://user-images.githubusercontent.com/50972986/63770425-19249b80-c910-11e9-857d-9afc746ec3d4.png)

<br>

2. 시간여행 예제 - checkout 명령어 

- commit한 기록 확인하기

  `$ git log`

   ![1566893970191](https://user-images.githubusercontent.com/50972986/63770428-1c1f8c00-c910-11e9-97a2-d2f2140b8900.png)

  <br>

  - commit의 요약본을 sha(Secure Hash Algorithm)을 통해 암호화하여 저장

  - 2의 256의 값을 가지기 때문에 충돌 걱정 X

  - log option

    - git log --oneline : 모든 커밋에 대해 커밋 key값 7자리와 커밋 내용만 보기
    - git log -1 : 가장 최신 commit 1개에 대한 log만 보기

    <br>

- 과거의 commit 지점으로 이동하기

  `$ git checkout 과거커밋key값5자리입력 `

   ![1566894287273](https://user-images.githubusercontent.com/50972986/63770439-22156d00-c910-11e9-9dcc-bb60dd221089.png)

  - 하늘색 글자 ((f7481e4... )) 를 보면 현재 head의 위치를 알 수 있음
  - Head가 이동함

  <br>

- 최종 commit으로 돌아가기

  `$ git checkout master`

<br>

## ※ 주의할 점!

- Github에서도 파일 내용을 수정할 수 있지만

  push 과정에서 충돌 발생할 위험이 있기 때문에

  아직은 로컬 컴퓨터에서 수정하도록 함

- 문제

  ![1566906533926](https://user-images.githubusercontent.com/50972986/63770445-2772b780-c910-11e9-910a-e773e3a204a3.png)

- 해결 방법

 ![1566906643234](https://user-images.githubusercontent.com/50972986/63770449-28a3e480-c910-11e9-884c-4ae12cf3c9ae.png)

------

#### git 공부하기 유용한 사이트

- Git & Github 내용 정리 : <https://opentutorials.org/course/2708>

- Git 구조 : <https://git-school.github.io/visualizing-git/>

