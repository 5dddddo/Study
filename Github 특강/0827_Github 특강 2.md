# 0827_Github 특강 2

## 실습

## 3. 끝말잇기 예제 - 공유 repository 사용하기

![1566907249278](https://user-images.githubusercontent.com/50972986/63770605-87695e00-c910-11e9-90a0-7460977bc95e.png)

#### [ User 1 ] 

- Github에 새 repository ggutmal 생성

- Git Bash

  1. dir 생성

     `$ mkdir ggutmal`

  2. 끝말잇기를 위한 .md 파일 생성

     `$ code .`  => README.md 생성

     ![1566896322876](https://user-images.githubusercontent.com/50972986/63770617-8cc6a880-c910-11e9-878c-0f2b7cfff017.png)

  3. git 시작하기

     `$ git init `

  4. 변경된 파일 추가

     `$ git add file.md`

  5. commit

     `$ git commit -m "끝말잇기 시작"`

  6. Github에서 만든 repository를 원격 저장소로 지정

     `$ git remote add origin https://github.com/5dddddo/ggutmal.git`

  7. Githup에 push

     `$ git push -u origin master`

     ![1566896549115](https://user-images.githubusercontent.com/50972986/63770631-9223f300-c910-11e9-8b95-7cf568f6d65c.png)

     <br>

  8.  Github - [ggutmal Repository] - [Settings] - [Collaborators] - 상대방ID 추가

     

<br>

#### [ User 2 ]

- 메일에서 Collaborators 수락

![1566895676218](https://user-images.githubusercontent.com/50972986/63770644-9a7c2e00-c910-11e9-9f51-6a874d1d8e9e.png)

- Git Bash

  1. [ User 1] 이 생성한 repository 복사하기

     `$ git clone https://github.com/5dddddo/ggutmal.git`

  2. 끝말잇기 후 push

     1. Code Editor 실행

        `$ code .`  : 코드 수정

     2. git 시작하기

        `$ git init `

     3. 변경된 파일 추가

        `$ git add file.md`

     4. commit

        `$ git commit -m "끝말잇기 시작"`

     5. push

        `$ git push -u origin master`

<br>

#### [ User 1 ] 

- 원격 저장소 ( Repository ) 가져오기

  `$ git pull origin master`

  ![1566896240227](https://user-images.githubusercontent.com/50972986/63770656-9fd97880-c910-11e9-8b9e-69142762b75f.png)

- [ User 2 ]의 2번 과정 반복

<br>

### 끝말잇기 예제의 문제점 : 단방향 소통, 동시성 X

### 							 해결책 : Branch

------

<br>

## 4. Branch & Merge 예제

### Branch

- Branch 생성

  `$ git branch NEW_BRANCH`

- 어느 branch에 위치햇는지, 어떤 branch가 있는지 확인

  `$ git branch`

  ![1566887208677](https://user-images.githubusercontent.com/50972986/63770932-327a1780-c911-11e9-8a53-328e5a760f3d.png)
  - 현재 master branch에 위치

<br>

- branch 이동 

  ​	: branch 이동이나 이전 commit으로 이동하는 것 모두 checkout 명령어 사용

  ![1566887359349](https://user-images.githubusercontent.com/50972986/63770985-4291f700-c911-11e9-850e-dc823958f944.png)

  - branch option

    - `$ git checkout -b NEW_BRANCH` : 생성과 동시에 새 branch로 이동

    - `$ git branch -d 브랜치이름` : branch 삭제

<br>

- 다른 branch의 내용을 볼 수 없음
- 완전히 분리된 공간

<br>

### Merge

- Comment branch의 hello.py 저장

  ![1566904775804](https://user-images.githubusercontent.com/50972986/63770715-b7b0fc80-c910-11e9-8ddf-2f25de357b73.png)

- master branch의 hello.py 저장

  ![1566904808924](https://user-images.githubusercontent.com/50972986/63771045-5ccbd500-c911-11e9-95a1-b0855a33bc6b.png)

- 현재 branch의 위치를 master로 

  `$ git checkout master`

  - master를  항상  중심으로 생각하고 다른 Branch를 병합

![1566888395053](https://user-images.githubusercontent.com/50972986/63771065-66edd380-c911-11e9-8317-9521a1ff9a5d.png)

<br>

- merge하려는 두 branch가 다른 파일을 수정했으면 auto merge 되지만 동일한 파일을 수정한 후 merge하면 충돌 발생

  - 선택

  1. master의 수정 내용으로 덮어쓰기

   	2. comment branch의 수정 내용으로 덮어쓰기
   	3. 둘 다 살리기

<br>

- Accept Both Changes 클릭 : 둘 다 살리기

![1566888493334](https://user-images.githubusercontent.com/50972986/63771085-70773b80-c911-11e9-8020-a494ddec1eb2.png)

<br>

- merge한 결과 

![1566888629094](https://user-images.githubusercontent.com/50972986/63771098-78cf7680-c911-11e9-82e5-3efb3d3c8e98.png)

<br>

- 두 branch를 merge를 하면 log 기록도 합쳐지는 것을 볼 수 있음

![1566888650649](https://user-images.githubusercontent.com/50972986/63771121-8127b180-c911-11e9-9115-7e44844bd6f9.png)

![1566888726940](https://user-images.githubusercontent.com/50972986/63771154-8dac0a00-c911-11e9-920c-5abd14d885d3.png)





ctrl + l : 콘솔 clear  여러줄 삭제 / ctrl + u 한줄 삭제



------

#### git 공부하기 유용한 사이트

- Git branching 실습 : https://learngitbranching.js.org/?locale=ko

- <http://git-school.github.io/visualizing-git/>

  git 쿰척 확인

  

