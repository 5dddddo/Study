# Linux 기본 명령어 (06/26)

#### 시스템 종료
- shutdown -P now
- halt -p
- init 0(runlevel)

#### 재부팅
- shutdown -r now
- reboot
- init 6(runlevel)

#### 로그아웃
- exit
- logout

##### => ROOT 권한에서 실행 가능한 명령어



#### 가상 콘솔
CTRL + ALT + F1~F6 : 가상 콘솔 실행

##### => CentOS 에서 최대 6명의 사용자가 로그인 할 수 있음.



#### RunLevel : 시스템이 가동되는 방법

- Power Off - 0 : 종료
- Rescue - 1 : 시스템복구모드
- Multi-User - 2~4 : Text 기반 다중 사용자 모드
- Graphical - 5 : 그래픽 기반 다중 사용자 모드
- Reboot - 6 : reboot

----------------------------------------------------------------------

- ls -al runlevel* : /lib/systemd/system 안의 runlevel* 을 ls로 출력:

- /etc/systemd/system/default.target : 처음 부팅 시 어떤 runlevel로 실행할지를 지정해주는 링크

  ​		위 링크의 초기값은 graphical.target으로 GUI(로그인 form)로 부팅된다.

-  ln(링크 생성) -sf(심볼릭 링크) 지칭 되어 링크 거는 곳/  링크 걸 곳 : 첫 부팅 시 runlevel 링크 변경

  - ln -sf /lib/systemd/system/multi-user.target /etc/systemd/system/default.target

  ​		: 부팅을 CLI 기반으로 변경

  - ln -sf /lib/systemd/system/graphical-user.target /etc/systemd/system/default.target : 부팅을 GUI 기반으로 변경



#### 기본 명령어

- pwd: print working directory
- cd: change directory
- ls: list(현재 디렉토리 안의 파일/ 디렉토리 목록 출력)

