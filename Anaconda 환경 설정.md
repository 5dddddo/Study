# Anaconda 환경 설정

1. python 3.7 설치

2. anaconda prompt 관리자 권한 실행

3. python -m pip install --upgrade pip

   pip 최신 버전으로 업그레이드

4. 가상환경 생성

   나중에 사용할 tensorflow는 python 3.6 버전에서 동작

   conda create -n 새 가상환경이름

   ex ) conda create -n cpu_env python python=3.6 openssl

5. nb_conda 설치 : conda install nb_conda
6. activate cpu_env : 생성한 가상환경으로 전환
7. jupyter notebook : 개발환경 실행

​								  내장 웹서버가 동작

​		만약, 개발환경(크롬)에서 우리 가상환경이 안 보일 경우

​		python -m ipykernel install --user --name 등록할가상환경 --display--name[CPU_ENV]

8. jupyter notebook --generate-config : working directory 설정

9.  C:\Users\student\.jupyter로 가서 생성된 CONFIG 파일의 c.NotebookApp.notebook_dir = 'C:/python_ML'로 변경

   