@echo off
REM MSVC 14.29 환경 세팅
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64 -vcvars_ver=14.29

REM Conda 환경 활성화
call conda activate util_project

REM Python 코드 실행
python "C:\Users\wjdtm\Desktop\workspace\python_cuda\main.py"

pause
