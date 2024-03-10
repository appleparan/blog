---
layout: post
title: Environment Modules 사용하기
author: jongsukim
date: 2020-10-12 12:00:00 +0900
categories: [Programming]
tags:
  - Modules
  - Environments
  - Server Engineering
  - HPC
math: false
---

오랫동안 연구실에서 서버관리를 담당해왔다. 처음에는 단순 서버 on/off 정도만 하는 일이었는데, 별것도 아닌걸로 업체 통하면 시간도 오래걸리고 설명하기도 복잡하다보니 내가 처리하게 되었다. 그러다보니 점점 일이 늘어나서 서버 구축포함 대부분의 이슈를 내 선에서 해결하고자 노력하고 있다.

그런데 처음에는 다들 Fortran만 써서 상관없었지만 연구주제가 다양해지면서 다양한 환경을 구축할 필요가 생겼다. 다른 컴파일러, 다른 언어, 다른 서버 등등. 당연히 환경변수를 건드려야할 일이 많았다. 하지만, 유저들 대부분 환경변수가 뭔지도 모르는 사람들이다. 그래서 그동안 사용한 방법은 curl을 통해 미리 작성된 .bashrc를 받게 하는 것이었다. 디폴트 bashrc를 변경할 수도 있었겠지만, 상황에 따라 각자의 bashrc를 업데이트해야하는 경우도 생겨서 좀 더 안전하게 가려고 했다. 이 방법은 유연하게 대처하기 힘들고 특이사항이 생기면 내가 직접 수정해줘야하고, 환경변수가 지속적으로 append되는 경우 불필요하게 환경변수가 중복되는 경우가 생기기도 했다. 그러나, bashrc를 직접 건드리다가 PATH 같은 변수들을 날려먹고 나한테 찾아오는 경우가 종종 있었기에 나쁘지 않다고 생각했다.

그러다가 [KISTI 누리온 매뉴얼](https://www.ksc.re.kr/gsjw/jcs/hd#docout-0)을 접할 기회가 생겼는데, `module`이라는 걸 쓴다는걸 알게 되었다. 우리 연구실보다도 훨씬 많은 사용자와 많은 환경을 지원해야하는 KISTI에서 각자 환경에 맡게 module을 load해서 쓰는 방법이었다. 패키지 페이지는 다음과 같다.

* [Environment Modules](http://modules.sourceforge.net/)

홈페이지 소개란에 있던 이 패키지의 사용목적은 정확히 내가 원하는 그것이었다.

> Typically users initialize their environment when they log in by setting environment information for every application they will reference during the session. The Environment Modules package is a tool that simplify shell initialization and lets users easily modify their environment during the session with modulefiles.

어차피 현재까지 연구실 서버에서 쓰는 환경이 compiler 버전 변경, `Python` 사용시 `.pyenv` or `Anaconda` 방식 변경, `CUDA` 사용 여부, `ANSYS` 사용여부 정도라서 `PATH`와 `LD_LIBRARY_PATH` 정도만 append 하는 정도라 복잡한 기능을 쓸 필요는 없었다. 그래서 단순하게 사용법만을 소개해둔다.

## 사용법

`module`의 사용법은 쉽다. 유저들에겐 아래 명령어들만 숙지시키면 된다.

* `module avail` : 사용가능한 모듈 보기 (줄여서 `module av`)
* `module load MODULENAME` or `module add MODULENAME` : module avail로 확인한 특정 `MODULENAME` load
* `module unload MODULENAME` or `module rm MODULENAME` : module avail로 확인한 특정 `MODULENAME` unload
* `module list` : 현재 사용중인 모듈 출력
* `module purge` : 현재 사용중인 **모든** 모듈 삭제

Job scheduler 사용시 `module purge` 시키고, 다음 예시와 같이 필요한 module을 `load` 하도록 했다.

  ```
  module purge
  module load gcc/10.1
  module load gcc/10.1/fftw3/3.3.8
  ```

## 설치

1. [modules의 INSTALL 문서](https://modules.readthedocs.io/en/latest/INSTALL.html#installation-instructions) 를 참고해서 모듈을 설치한다. 일반적인 소스컴파일 과정인 `./configure`, `make`, `make install`을 따른다. CentOS의 경우 심플하게 `yum install environment-modules` 써도 된다.
2. Configuration section을 따라 initialization을 실행한다. 나는 유저들의 bashrc에 `source PREFIX/init/bash` 를 넣는걸 선호했다. (`bash` 사용시) `PREFIX`는 default가 `/usr/local/Modules` (소스 컴파일 default 값)이거나 `/usr/share/Modules`(`yum 설치시`)로 잡힌다.
3. Configuration에서 default로 불러들일 `module path`와 `modulefiles`를 지정하는 파트가 있는데 사용자들이 어떤걸 쓸지 어떻게 알고 정하나 싶어서 나는 지정하지 않았다.

## How it works

`Modules`의 원리는 다음과 같다. 

* 지정된 위치 (디폴트는 `/usr/local/Modules/modulefiles`)의 `modulefiles`를 읽고 그 `modulefiles`를 읽어서 필요한 환경변수를 추가하거나 삭제하는 것.
* modulefiles 디렉토리내의 서브 디렉토리는 `module list`에서 `/`로 처리된다. (i.e. `gcc/10.1/fftw3/3.3.8`은 `PREFIX/modulefiles/gcc/10.1/fftw3/3.3.8`라는 모듈 파일이 존재하는 것이다) 실제로 가보면 예제들이 몇개 있는데 대부분 참고용이니 다른디렉토리에 복사해두고 지웠다.

## modulefile 생성하기

그럼 제일 중요한 modulefile은 어떻게 구성되어있냐 하면, [자체 문법](https://modules.readthedocs.io/en/latest/modulefile.html)이 있다. 하지만 이걸 다 알고 쓸 필요는 없다. 일반적인 쉘 스크립트를 modulefiles로 자동으로 변환해주는 파이썬 스크립트가 패키지 안에 때문!

예를 들어,

1. gcc 사용을 위해 환경변수 `PATH`와 `LD_LIBRARY_PATH`가 필요하다고 하자.

    ```
    export PATH=/APP/gcc/10.1/bin:$PATH
    export LD_LIBRARY_PATH=/APP/gcc/10.1/bin:$LD_LIBRARY_PATH
    ```

    가 필요하다고 하자. 이 부분만을 `_bashrc_gcc10`이라고 저장한다음에

2. 이를 다음 명령어를 통해 modulefile로 출력한다. 

    ```
    PREFIX/bin/createmodule.sh _bashrc_gcc10 > modules_gcc10
    ```

    이라고 하면 modulefile `modules_gcc10`이 만들어진다. python script도 있다.

    ```
    python PREFIX/bin/createmodule.py _bashrc_gcc10 > modules_gcc10
    ```

3. 이렇게 만들어진 modulefile을 `modulefiles` 위치에 복사하면 끝. 구조적으로 관리가 필요하다면 modulefiles 디렉토리를 환경에 따라 디렉토리 구조로 바꿔주면 된다.

    ```
    cp modules_gcc10 PREFIX/modulefiles/gcc10
    ```

4. 바로 `module avail`을 통해 확인할 수 있다. (reboot 불필요)

    ```
    ---------------- PREFIX/modulefiles ----------------
    gcc10
    ```

5. 삭제도 그냥 single `modulefile`이나 서브 디렉토리를 삭제하면 된다.

참 쉽죠?

## Pros and Cons

### Pros
* 사용자들이 위험하게 .bashrc나 .bash_profile을 건드릴 필요가 없다.
* PATH등이 불필요하게 append돼서 duplicate될 일이 없다.
* 관리자 입장에서 module의 추가, 삭제 및 수정이 매우 쉽고 편리하다.
* 문서의 Cookbook section을 보면 알겠지만, 다양한 방식으로 사용자화가 가능하다.
* 제일 중요한것, **유저들이 사용하기 편리하다.**

### Cons
* 아직까지 단점을 모르겠다.