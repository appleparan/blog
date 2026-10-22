---
layout: post
title: Apptainer Setup Guide
author: jongsukim
date: 2024-05-23 00:01:00 +0900
categories: [Server Engineering, HPC]
tags:
  - Apptainer
  - Server Engineering
  - HPC
---

**2024년 5월 기준으로 설명한 글임을 명시한다.**

## Introduction

Reproducible Research의 일환으로 많은 사람들이 컨테이너를 이용한 가상화 기술을 활용하고 있다.
이를 위해서 보통 Docker기반의 컨테이너를 많이 사용한다.

Docker는 서비스를 올릴때는 더할나위 없이 좋은 툴이지만,
Multi-user가 있는 HPC환경에서는 적합하지 않다.
왜냐하면 HPC는 개개인이 별도의 시스템을 사용하는 것이 아닌
네트워크를 통한 스토리지 서버를 구축하여 다수의 유저가 이를 공유해서 사용한다.

즉, previleged user가 아닌 유저가 사용하는 경우가 대부분이다.

하지만 Docker 컨테이너의 기본 유저는 root이기 때문에 Docker 컨테이너 내부에서 스토리지를 마운트해서 작업한다고 해보자
Docker 내부에서 쓴 파일들은 스토리지 밖에서는 권한문제때문에 쉽게 수정할수도 없고,
더욱 위험한건 root권한을 가지고 다름 사람들의 디렉토리를 건드릴 수도 있다.

물론 Docker의 유저를 바꾸면 된다.
그러나, 이런 경우 같은 내용의 이미지일지라도 유저가 다르다는 이유하나만으로 여러개의 이미지를 만들 수 밖에 없다.

이런 문제로 HPC를 위한 컨테이너인 Singularity라는 컨테이너 기술이 생겼고, 이제는 Linux Foundation안에서 Apptainer라는 이름을 사용하고 있다.

이 포스트에서는 Docker에 비해 널리 알려지지 않은 Apptainer의 설치와 사용방법을 소개하고자 한다.
이전 포스트에서 설명한 [Slurm](https://blog.liam.kim/posts/2024/05/Slurm-Setup-Guide/)과 결합하면
많은 ML연구자들과 HPC와 연관된어 있는 연구자들이 도움을 많이 받을 것이라 본다.

## Basic Concepts of Apptainer



## Install Apptainer

1. [시스템 의존성 패키지](https://github.com/apptainer/apptainer/blob/main/INSTALL.md#install-system-dependencies)들을 설치한다. 어차피 prebuilt package 쓸거라 큰 상관은 없어보이기는 하지만, fakeroot같은 추가기능을 위해서 설치하면 좋을 것 같다.
    ```shell
    # Ensure repositories are up-to-date
    sudo apt-get update

    # Install debian packages for dependencies
    sudo apt-get install -y \
        build-essential \
        libseccomp-dev \
        pkg-config \
        uidmap \
        squashfs-tools \
        fakeroot \
        cryptsetup \
        tzdata \
        curl wget git \
        autoconf \
        automake \
        libtool \
        pkg-config \
        libfuse3-dev \
        zlib1g-dev \
        libssl-dev  \
        uuid-dev
    ```

2. Apptainer는 여러가지 추가기능이 있다. 그 중에서도 개인적으로 생각했을 때 중요한 기능은 다음과 같다. [공식 문서](https://apptainer.org/docs/admin/main/installation.html#system-requirements)
    * unprevileged user namespace : non-previleged user가 컨테이너를 실행할 수 있게 한다.
    * fakeroot : 컨테이너 내부에서는 root처럼 작동해서 패키지 설치들을 가능하게 한다.

3. 또한 Apptainer에는 두 가지 모드가 있다. 여기서는 보통 계산을 돌리는 용도기 때문에 sandbox모드를 쓸 이유가 없고, SIF파일을 사용한다고 가정한다. SIF파일을 써야 fakeroot를 사용할때 제약이 많이 없어진다.
    * sandbox : 컨테이너를 수정할 수 있는 모드
    * SIF File : 읽기전용 모드. 패키지 설치등은 할 수 있지만 컨테이너를 다시 만들면 초기화

4. 이제 본격적으로 설치해보자. PPA가 따로 있어서 매우 심플하다.
Apptainer로 루트 권한이 필요한 작업을 많이 하기 때문에 setuid 기능을 사용할 것이고 이를 위해 `apptainer-suid`를 설치한다.
    ```shell
    sudo apt update
    sudo apt install -y software-properties-common
    sudo add-apt-repository -y ppa:apptainer/ppa
    sudo apt update
    sudo apt install -y apptainer-suid
    ```

5. 버전을 확인한다.
    ```
    $ apptainer --version
    apptainer version 1.3.1
    ```

## Build and Push Image

Apptainer의 자체 문법을 사용해도 되지만, 딥러닝을 위해 NVIDIA Docker image를 활용해보려고 한다.

1. 예를 들어 다음과 같은 ngc이미지로 만드는 Dockerfile이 있다고 해보자.

    ```Dockerfile
    FROM nvcr.io/nvidia/tensorrt:24.04-py3

    # 필수 패키지 업데이트 및 설치
    RUN apt-get update && apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        libssl-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        xz-utils \
        tk-dev \
        libffi-dev \
        liblzma-dev \
        libgdbm-dev \
        libxml2-dev \
        libxmlsec1-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

    # Python 소스 다운로드 및 빌드
    ENV PYTHON_VERSION=3.12.3
    RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz \
        && tar -xf Python-$PYTHON_VERSION.tgz \
        && cd Python-$PYTHON_VERSION \
        && ./configure --enable-optimizations \
        && make -j $(nproc) \
        && make altinstall \
        && cd .. \
        && rm -rf Python-$PYTHON_VERSION Python-$PYTHON_VERSION.tgz

    # 심볼릭 링크 설정
    RUN ln -s /usr/local/bin/python3.12 /usr/bin/python3 \
        && ln -s /usr/local/bin/pip3.12 /usr/bin/pip3

    # 작업 디렉토리 설정
    WORKDIR /app

    # 필요한 패키지 설치 (여기서는 예시로, 실제로 필요한 패키지를 대체하세요)
    COPY requirements.txt requirements.txt
    RUN pip3 install --no-cache-dir -r requirements.txt

    # 애플리케이션 코드 복사
    COPY . .
    ```

2. 이걸 Docker 이미지로 먼저 빌드한다.
    ```shell
    docker buildx build -t hello:0.1 . -f Dockerfile
    ```

3. Docker 이미지로 Apptainer 이미지(SIF파일)로 변환한다.
    ```
    apptainer build hello_apptainer_v0.1.sif docker://hello:0.1
    ```

4. Apptainer로 OCI Artificats이기 때문에 필요한 경우 Harbor같은 OCI Registries에도 push 수 있다.
    ```
    apptainer push hello_apptainer_v0.1.sif oras://<harbor_URL>/hello_org/hello:0.1.0
    ```

5. Pull할때는 다음과 같이 하면 된다.
    ```
    apptainer pull --name <DOWNLOADED_SIF_FILENAME>.sif oras://<harbor_URL>/hello_org/hello:0.1.0
    ```

## Slurm Job Script

Apptainer의 중요한 옵션들을 알려주려고 한다.

1. `exec` : 어떤 명령어를 실행할때 사용하는 apptainer 명령어이다.
1. `--nv` : 이 옵션은 NVIDIA GPU를 사용하는 컨테이너를 실행할때 필요한 설정들을 자동으로 불러온다. 호스트의 드라이버, cuDNN같은 라이브러리도 자동으로 불러오기 때문에 컨테이너 안에서 추가로 설치할 이유가 없다.

2. `--bind` : 파일 시스템을 마운트할때 좋다. NFS를 마운트할때도 좋고, `data/`와 같이 공통적으로 쓰는 디렉토리를 마운트할 때 좋다.
    이러면, 코드에서는 항상 같은 폴더(예를 들면 `/data`)같은 prefix를 고정하고 실험을 돌릴 수 있게 된다.

3. `--env` : 환경변수를 설정할 때 좋다. job script에 다음과 같이 설정하고 사용한다.
    ```shell
    #!/bin/sh
    PYENV_ROOT=$HOME/.pyenv

    apptainer exec --env PYENV_ROOT=$PYENV_ROOT blahblah.sif /bin/bash -c "python my_script.py"
    ```
4. `--fakeroot` : 앞서 설명한 것처럼 마치 root인것처럼 실행하게 해준다.

5. `--writable-tmpfs` : 임시 파일 시스템을 사용해 마치 컨테이너 내부 파일을 변경하는 것처럼 보이게 해준다.
    컨테이너가 종료되면 모든 변경사항들이 사라진다.

6. `/bin/bash -c '커맨드1; 커맨드2; 커맨드3'` : 여러개의 명령어를 실행하게 해준다.
    예를 들면, `pyenv`를 컨테이너 내부에서 쓰려면 `export PATH`와 `eval "$(pyenv init -)"`, `eval "$(pyenv virtualenv init -)"`와 같은 쉘 명령어를 전부 실행해야하는데 이를 커맨드1, 커맨드2 등에 매핑시켜서 진행할 수 있다.

7. 이렇게 해서 최종 스크립트를 살펴보면 다음과 같다.

```shell
#!bin/bash
#SBATCH --job-name=잡이름
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       # 프로세스 수 (MPI RANK 수 혹은 num_workers)
#SBATCH --cpus-per-task=1         # 프로세스 별 Thread 수 수
#SBATCH --mem=128GB               # 메모리 제한
#SBATCH --partition=파티션이름      # 파티션 이름
#SBATCH --gres=gpu:장치이름:장치수   # GRES 자원선택
#SBATCH --output=%x-%j.log        # 잡이름-잡넘버.log 형식으로 output 파일 생성

# uv를 통해 'hello_world.py' 실행
export CMD='uv run python hello_world.py'

SIF_FILE_PATH='/path/where_sif_file_exists.sif'
export DATA_DIR='$HOME/data'
export HF_HOME='$HOME/.cache/huggingface'
export UV_PYTHON_INSTALL_DIR='$HOME/.local/share/uv/python'
export UV_TOOL_DIR='$HOME/.local/share/uv/tools'

# 1. $HOME/data를 컨테이너 안에서 /data로 bind
# 2. $HF_HOME, $UV_PYTHON_INSTALL_DIR, $UV_TOOL_DIR 환경변수 전달
apptainer exec --nv \
    --bind $DATA_DIR:/data:rw \
    --bind /dev/shm:/dev/shm \
    --env $HF_HOME:$HF_HOME \
    --env $UV_PYTHON_INSTALL_DIR:$UV_PYTHON_INSTALL_DIR \
    --env $UV_TOOL_DIR:$UV_TOOL_DIR \
    --writeable-tmpfs \
    $SIF_FILE_PATH \
    /bin/bash -c 'eval ". $HOME/.local/bin/uv/env";
    eval "\$CMD";'
```

8. 이를 `job.sh`라고 저장한다면 다음과 같이 `sbatch` 명령어를 통해 실행시킬 수 있다.
```shell
sbatch job.sh
```
