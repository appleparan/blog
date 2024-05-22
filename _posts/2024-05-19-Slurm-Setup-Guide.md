---
layout: post
title: Slurm Setup Guide
author: jongsukim
date: 2024-05-19 00:01:00 +0900
categories: [Server Engineering, HPC]
tags:
  - Slurm
  - Server Engineering
  - HPC
---

**2024년 5월 기준으로 설명한 글임을 명시한다.**

## Introduction

난 Jupyter를 싫어한다.

실험과 Literate programming 관점에서는 최적의 툴이지만, 대부분 잘못 사용한다고 생각한다. 많은 data scientist들이 작성하는 Jupyter 코드들은 문서는 없고 코드만 있다. 특히 industry에서 production을 위한 코드를 Jupyter로 짜는 것은 무책임하다고 생각하고 있다.
게다가 GPU관리 측면에서 Jupyter가 GPU 자원을 낭비하는 만악의 근원이라고 생각하고 있다.
Google Colab처럼 timeout도 정해두고 잘 관리되면 괜찮지만, 대부분은 시간별로 할당하고 사용자가 GPU를 알아서 반납할때까지 점유하도록 한다.
코드를 작성하거나 편집할때는 GPU자원이 필요한 것이 아니기 때문에 실제 GPU가 작동하는 시간은 점유하고 있는 시간에 비해 적을 수 밖에 없다.

그러기에 최대한 오래 점유하려고 하고, 다른 사람들은 해당 사용자가 GPU반납하기만을 기다리게 된다.
다른 사용자들은 이런 경험을 겪은 후에는 본인도 오래 점유하려고 하고, 결국 악순환으로 인해 언제나 GPU는 부족하게 된다.
가게로 치면 GPU 회전율이 낮은것이다.

이를 해결하는 방법 중 하나가 job scheduler를 이용하여 batch system을 쓰는것이다.
batch system을 쓰는 것은 interactive하게 코드를 작성하는 Jupyter보다 사용하기에는 조금 더 어려울 수 있겠지만,
필요할 때만 GPU를 할당받고 효율적으로 사용할 수 있다.
참고로 로깅이나 plot을 실시간으로 보는 interactive함을 원한다면 [Weight & Biases](https://wandb.ai/site)나 [Tensorboard](https://www.tensorflow.org/tensorboard)같은 툴을 사용하면 되고, 디버깅은 디버깅용 노드를 따로 마련하는 방법이 있을 수 있다.

Job scheduler 중에서도 GPU 클러스터에서 가장 많이 사용하는 것이 [**slurm**](https://slurm.schedmd.com/documentation.html)이다.
어쩄든, slurm을 처음부터 세팅하는 것은 생각보다 어렵다. 왜냐하면, slurm설정이 처음 보면 난해하기 때문이다. 그래서 이를 알려주고자 한다.

여기에 HPC 환경을 위한 Container인 Apptainer(former Singularity)도 같이 설정해서 컨테이너 환경에서 Reproducible한 연구가 될 수 있도록 가이드할 예정이다.

### Batch System

slurm을 셋업하기 앞서, batch system의 전체적인 구조 및 workflow를 소개하고자 한다.

{% img align="center" style='background-color: #fff' caption='<a href="https://hpc-wiki.info/hpc/File:Batch_System.png">Schematic of how users can access the batch system</a>' src='/assets/images/post/2024-05-10-Slurm-Setup-Guide/01-Schematic-Batch-System.png' %}

우선, 유저는 로그인 노드(login node) 혹은 메인 노드(main node)라는 서버에 접속해서 모든것을 수행한다. 유저는 계산 노드(computing node)에 접근할 수 없다.
그리고 job script 파일을 통해 job scheduler에 계산 혹은 실험(job)을 submit하고 job scheduler는 유저가 작성한 job script 파일을 보고 스케줄링 시스템에 따라 적절한 계산노드에 job을 할당한다.
만약에 남는 자원이 없다면 대기열(queue)에 등록하고 자리가 빌 때까지 기다리게 된다.

로그인 노드와 각 계산 노드는 당연히 동일한 유저가 있어야 하고, NAS 같이 별도의 파일서버가 있어서 파일 시스템도 공유해야 job 결과를 메인노드에서도 확인할 수 있게 된다. 이를 이해하기 위해서는 [공개 키 암호방식](https://ko.wikipedia.org/wiki/%EA%B3%B5%EA%B0%9C_%ED%82%A4_%EC%95%94%ED%98%B8_%EB%B0%A9%EC%8B%9D)에 대해서는 필수적으로 공부할 필요가 있다.

{% img align="center" style='background-color: #fff' caption='<a href="https://hpc-wiki.info/hpc/File:Scheduler.png">Schematic of how a scheduler may distribute jobs onto nodes</a>' src='/assets/images/post/2024-05-10-Slurm-Setup-Guide/02-Scheduler.png' %}

주의할 점은 Job scheduler는 단순히 비어있는 공간에 유저가 요청한 자원을 할당한다는 점이다. 만약에 Job schduler에는 1개의 GPU를 사용한다고 명시했는데, 코드 상에서 강제로 GPU를 2개 사용해버리면 다른 유저가 사용하고 있는 GPU를 같이 사용하게 돼서 문제가 생길 수 있다.

## Setup Cluster

데모를 위해 GCP(Google Cloud Platform)를 사용해서 가상의 HPC 클러스터를 설정해보겠다.
사람마다 클러스터 환경이 조금씩 다르기에 초기 셋업도 같이 공유하고자 하는 것이 목적이다.

### Create Project

1. GCP에서 slurm-demo라는 프로젝트를 생성
2. 빌링 설정

### Setup VPC (Virtual Private Cloud) Network

#### Assumption
* 클러스터 노드들이 같은 네트워크 안에 묶여있어야 한다.
* 일반적으로 계산 노드들은 외부망과 차단되어 있다. (메인 노드 제외)
* 로그인 노드 = 계산 노드일 때도 있다.

#### Method
1. GCP에서 VPC network 선택
2. Enable Compute Engine API
3. Create VPC network 선택
4. 다음과 같이 설정 (단순하게 하기 위해 최대한 자동 설정을 사용한다.) 나머지는 건드리지 않는다.
    1. Name : hpc-cluster-vpc
    2. Subnet creation mode : Automatic
    3. Firewall rules : hpc-cluster-vpc-allow-ssh
5. 만들고 나서 Firewall에 allow-internal 항목이 있는지 체크

### Setup Login Node

1. Name : hpc-node-login
2. Region과 Zone을 고른다.
    * Zone : us-west4
    * Region : us-west4-a
3. Machine Configuration
    * E2 선택 후 다음 프리셋 선택
    * Preset : e2-standard-4 (4 vCPU, 2 core, 16 GB memory)
    * VM provisioning model : 가격 절감을 위해 Spot 선택
4. Boot disk
    * OS : Ubuntu
    * Version : Ubuntu 24.04 LTS (built on 5/16)
    * Size : 120 GB
5. Advanced options
    1. Networking
        * Hostname : slurm-demo.hpc-node-login
    2. Network interfaces 위에서 만든 VPC를 붙인다.
        * Network : hpc-cluster-vpc
        * Subnetwork : hpc-cluster-vpc IPv4
    3. Network Service Tier : Standard

### Setup Compute Node Template

우선 Compute template을 만들어서 생성하는게 편하다.
Virtual machines -> Instance templates -> Create Instance Template를 클릭하여 다음과 같이 설정한다.

1. Name : hpc-node-compute-template
2. Region과 Zone을 고른다.
    * Zone : us-east5
    * Region : us-east5-a
3. Machine Configuration
    * GPU type : NVIDIA T4
    * Number of GPUs : 2
    * Machine type : n1-standard-1
    * VM provisioning model : 가격 절감을 위해 Spot 선택
4. Boot disk
    * OS : Ubuntu
    * Version : Ubuntu 24.04 LTS (built on 5/16)
    * Size : 80 GB
5. Advanced options
    1. Networking
        * Hostname : slurm-demo.hpc-node-compute
    2. Network interfaces 위에서 만든 VPC를 붙인다.
        * Network : hpc-cluster-vpc
        * Subnetwork : hpc-cluster-vpc IPv4
    3. Network Service Tier : Standard

그 다음 instance를 만들 때 New VM instance from template을 클릭한뒤 템플릿대로 생성한다.

### Conclusion

노드 구성은 다음과 같다.
* 1 Login node (main node)
* 1 CPU compute node (= login node)
    * Login node에서 job을 수행할 수 있도록 할 예정이다.
    * 이렇게 하는 이유는 GPU instance는 비싸기 때문에 CPU job을 우선적으로 세팅하고 테스트할 예정이기 때문이다.
    * 그 다음에 GPU compute node를 추가해서 확인할 예정
* 2 GPU compute node
    * 각 노드 당 2개의 T4를 가지고 있다고 가정한다.
    * 이 클러스터의 총 GPU는 NVIDIA T4 4대이다.

## Slurm Setup Guide (CPU)

자 이제 Login node instance를 실행하고 다음과 같이 slurm을 설치한다. (**Ubuntu 24.04 LTS 기준**)

### Install Slurm

1. System upgrade. 기본적으로 시스템을 최신상태로 유지한다. 만약 nvidia driver가 미리 깔려있었다면, nvidia driver가 업데이트될 수도 있는데, 이러면 driver mismatch 에러가 나면서 nvidia-smi부터 안되기 시작할 수 있다. 그러기에 다음 명령어 수행 후 재부팅을 한번 진행하면 좋다.
    ```shell
    sudo apt update && sudo apt upgrade -y
    ```

2. 다음 패키지들을 설치한다. 패키지 목록은 [pyenv wiki](https://github.com/pyenv/pyenv/wiki#suggested-build-environment)에서 가져온 Suggested build environment인데, 설치하다보면 어차피 많이 겹쳐서 같이 설치하면 좋다.
    ```shell
    sudo apt update && sudo apt install -y build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev curl git \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    ```

3. slurm을 설치한다.
    ```shell
    sudo apt install slurm-wlm slurm-wlm-doc
    ```

4. `mailutils`를 설치해서 slurm이 `/bin/mail`이 없다고 complain하는걸 막는다.
    [single_machine_slurm_on_ubuntu](https://gist.github.com/ckandoth/2acef6310041244a690e4c08d2610423)를 참고했다.

    ```shell
    sudo apt install -y mailutils
    ```
    어떤 메일 시스템을 사용할지 물어보는데, 외부와 메일을 주고 받지는 않을 것이기 때문에 local system용으로 설정한다.

5. `/etc/hosts` 맨 아래에 hostname을 **추가**한다.
    ```shell
    10.182.0.4 slurm-demo.hpc-node-login
    ```

    `ping`을 통해 체크해본다.
    ```shell
    ping slurm-demo.hpc-node-login
    ```

6. 나중을 위해 `spool` 디렉토리도 만들어준다. 왜인지는 몰라도, 자동으로 만들어지지 않아서 나중에 에러가 생긴다.
    ```shell
    sudo mkdir -p /var/spool/slurmctld
    sudo mkdir -p /var/spool/slurmd
    sudo chown slurm:slurm /var/spool/slurmctld
    sudo chown slurm:slurm /var/spool/slurmd
    sudo chmod 755 /var/spool/slurmctld/
    sudo chmod 755 /var/spool/slurmd/
    ```

7. pid 파일을 위한 디렉토리도 만들어준다.
    ```shell
    sudo mkdir -p /var/run/slurm
    sudo chown slurm:slurm /var/run/slurm
    sudo chmod 755 /var/run/slurm
    ```

7. 본격적인 Setup에 앞서 인증을 위해 MUNGE를 설치하고 slurm Accounting을 위해 MariaDB를 셋업한다.

### Install MUNGE

MUNGE (MUNGE Uid 'N' Gid Emporium)는 HPC환경을 위한 인증서비스이다.
인프라 관리 초보 시절에 가장 이해가 안되던 부분이 바로 다양한 노드에 있는 동일한 유저들을 어떻게 서로 인증 하는지 궁금했다.
리눅스의 유저와 그룹 그리고 UID, GID에 대한 이해가 있다면 모든 노드에 같은 UID, GID를 공유해야한다는 점을 알아두어야 한다.
MUNGE는 그 위에서 작동한다. MUNGE는 관리자 권한(privileged permission), 예약된 포트, 또는 플랫폼 특화 방법을 사용하지 않고 인증 정보를 생성하고 검증할 수 있다.

1. MUNGE를 설치한다.
    ```shell
    sudo apt install munge libmunge-dev libmunge2
    ```

2. MUNGE key를 생성한다. 해당 키는 다른 노드에 복사해야 인증이 이루어질 수 있기에 잘 보관해야한다.
    **그러기에 Login node에서만, 그것도 MUNGE 처음 설치할 때만 수행하는 작업이다.**
    ```shell
    sudo /usr/sbin/mungekey --create
    ```
    만약 이미 키가 존재한다면, 있다고 에러가 뜰 수 있다. 찝찝하면 다시 지우고 다시 만들어도 된다.
    ```
    sudo rm /etc/munge/munge.key
    sudo /usr/sbin/mungekey --create
    sudo ls /etc/munge -alh
    ```
    오래된 버전에서는 `/usr/sbin/mungekey`대신에 `/usr/sbin/create-munge-key -r`를 사용하는 경우도 있다.

3. 키의 소유주를 munge유저로 바꿔야 한다. 바꾸기 전에 우선 munge 유저가 존재하는지 체크해본다.
    ```shell
    cat /etc/passwd | grep munge
    cat /etc/group | grep munge
    ```
4. 체크한 다음, 파일 유저와 그룹을 munge로 바꿔준다.
    ```shell
    sudo chown munge:munge /etc/munge/munge.key
    ```

5. 파일 권한도 400으로 바꿔서 파일 소유주만 읽을 수 있도록 한다.
    ```shell
    sudo chmod 400 /etc/munge/munge.key
    ```

6. munge를 재시작한다.
    ```shell
    sudo systemctl restart munge
    ```

7. munge가 정상적으로 작동하는지 테스트한다.
    ```shell
    munge -n | unmunge
    ```
    다음과 같이 STATUS에 Success가 나오면 정상이다.
    ```shell
    $ munge -n | unmunge
    STATUS:          Success (0)
    ENCODE_HOST:     slurm-demo.hpc-node-login (10.182.0.4)
    ENCODE_TIME:     2024-05-19 07:44:43 +0000 (1716104683)
    DECODE_TIME:     2024-05-19 07:44:43 +0000 (1716104683)
    TTL:             300
    CIPHER:          aes128 (4)
    MAC:             sha256 (5)
    ZIP:             none (0)
    UID:             MYUSERNAME (1001)
    GID:             MYUSERNAME (1002)
    LENGTH:          0
    ```

### Install MariaDB

slurm에는 Accounting이라는 기능이 있다. Job scheduler의 회계같은 기능이라고 보면 되는데,
이 기능은 job이 사용한 리소스등을 기록하는 역할을 하고 자원 제한(reousrce limit)등에 이용할 수 있다.
여튼, accounting을 사용하기 위해서는 어딘가 기록을 해야하는데, 아무래도 파일보다는 DB에 기록하는게 좋다.
유저들이 자기의 job을 조회하는 등에서 파일은 불리한 점이 많고, 점점 지날수록 용량도 많이 차지하기 때문이다.
그러기 위해서 MariaDB를 설정하고자 한다.

1. MariaDB 설치
    ```shell
    sudo apt install -y mariadb-server mariadb-client libmariadb-dev-compat
    ```
2. 설치후 MariaDB initial setup을 한다.
    ```shell
    sudo mysql_secure_installation
    ```
    1. 처음 루트 패스워드를 입력한다. 처음이므로 엔터를 입력 "Enter current password for root (enter for none): "
    2. Root password를 변경할 예정이므로 unix_socket authenication을 사용하지 않는다. "Switch to unix_socket authentication [Y/n] n"
    3. 새로운 root password를 설정한다. "Change the root password? [Y/n] y"
    4. 익명 유저 로그인을 막았다. "Remove anonymous users? [Y/n] y"
    5. 보안을 위해 root login을 remote에서 하는걸 막는다. "Disallow root login remotely? [Y/n] n"
    6. test database를 제거한다. "Remove test database and access to it? [Y/n] y"
    7. 지금까지 설정한것을 반영하기 위해 privilege table를 reload한다. "Reload privilege tables now? [Y/n] y"

3. slurm accounting table을 만들기 위해 root로 로그인한다. (다음 명령어 입력후 위에서 설정한 패스워드를 입력한다.)
    ```shell
    sudo mysql -u root -p
    ```

4. 다음과 같은 MySQL shell이 보일 것이다.
    ```shell
    MariaDB [(none)]>
    ```

5. MySQL shell에서 accounting을 위한 DATABASE를 만든다.
    ```shell
    CREATE DATABASE slurm_acct_db;
    ```

6. Slurm의 DB 패스워드를 "SOME_SLURM_PASSWORD"라고 하고, 다음과 같이 slurm을 위한 DB user `slurm`를 생성한다.
    host는 `localhost`를 강제해서 로컬에서만 연결할 수 있게 하였다.
    물론 보안상 root 유저와 다른 패스워드를 사용해야한다.
    ```shell
    CREATE USER 'slurm'@'localhost' IDENTIFIED BY 'SOME_SLURM_PASSWORD';
    ```

7. DB user `slurm`에게 `slurm_acct_db`의 모든 권한을 부여한다.
    ```shell
    GRANT ALL PRIVILEGES ON slurm_acct_db.* TO `slurm`@`localhost`;
    ```

8. 위의 `GRANT ALL PRIVILEGES`를 바로 반영하기 위해 PRIVILEGES table을 reload한다.
    ```shell
    FLUSH PRIVILEGES;
    ```

9. MySQL 쉘을 나간다.
    ```shell
    EXIT;
    ```

10. slurm에서 DB를 연결하기 위해 `slurmdbd`패키지를 설치한다.
    ```shell
    sudo apt install -y slurmdbd
    ```

11. [Slurm Accounting Configuration Before Build](https://slurm.schedmd.com/accounting.html#slurm-accounting-configuration-before-build)를 참고하여 `/etc/mysql/my.cnf`파일의 다음 항목을 적절히 조정한다.
    공식 문서에서 예시로 든 값은 다음과 같다.
    ```shell
    [mysqld]
    innodb_buffer_pool_size=4096M
    innodb_log_file_size=64M
    innodb_lock_wait_timeout=900
    max_allowed_packet=16M
    ```

12. MariaDB와 slurmdbd를 재시작한다.
    ```shell
    sudo systemctl restart mysqld
    sudo systemctl restart slurmdbd
    ```

### Setup slurm

이제 본격적으로 slurm 환경설정을 해야한다.
여기부터는 각자 시스템마다 다른 환경을 지니고 있어 시스템 사양 특히 CPU와 메모리를 알아둘 필요가 있다.
현재는 login node와 compute node가 같은 노드이므로 서버를 바꾸지 않고 바로 진행해보도록 하겠다.

#### Find System Information

1. Memory 알아내기
    * slurm configuration의 RealMemory에 해당하는 값을 알 필요가 있다.
    * RealMemory는 Megabytes단위를 적어주면 되는데 다음과 같은 명령어를 입력하고 "Total"에 해당하는 값을 적어주면 된다.
        ```shell
        free -m
        ```
    * 예를 들어 본 데모에서는 16GB VM을 설정했고 다음과 같은 output이 나왔다. 이 때 RealMemory는 15990이 될 예정이다.
    ```shell
        $ free -m
                       total        used        free      shared  buff/cache   available
        Mem:           15990         694       14162           0        1430       15295
        Swap:              0           0           0
    ```
2. CPU 정보 알아내기
    * slurm configuration의 CPUs, Sockets, CoresPerSocket, ThreadsPerCore를 알아내야한다. 아마 가장 실수하기 좋을 부분일 것이다.
    * 각각은 다음과 같은 의미를 지닌다.
        * CPUs : 노드의 logical processor의 개수. 생략할 경우, Boards(메인보드수인데 보통은 1), Sockets, CoresPerSocket, ThreadsPerCore의 곱으로 결정된다.
        * Sockets : 노드의 physical processor의 개수.
        * CoresPerSocket : 소켓 하나의 Core 수
        * ThreadsPerCore : Physical core하나에 논리적인 Thread 수
    * 예를 들어 [AMD EPYC 9354](https://www.amd.com/ko/products/processors/server/epyc/4th-generation-9004-and-8004-series/amd-epyc-9354.html)를 사용한다고 하자.
        * 보통 2P를 쓴다. 즉 해당 CPU 2개를 한 보드에 꼽아서 쓴다. 따라서 Sockets는 2이다.
        * CoresPerSocket은 해당 스펙의 CPU 코어수 즉 32이다.
        * ThreadsPerCore은 Intel의 HyperThreading, AMD의 SMT를 생각하면 된다.
            해당 CPU Spec에 쓰레드 수는 64, CPU 코어 수는 32 이므로 ThreadsPerCore는 2이다.
        * CPUs는 Sockets * CoresPerSocket * ThreadsPerCore = 2 * 32 * 2 = 128이다.
    * 리눅스 커맨드 상에서는 다음 명령어를 사용한다.
        ```shell
        cat /proc/cpuinfo
        ```
    * 그러나 너무 길어서 보기가 어려운데, 그럴 때는 model name으로 위의 페이지처럼 스펙 찾아서 작성하는게 편하다.

#### Configuration File (slurm.conf)

자 이제 본격적으로 slurm.conf 파일을 작성할 필요가 있다.
항목이 많지만, 이걸 slurm 공식 사이트에서 자동으로 생성해준다. (웹에서는 최신버전만 지원)

[Slurm Version 23.11 Configuration Tool](https://slurm.schedmd.com/configurator.html)로 이동한다.

이제 다음과 같은 항목만 작성한다.

1. Cluster Name - `ClusterName`
    * 말 그대로 클러스터 이름이다. 알아서 작성한다. 데모에서는 `hpc-demo-cluster`라고 지정했다.
2. ControlMachine - `SlurmCtldHost`
    * Slurm Control Host, 즉 Login node의 hostname을 적어주면 된다.
    * login node의 쉘에서 리눅스 명령어 `hostname`을 실행해서 나온 값을 적어준다.
    * 본 데모에서는 `slurm-demo.hpc-node-login`이라고 하였다.
3. Compute Machines
    * 계산 노드에 대해서 작성해주는 곳이다.
    * `NodeName` : 계산 노드의 hostname을 적는 곳이다.
        * 기본값이 `linux[1-32]`처럼 같은 사양의 노드는 한번에 작성할 수 있다.
    * `PartionName`
        * 노드의 그룹을 만들 수 있고 이를 Partition이라고 한다.
        * 적당한 이름을 지정하면 된다. 본 데모에서는 `cpu`이라고 지정했다.
    * `CPUs`, `Sockets`, `CoresPerSocket`, `ThreadsPerCore`, `RealMemory`
        * 위 섹션에서 찾은 값으로 작성한다.
4. Event Logging
    * Compute Machines부터 Event Logging 전까지는 특별한 사항이 없으면 건드릴 일이 없다.
    * `/var/log/slurm`에 로그파일을 몰아넣는다. 참고로 이 로그 파일들은 [`logrotate`](https://www.digitalocean.com/community/tutorials/how-to-manage-logfiles-with-logrotate-on-ubuntu-22-04)를 사용하여 관리하면 과도하게 로그파일이 커지는 것을 막을 수 있다.
        * `SlurmctldLogFile` : `/var/log/slurm/slurmctld.log`
        * `SlurmdLogFile` : `/var/log/slurm/slurmd.log`
5. Job Completion Logging
    * `FileTxt`로 설정하고 다음 파일에 저장하도록 한다. 정말 큰 HPC 시스템이 아니면 이 로그는 `logrotate`를 안써도 상관없었다.
    * `JobCompLoc` : `/var/log/slrum/job_completions`
6. Job Accounting Gather
    * `Linux`를 선택 (`cgroup`을 성공하지 못했다.)
    * `JobAcctGatherFrequency`는 알아서 설정해준다. 디폴트 써도 상관없었다.
    * `JobAcctGatherFrequency` : 30
7. Job Accounting Storage
    * 다음과 같이 설정한다. `AccountingStoragePass`는 MUNGE가 알아서 해줄 것이기 때문에 빈칸으로 둔다.
        * `SlurmDBD` 선택
        * `AccountingStorageLoc` : `slurm_acct_db`
        * `AccountingStorageHost` : `localhost`
        * `AccountingStoragePort` : 6819
        * `AccountingStorageUser` : slurm

이렇게 하고 Submit을 누른다음 나온 설정파일을 복사하고, 일부분을 수정해야한다.

* `#UnkillableStepTimeout=60` -> `UnkillableStepTimeout=240`
    * 리소스를 많이 점유하는 job들은 반환하는데 시간이 오래걸리기 때문에 좀 더 오래 기다려준다.
    * [UnkillableStepTimeout](https://slurm.schedmd.com/slurm.conf.html#OPT_UnkillableStepTimeout)
* `#JobAcctGatherTypejobacct_gather/linux=` -> `JobAcctGatherType=jobacct_gather/linux`
    * 버그인듯
* `SlurmctldPidFile=/var/run/slurmctld.pid` -> `SlurmctldPidFile=/var/run/slurm/slurmctld.pid`
    * 디렉토리 권한 문제로 변경
* `SlurmdPidFile=/var/run/slurmd.pid` -> `SlurmdPidFile=/var/run/slurm/slurmd.pid`
    * 디렉토리 권한 문제로 변경
* `SelectTypeParameters=CR_CPU_Memory` 추가
    * `SelectType`은  Slurm이 작업(job)을 실행할 노드와 자원을 선택하는 방법을 정의한다.
    * `SelectTypeParameters=CR_CPU_Memory`은 Slurm이 CPU와 메모리를 함께 관리하여 작업이 CPU와 메모리를 동시에 요청할 수 있게 해준다.
    * [SelectTypeParameter](https://slurm.schedmd.com/slurm.conf.html#OPT_SelectTypeParameters)을 참고

그리고, 해당 내용을 `/etc/slurm/slurm.conf`로 저장한다. 파일이 없으면 만들어야 한다.

```shell
# slurm.conf file generated by configurator.html.
# Put this file on all nodes of your cluster.
# See the slurm.conf man page for more information.
#
ClusterName=hpc-demo-cluster
SlurmctldHost=slurm-demo.hpc-node-login
#SlurmctldHost=
#
#DisableRootJobs=NO
#EnforcePartLimits=NO
#Epilog=
#EpilogSlurmctld=
#FirstJobId=1
#MaxJobId=67043328
#GresTypes=
#GroupUpdateForce=0
#GroupUpdateTime=600
#JobFileAppend=0
#JobRequeue=1
#JobSubmitPlugins=lua
#KillOnBadExit=0
#LaunchType=launch/slurm
#Licenses=foo*4,bar
#MailProg=/bin/mail
#MaxJobCount=10000
#MaxStepCount=40000
#MaxTasksPerNode=512
#MpiDefault=
#MpiParams=ports=#-#
#PluginDir=
#PlugStackConfig=
#PrivateData=jobs
ProctrackType=proctrack/cgroup
#Prolog=
#PrologFlags=
#PrologSlurmctld=
#PropagatePrioProcess=0
#PropagateResourceLimits=
#PropagateResourceLimitsExcept=
#RebootProgram=
ReturnToService=1
SlurmctldPidFile=/var/run/slurm/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurm/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/spool/slurmd
SlurmUser=slurm
#SlurmdUser=root
#SrunEpilog=
#SrunProlog=
StateSaveLocation=/var/spool/slurmctld
#SwitchType=
#TaskEpilog=
TaskPlugin=task/affinity,task/cgroup
#TaskProlog=
#TopologyPlugin=topology/tree
#TmpFS=/tmp
#TrackWCKey=no
#TreeWidth=
#UnkillableStepProgram=
#UsePAM=0
#
#
# TIMERS
#BatchStartTimeout=10
#CompleteWait=0
#EpilogMsgTime=2000
#GetEnvTimeout=2
#HealthCheckInterval=0
#HealthCheckProgram=
InactiveLimit=0
KillWait=30
#MessageTimeout=10
#ResvOverRun=0
MinJobAge=300
#OverTimeLimit=0
SlurmctldTimeout=120
SlurmdTimeout=300
UnkillableStepTimeout=240
#VSizeFactor=0
Waittime=0
#
#
# SCHEDULING
#DefMemPerCPU=0
#MaxMemPerCPU=0
#SchedulerTimeSlice=30
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU_Memory
#
#
# JOB PRIORITY
#PriorityFlags=
#PriorityType=priority/multifactor
#PriorityDecayHalfLife=
#PriorityCalcPeriod=
#PriorityFavorSmall=
#PriorityMaxAge=
#PriorityUsageResetPeriod=
#PriorityWeightAge=
#PriorityWeightFairshare=
#PriorityWeightJobSize=
#PriorityWeightPartition=
#PriorityWeightQOS=
#
#
# LOGGING AND ACCOUNTING
#AccountingStorageEnforce=0
AccountingStorageHost=localhost
AccountingStoragePort=6819
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageUser=slurm
# AccountingStoragePass=
# AccountingStoreFlags=
#JobCompHost=
JobCompLoc=/var/log/slurm/job_completions
#JobCompParams=
#JobCompPass=
#JobCompPort=
JobCompType=jobcomp/filetxt
#JobCompUser=
#JobContainerType=
JobAcctGatherFrequency=30
JobAcctGatherType=jobacct_gather/linux
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm/slurmd.log
#SlurmSchedLogFile=
#SlurmSchedLogLevel=
#DebugFlags=
#
#
# POWER SAVE SUPPORT FOR IDLE NODES (optional)
#SuspendProgram=
#ResumeProgram=
#SuspendTimeout=
#ResumeTimeout=
#ResumeRate=
#SuspendExcNodes=
#SuspendExcParts=
#SuspendRate=
#SuspendTime=
#
#
# COMPUTE NODES
NodeName=slurm-demo.hpc-node-login CPUs=4 RealMemory=15990 Sockets=1 CoresPerSocket=2 ThreadsPerCore=2 State=UNKNOWN
PartitionName=cpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

또한 `slurm.conf`의 소유주는 `slurm:slurm`으로 한다.

```shell
sudo chown slurm:slurm /etc/slurm/slurm.conf
```

또한 State를 저장하기 위해 empty file을 만들어준다.

```shell
sudo touch /var/spool/slurmctld/trigger_state
sudo chown slurm:slurm /var/spool/slurmctld/trigger_state
sudo chmod 644 /var/spool/slurmctld/trigger_state
```

#### Configuration File (slurmdbd.conf)

Job accounting을 위해 `slurmdbd.conf`도 다음과 같이 `/etc/slurm/slurmdbd.conf`에 만들어준다.

```shell
#
# Example slurmdbd.conf file.
#
# See the slurmdbd.conf man page for more information.
#
# Archive info
#ArchiveJobs=yes
#ArchiveDir="/tmp"
#ArchiveSteps=yes
#ArchiveScript=
#JobPurge=12
#StepPurge=1
#
# Authentication info
AuthType=auth/munge
#AuthInfo=/var/run/munge/munge.socket.2
#
# slurmDBD info
DbdAddr=localhost
DbdHost=localhost
#DbdPort=7031
SlurmUser=slurm
#MessageTimeout=300
DebugLevel=verbose
#DefaultQOS=normal,standby
LogFile=/var/log/slurm/slurmdbd.log
PidFile=/var/run/slurm/slurmdbd.pid
#PluginDir=/usr/lib/slurm
#PrivateData=accounts,users,usage,jobs
#TrackWCKey=yes
#
# Database info
StorageType=accounting_storage/mysql
StorageHost=localhost
StoragePort=3306
StoragePass=SOME_SLURM_PASSWORD
StorageUser=slurm
StorageLoc=slurm_acct_db
```

그리고 `slurmdbd.conf`의 권한은 600으로 바꿔준다.

```shell
sudo chown slurm:slurm /etc/slurm/slurmdbd.conf
sudo chmod 600 /etc/slurm/slurmdbd.conf
```

### Run slurm

1. 본격적으로 slurm을 가동해볼 차례이다.
    ```shell
    sudo systemctl restart slurmdbd
    sudo systemctl restart slurmctld
    sudo systemctl restart slurmd
    ```

2. 안되면 `sudo systemctl status 데몬이름`을 통해서 오류와 로그파일을 보고 설정파일을 고치면 된다.
3. 다음 명령어를 통해 node의 상태를 확인한다.
    ```shell
    sinfo
    ```

    다음과 같이 IDLE상태면 정상이다.
    ```shell
    $ sinfo
    PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    cpu*         up   infinite      1   idle slurm-demo.hpc-node-login
    ```
4. 만약 STATE가 idle이 아니면 노드 설정이 잘못되었다고 볼 수 있다. `slurm.conf` 설정을 체크해본다.

#### Setup Accounting

slurm의 Accounting을 사용하기 위해서는 Accounting에 클러스터와 유저를 등록해야한다.
[이 링크](https://wiki.fysik.dtu.dk/Niflheim_system/Slurm_accounting/#create-accounts)의 매뉴얼이 Accounting을 이해하는데 많은 도움이 될 것이다.

1. 클러스터를 등록한다. 등록이 이미 되어있으면, 이미 등록되었다고 나올 것이다.
    ```shell
    sudo sacctmgr add cluster hpc-demo-cluster
    ```

2. 다음 명령어를 통해 클러스터를 확인해본다.
    ```shell
    sudo sacctmgr show clusters
    ```

3. Account를 등록한다. 일종의 유저 그룹이라고 할 수 있다. [Account](https://slurm.schedmd.com/accounting.html#account-options)
    ```shell
    sudo sacctmgr add account acc_group Description="Some Departments" Organization=acc_group
    ```
    Account는 계층구조로 이루어질 수도 있다. `acc_group` 밑에 `acc_sub_group`이 존재할 수도 있다.

    ```shell
    sudo sacctmgr add account acc_sub_group Description="Some Sub Departments" Organization=acc_sub_group parent=top_group
    ```

4. 다음 명령어를 통해 Account를 확인할 수 있다.
    ```shell
    sudo sacctmgr show account
    ```

5. 이제 Accoutning User를 등록한다. `DefaultAccount`는 무조건 지정해야한다. [Account User](https://slurm.schedmd.com/accounting.html#user-options) 이 때, xxx는 시스템 유저 아이디랑 매칭시켜야 정확한 관리가 가능하다.
    ```shell
    sudo sacctmgr create user name=xxx DefaultAccount=yyy
    ```

    User는 여러개의 Account에 등록할 수 있다. 이때는 `sacctmgr add`를 통해 유저를 다른 account에 지정한다.
    ```shell
    sudo sacctmgr add user xxx Account=yyy
    ```

6. 다음 명령어들을 통해 Accounting User 현황을 확인할 수 있다.
    ```shell
    sudo sacctmgr show user
    sudo sacctmgr show user -s
    sudo sacctmgr show account -s xxx
    ```

### Use slurm

#### Submit Job (Test)

기본적으로는 srun을 통해 간단하게 slurm을 테스트해볼 수 있다.
다음은 `cpu` 파티션에 `echo "Running in cpu partition"`을 실행시켜서 slurm을 테스트 하는 경우이다.

```shell
srun -p cpu echo "Running in cpu partition"
```
실행시킬 경우 다음과 같이 출력된다.

```shell
$ srun -p cpu echo "Running in cpu partition"
Running in cpu partition
$
```

잡 로그를 확인해보기 위해 accounting 기능을 확인해본다.

```shell
sacct
```

특정 시간 범위의 조회의 경우 다음과 같은 명령어로 체크할 수 있다.

```shell
sacct --starttime=2023-05-01 --endtime=2023-05-02
```

특정 사용자의 경우는 `--user`를 사용한다.

```shell
sacct --user=username
```

상태에 따라서는 다음과 같은 명령어를 사용한다. (상태 예시는 `FAILED`, `CANCELLED`, `TIMEOUT`, `COMPLETED` 등이 있다)
```shell
sacct --state=COMPLETED
```
#### Submit Job (Job Script)

`srun` 옵션을 매번 작성하기는 쉽지 않다. 따라서 보통 잡 스크립트(job script)파일을 작성하고 `sbatch` 명령어를 submit한다.

위에서 실행한 잡은 다음 스크립트를 `sbatch job_script.sh`로 실행한것과 같다.

```shell
#!/bin/bash
#SBATCH --job-name=hello_world      # 작업 이름
#SBATCH --output=hello_world.out    # 표준 출력 파일 이름
#SBATCH --error=hello_world.err     # 표준 에러 파일 이름
#SBATCH --time=00:05:00             # 작업 시간 제한 (HH:MM:SS)
#SBATCH --partition=cpu             # 파티션 이름
#SBATCH --ntasks=1                  # 총 실행할 작업 수
#SBATCH --cpus-per-task=1           # 작업당 CPU 코어 수
#SBATCH --mem=1G                    # 작업당 메모리 요구량

# 실행할 명령어
echo "Running in cpu partition"
```

아까와 다르게 `sbatch` 명령어로 실행시키면 `hello_world.out`과 `hello_world.err`파일이 생성되고 에러는 없으므로 `hello_world.err`파일은 빈 파일, 그리고 `hello_world.out`파일에는 "Running in cpu partition"가 출력되어서 나온다.

돌아가고 있는 job은 `squeue` 명령어를 통해 확인할 수 있다.
```shell
$ squeue
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
    4       cpu hello_wo some_use  R       0:01      1 slurm-demo.hpc-node-login
```

돌아가고 있는 job을 취소하는것은 `squeue`를 통해 JOBID를 확인하고 `scancel` 명령어를 사용한다.
```shell
$ scancel 4
```

## Slurm Setup Guide (GPU)

### Add GPU Node to slurm

지금까지는 CPU만 있는 노드에 slurm을 세팅해봤다.
하지만 이제 GPU 노드를 추가해보고자 한다.

계산 노드 2개를 다음과 같이 각각 만들어준다.

### Setup Compute Node in GCP

xx를 01과 02로 해서 2개를 만들었다.

1. Name : hpc-node-compute-xx
2. Region과 Zone을 고른다.
    * Zone : us-west4
    * Region : us-west4-a
3. Machine Configuration
    * N1 선택 후 다음 프리셋 선택
    * Preset : n1-standard-1
    * VM provisioning model : GCP에서 Spot으로 T4 GPU를 절대 할당받을수 없었기에 눈물을 머금고 Standard
4. Boot disk
    * OS : Ubuntu
    * Version : Ubuntu 24.04 LTS (built on 5/16)
    * Size : 80 GB
5. Advanced options
    1. Networking
        * Hostname : slurm-demo.hpc-node-computexx
    2. Network interfaces 위에서 만든 VPC를 붙인다.
        * Network : hpc-cluster-vpc
        * Subnetwork : hpc-cluster-vpc IPv4
    3. Network Service Tier : Standard


즉 지금 다음과 같이 3개의 노드가 있다.

1. hpc-node-login (10.182.0.4)
    * login 노드 및 cpu용 compute 겸용
2. hpc-node-compute01  (10.182.0.2)
    * gpu용 compute
3. hpc-node-compute02  (10.182.0.3)
    * gpu용 compute

세 노드에 user는 모두 같은 구성이다. (UID=1001인 MYUSERNAME이 있다고 가정)

진행하기 전에 각 노드 `hosts` 파일에 hostname을 추가해준다.

```
10.182.0.4 slurm-demo.hpc-node-login
10.182.0.2 slurm-demo.hpc-node-compute01
10.182.0.3 slurm-demo.hpc-node-compute02
```

### Install CUDA

각 compute노드마다 다음과 같이 패키지를 업데이트하고 CUDA를 설치한다.

1. 디바이스 확인 및 `ubuntu-drivers-common` 설치
    ```shell
    sudo apt update
    sudo apt upgrade -y
    sudo lspci | grep -i nvidia
    sudo apt install ubuntu-drivers-common
    ```

2. nvidia driver 확인
    ```shell
    $ sudo ubuntu-drivers devices
    udevadm hwdb is deprecated. Use systemd-hwdb instead.
    udevadm hwdb is deprecated. Use systemd-hwdb instead.
    udevadm hwdb is deprecated. Use systemd-hwdb instead.
    udevadm hwdb is deprecated. Use systemd-hwdb instead.
    ERROR:root:aplay command not found
    == /sys/devices/pci0000:00/0000:00:05.0 ==
    modalias : pci:v000010DEd00001EB8sv000010DEsd000012A2bc03sc02i00
    vendor   : NVIDIA Corporation
    model    : TU104GL [Tesla T4]
    driver   : nvidia-driver-535-server - distro non-free
    driver   : nvidia-driver-535 - distro non-free recommended
    driver   : nvidia-driver-470-server - distro non-free
    driver   : nvidia-driver-470 - distro non-free
    driver   : xserver-xorg-video-nouveau - distro free builtin
    ```

3. 현재 드라이버 기준으로 가장 최신인 `nvidia-driver-535-server`를 설치한다.
    ```shell
    sudo apt install nvidia-driver-535-server
    ```

4. 재부팅
    ```shell
    sudo reboot
    ```

5. CUDA 설치. 이제는 NVIDIA Repo를 추가 안해도 바로 CUDA 설치가 되는 듯하다.
    ```shell
    sudo apt install nvidia-cuda-toolkit
    ```

6. nvidia-smi로 GPU driver가 제대로 로드되었는지 확인해본다.
    ```shell
    nvidia-smi
    ```

### SSH & MUNGE Key configuration

1. 로그인 노드로 다시 돌아가서 SSH Key를 생성한다.
    ```shell
    ssh-keygen -t ed25519
    ```

2. 키 파일명들을 `compute-node`로 바꿔준다.
    ```shell
    mv ~/.ssh/id_ed25519 ~/.ssh/compute-node
    mv ~/.ssh/id_ed25519.pub ~/.ssh/compute-node.pub
    ```

3. ssh agent를 편하게 관리하는 [keychain](https://www.funtoo.org/Funtoo:Keychain) 설치
    ```shell
    sudo apt install keychain
    ```

4. 다음을 `~/.bashrc`에 추가해서 키를 등록한다
    ```shell
    eval `keychain --eval --agents ssh compute-node`
    ```

5. `source`를 통해 `.bashrc`를 다시 로드한다.
    ```shell
    $ source ~/.bashrc

    * keychain 2.8.5 ~ http://www.funtoo.org
    * Starting ssh-agent...
    * Adding 1 ssh key(s): compute-node
    * ssh-add: Identities added: compute-node
    ```

6. 혹시 모르니 `~/.ssh/config` 도 다음과 같이 만들어준다.
    ```
    Host slurm-demo.hpc-node-compute01
        HostName slurm-demo.hpc-node-compute01
        User MYUSERNAME
        IdentityFile ~/.ssh/compute-node

    Host slurm-demo.hpc-node-compute02
        HostName slurm-demo.hpc-node-compute02
        User MYUSERNAME
        IdentityFile ~/.ssh/compute-node
    ```

7. public key를 클립보드에 복사한다.
    ```shell
    cat ~/.ssh/compute-node.pub
    ```

8. 계산 노드에 들어가서 `~/.ssh` 디렉토리를 만들고 `authorized_keys`에 public key를 넣어준다. `~/.ssh` 와 `~/.ssh/authorized_keys`는 보안을 위해 소유자만 접근할 수 있는 권한을 설정해준다.
    ```shell
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    echo PUBIC_KEY_복사한거 >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    ```

9. 로그인 노드로 다시 돌아가 접속이 되는지 테스트해본다.
    ```
    ssh MYUSERNAME@slurm-demo.hpc-node-compute-01
    ssh MYUSERNAME@slurm-demo.hpc-node-compute-02
    ```

### Install slurm

각 계산노드에도 다음과 같이 slurm을 설치한다.

1. slurm 설치 (compute node이므로 `slurmd`만 설치)
    ```shell
    sudo apt install slurmd
    ```

2. MUNGE key를 ssh 디렉토리에 복사해놓는다. (아무 디렉토리에 복사하기엔 양심에 찔림) 그리고 소유주도 임시적으로 바꿔준다.
    ```shell
    sudo cp /etc/munge/munge.key ~/.ssh/munge.key
    sudo chown MYUSERNAME:MYUSERNAME ~/.ssh/munge.key
    ```

3. 로그인 노드에서 기존에 MUNGE Key만든것을 계산노드로 전송한다.
    ```shell
    scp ~/.ssh/munge.key MYUSERNAME@slurm-demo.hpc-node-compute01:/home/MYUSERNAME/.ssh/munge.key
    scp ~/.ssh/munge.key MYUSERNAME@slurm-demo.hpc-node-compute02:/home/MYUSERNAME/.ssh/munge.key
    ```

4. 각 계산 노드로 들어가 MUNGE Key를 확인하고 소유주를 `munge`로 바꾼뒤 원래 있어야할 경로(`/etc/munge/`)로 복사한다.
    ```shell
    sudo chown munge:munge ~/.ssh/munge.key
    sudo mv ~/.ssh/munge.key /etc/munge/munge.key
    ```

5. 계산 노드에서 MUNGE를 재시작한다.
    ```shell
    sudo systemctl restart munge
    ```

### slurm.conf Modification

기존의 `slurm.conf`의 node와 partition부분에 새로운 compute노드를 추가하고, 이 설정을 동일하게 모든 노드에 업데이트해야한다.

1. 로그인 노드로 들어가서 `slurm.conf`의 맨 마지막 부분 다음 부분에 주목한다.
    ```shell
    NodeName=slurm-demo.hpc-node-login CPUs=4 RealMemory=15990 Sockets=1 CoresPerSocket=2 ThreadsPerCore=2 State=UNKNOWN
    PartitionName=cpu Nodes=ALL Default=YES MaxTime=INFINITE State=UP
    ```

2. 기존 cpu partition의 Nodes부분만 수정하고(계속 쓸테니), 새롭게 GPU node와 partition을 추가한다. 여기서 조심해야할 것은 NodeName에 `-`이 너무 많으면 slurm이 node를 제대로 인식 못할 수 있다.
    ```shell
    NodeName=slurm-demo.hpc-node-login CPUs=4 RealMemory=15990 Sockets=1 CoresPerSocket=2 ThreadsPerCore=2 State=UNKNOWN
    PartitionName=cpu Nodes=slurm-demo.hpc-node-login Default=YES MaxTime=INFINITE State=UP

    # Define the types of GRES available
    GresTypes=gpu

    NodeName=slurm-demo.hpc-node-compute01 Gres=gpu:2 CPUs=1 RealMemory=3661 Sockets=1 CoresPerSocket=1 ThreadsPerCore=1 State=UNKNOWN
    NodeName=slurm-demo.hpc-node-compute02 Gres=gpu:2 CPUs=1 RealMemory=3661 Sockets=1 CoresPerSocket=1 ThreadsPerCore=1 State=UNKNOWN

    PartitionName=gpu Nodes=slurm-demo.hpc-node-compute[01-02] Default=YES MaxTime=INFINITE State=UP
    ```

3. 로그인 노드에서 scp를 사용해서 `slurm.conf`를 전파한다.
    ```shell
    sudo cp /etc/slurm/slurm.conf ~/slurm.conf
    sudo chown MYUSERNAME:MYUSERNAME ~/slurm.conf
    scp ~/slurm.conf MYUSERNAME@slurm-demo.hpc-node-compute01:/home/MYUSERNAME/slurm.conf
    scp ~/slurm.conf MYUSERNAME@slurm-demo.hpc-node-compute02:/home/MYUSERNAME/slurm.conf
    ```

4. 각 계산노드에 들어가서 `slurm.conf`를 로그인 노드에서 복사한 `slurm.conf`로 교체해준다. 파일이 이미 있는 경우 백업을 해준다.
    ```
    sudo chown root:root ~/slurm.conf
    sudo mv /etc/slurm/slurm.conf /etc/slurm/slurm.conf.backup
    sudo mv ~/slurm.conf /etc/slurm/slurm.conf
    ```

5. 그리고 GPU 2개를 가정했을 때, `/etc/gres/gres.conf`를 만들어준다. [공식 문서](https://manpages.ubuntu.com/manpages/focal/en/man5/gres.conf.5.html)를 참고하면 좋다.
    ```shell
    # Node-specific GRES configuration for slurm-demo.hpc-node-compute-01
    Name=gpu Type=tesla File=/dev/nvidia0
    Name=gpu Type=tesla File=/dev/nvidia1
    ```

    ```shell
    # Node-specific GRES configuration for slurm-demo.hpc-node-compute-02
    Name=gpu Type=tesla File=/dev/nvidia0
    Name=gpu Type=tesla File=/dev/nvidia1
    ```

6. 로그인 노드의 slurm데몬들을 재시작한다.
    ```shell
    sudo systemctl restart slurmctld
    sudo systemctl restart slurmd
    ```

7. 계산 노드의 slurm데몬들을 재시작한다.
    ```shell
    sudo systemctl restart slurmd
    ```shell

8. `slurmd`의 동작상태를 확인해본다.
    ```shell
    sudo systemctl status slurmd
    ```

8. `gres.conf`은 계산 노드 로컬에 저장되므로 따로따로 관리되지만, `slurm.conf`은 모두 통일되어야 한다. 따라서 만약에 Resource limit등으로 `slurm.conf`를 수정했다면 이를 모든 계산노드에 전파할 필요가 있다.

9. 만약 안된다면 방화벽 문제일 수도 있다. 클라우드라면 VPC에 Firewall Rule이 등록되었는지 확인하자(Default port: 6817, 6818)
그리고 `ufw`가 활성화 되어있다면 `ufw`로 모든 노드의 방화벽 룰을 등록해주자.
    ```
    sudo ufw allow 6817/tcp
    sudo ufw allow 6818/tcp
    ```

10. 로그인 노드로 돌아와서 노드 상태를 확인한다. 다음과 같이 `unk*`이면 UNKNOWN상태라는 뜻이다.
    ```shell
    $ sinfo
    PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
    cpu          up   infinite      1   idle slurm-demo.hpc-node-login
    gpu*         up   infinite      2   unk* slurm-demo.hpc-node-compute[01-02]
    ```

11. idle상태가 되어야 해당 노드를 사용할 수 있다. idle로 강제로 바꿔주자.
    ```shell
    sudo scontrol update NodeName=slurm-demo.hpc-node-compute01 State=RESUME
    sudo scontrol update NodeName=slurm-demo.hpc-node-compute02 State=RESUME
    ```

12. `srun`으로 slurm을 테스트해본다.
    ```shell
    $ srun --nodes=1 --ntasks=1 --partition=gpu nvidia-smi
    Wed May 22 18:20:31 2024
    +---------------------------------------------------------------------------------------+
    | NVIDIA-SMI 535.161.08             Driver Version: 535.161.08   CUDA Version: 12.2     |
    |-----------------------------------------+----------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
    |                                         |                      |               MIG M. |
    |=========================================+======================+======================|
    |   0  Tesla T4                       Off | 00000000:00:04.0 Off |                    0 |
    | N/A   77C    P0              30W /  70W |      2MiB / 15360MiB |      0%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+
    |   1  Tesla T4                       Off | 00000000:00:05.0 Off |                    0 |
    | N/A   77C    P0              33W /  70W |      2MiB / 15360MiB |      8%      Default |
    |                                         |                      |                  N/A |
    +-----------------------------------------+----------------------+----------------------+

    +---------------------------------------------------------------------------------------+
    | Processes:                                                                            |
    |  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
    |        ID   ID                                                             Usage      |
    |=======================================================================================|
    |  No running processes found                                                           |
    +---------------------------------------------------------------------------------------+
    ```
13. 노드 상태도 확인해본다.
    ```shell
    $ scontrol show nodes
    NodeName=slurm-demo.hpc-node-compute01 Arch=x86_64 CoresPerSocket=1
    CPUAlloc=0 CPUEfctv=1 CPUTot=1 CPULoad=0.00
    AvailableFeatures=(null)
    ActiveFeatures=(null)
    Gres=gpu:2
    NodeAddr=slurm-demo.hpc-node-compute01 NodeHostName=slurm-demo.hpc-node-compute01 Version=23.11.4
    OS=Linux 6.8.0-1007-gcp #7-Ubuntu SMP Sat Apr 20 00:58:31 UTC 2024
    RealMemory=3661 AllocMem=0 FreeMem=3055 Sockets=1 Boards=1
    State=IDLE ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
    Partitions=gpu
    BootTime=2024-05-22T17:56:01 SlurmdStartTime=2024-05-22T18:18:36
    LastBusyTime=2024-05-22T18:20:32 ResumeAfterTime=None
    CfgTRES=cpu=1,mem=3661M,billing=1
    AllocTRES=
    CapWatts=n/a
    CurrentWatts=0 AveWatts=0
    ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a

    NodeName=slurm-demo.hpc-node-compute02 Arch=x86_64 CoresPerSocket=1
    CPUAlloc=0 CPUEfctv=1 CPUTot=1 CPULoad=0.00
    AvailableFeatures=(null)
    ActiveFeatures=(null)
    Gres=gpu:2
    NodeAddr=slurm-demo.hpc-node-compute02 NodeHostName=slurm-demo.hpc-node-compute02 Version=23.11.4
    OS=Linux 6.8.0-1007-gcp #7-Ubuntu SMP Sat Apr 20 00:58:31 UTC 2024
    RealMemory=3661 AllocMem=0 FreeMem=3105 Sockets=1 Boards=1
    State=IDLE ThreadsPerCore=1 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
    Partitions=gpu
    BootTime=2024-05-22T17:00:28 SlurmdStartTime=2024-05-22T18:18:29
    LastBusyTime=2024-05-22T18:17:03 ResumeAfterTime=None
    CfgTRES=cpu=1,mem=3661M,billing=1
    AllocTRES=
    CapWatts=n/a
    CurrentWatts=0 AveWatts=0
    ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a

    NodeName=slurm-demo.hpc-node-login Arch=x86_64 CoresPerSocket=2
    CPUAlloc=0 CPUEfctv=4 CPUTot=4 CPULoad=0.00
    AvailableFeatures=(null)
    ActiveFeatures=(null)
    Gres=(null)
    NodeAddr=slurm-demo.hpc-node-login NodeHostName=slurm-demo.hpc-node-login Version=23.11.4
    OS=Linux 6.8.0-1007-gcp #7-Ubuntu SMP Sat Apr 20 00:58:31 UTC 2024
    RealMemory=15990 AllocMem=0 FreeMem=14952 Sockets=1 Boards=1
    State=IDLE ThreadsPerCore=2 TmpDisk=0 Weight=1 Owner=N/A MCS_label=N/A
    Partitions=cpu
    BootTime=2024-05-22T14:43:47 SlurmdStartTime=2024-05-22T17:04:34
    LastBusyTime=2024-05-22T17:17:21 ResumeAfterTime=None
    CfgTRES=cpu=4,mem=15990M,billing=4
    AllocTRES=
    CapWatts=n/a
    CurrentWatts=0 AveWatts=0
    ExtSensorsJoules=n/a ExtSensorsWatts=0 ExtSensorsTemp=n/a
    ```

## Troubleshooting

Slurm을 쓰면서 가장 많이 겪는 문제 중 하나가 갑자기 노드가 drain상태에 빠지는 것이다.
만약, 특정 노드(예를 들어 `slurm-demo.hpc-node-compute01`)이 drain상태에 빠졌다면 수동으로 다음과 같이 복구할 수 있다.
```
scontrol: update NodeName=slurm-demo.hpc-node-compute01 State=DOWN Reason="undraining"
scontrol: update NodeName=slurm-demo.hpc-node-compute01 State=RESUME
```

이 문제의 원인을 한동안 몰랐는데, 최근에 이유를 추측할 수 있었다. Slurm job이 종료가 될 때, 돌고 있는 프로세스에 [`SIGTERM`](https://en.wikipedia.org/wiki/Signal_(IPC)#SIGTERM) signal을 보내게 되는데, `SIGTERM`을 보낸 후 어느정도 지나면 `SIGKILL`을 보낸다. 근데 만약 특정 시간이 지나도 Job 종료가 안되면 drain상태에 빠지는 것으로 추측한다.

특히, 서버의 자원이 대용량이 되어가면서 메모리 등을 반환하는데 시간이 예전보다 더 걸리는 경우가 많아서 `slurm.conf`의 timeout 시간들을 기본값보다 조금씩 늘려보는것도 나쁘지 않을 것 같다.

```
To cancel a job, invoke scancel without --signal option. This will send first a SIGCONT to all steps to eventually wake them up followed by a SIGTERM, then wait the KillWait duration defined in the slurm.conf file and finally if they have not terminated send a SIGKILL. This gives time for the running job/step(s) to clean up.
```

하지만, 자동으로 완전히 해결할 방법은 딱히 없어보인다.
timeout시간도 어느정도가 적정선인지는 시스템마다 경험적으로 알아내는 수밖에 없다.
모니터링을 열심히 하거나 cron으로 drain된 노드가 있으면 자동으로 undrain해주는 스크립트를 돌리거나 하는 방법밖에는 없어보인다.

## Conclusion

이렇게 slurm이 잘 되는 것을 확인했다. 많은 도움이 되었기를 바란다.
자세한건 이제 [공식 문서](https://slurm.schedmd.com/documentation.html)를 살펴보면서 바꿔보면 된다.

계산노드에 똑같은 작업을 반복해서 하기 귀찮다면 Ansible같은 여러 자동화툴이 존재하니까 써보면 나쁘지 않을 것같다.

그리고 다음 포스트에서는 Docker 대신 사용할 수 있는 Apptainer를 설치하고 이를 slurm에서 어떻게 돌릴 수 있는지 알아보겠다.
