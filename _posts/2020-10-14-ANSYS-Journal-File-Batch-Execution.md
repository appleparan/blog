---
layout: post
title: ANSYS Fluent Batch mode로 실행하기
author: jongsukim
date: 2020-10-14 12:00:00 +0900
categories: [Programming]
tags:
  - ANSYS
  - Batch mode
  - Journal Files
  - HPC
math: false
---

내가 담당하는 일은 아니지만, 연구실에서 2년전부터 ANSYS Fluent를 사용하고 있다. 그동안의 사용방식은 GUI로 바로 실행하는 형태였는데, 효율적 HPC 자원 관리를 위해 잡스케줄러를 사용해서 batch mode로 전환하고자 한다.

ANSYS는 프로그램의 사이즈 치고는 공개된 문서가 찾기 힘들고 리셀러 홈페이지에서 문서보기도 좀 복잡하기도 해서 batch mode로 어떻게 실행하는지 알기가 힘들었다. 게다가 난 내 일도 아니라서 ANSYS를 잘 쓸 줄 몰라서 더더욱 알기 힘들었다. 우리 연구실은 SGE를 주로 쓰는데 ANSYS에서 GUI형태로 SGE Job submission을 지원하기는 하지만 queue나 parallel environment설정을 어떻게 하는지 몰라서 포기했기에 어쩔 수 없이 아래와 같은 텍스트 방식을 고집할 수 밖에 없었다.

참고로 이번에도 역시 나의 구세주 [KISTI 매뉴얼](https://www.ksc.re.kr/gsjw/jcs/sft#docout-7)과 [Rescale 문서](https://docs.rescale.com/articles/ansys-start-here/)가 큰 도움이 되었다.

ANSYS를 batch 모드로 실행하는 과정은 간략하게 다음과 같다.

# ANSYS batch mode 방식 (간략하게)
1. ANSYS GUI Command를 텍스트로 실행할 수 있는 Journal file 생성
2. fluent 실행시 -i 옵션을 통해 journal file을 input으로 넣어줌

하지만 journal file의 문법이 참 찾기가 어려운데, 위에서도 설명했듯이 ANSYS는 GUI프로그램이지만 이를 Text로 실행할 수 있게 해주는 TUI(Text User Interface)가 존재하는데 이를 기록한 것이 journal file이다. TUI는 자체 커맨드도 있지만, 기본적으로는 [Scheme](https://en.wikipedia.org/wiki/Scheme_(programming_language))의 또 다른 dialect라고 할 수 있겠다. **단순하게 말하면 ANSYS Console에서 실행하는 명령어**이다. Scheme은 학부 때 과제로 경험해보고, 마법사책(SICP) 볼 때 말고는 접할 일이 없던 언어라 무척 당황했는데, 다행히 단순하게 돌릴때 Scheme문법을 사용할 일은 없었다.

애니웨이, ANSYS Fluent를 돌릴 때 크게 두 가지 케이스가 있는데 하나는 stationary simulation일 때고, 하나는 traisient simulation이다. 명령어가 살짝 다르기 때문에 각각의 journal file을 작성하고 이를 jobscript에서 fluent 실행할 때 넣어주는 방식을 취하기로 했다.

# ANSYS batch mode 방식 (자세하게), ANSYS 2020 R1 기준

1. ANSYS Fluent를 GUI로 실행한다.
2. 시뮬레이션에 필요한 Mesh 설정 및 각종 파라미터를 설정한다. TUI로 이 과정을 할 수도 있지만, 명령어를 다 알기도 힘들고, 최대한 journal file을 간단하게 만들고자 했다.
3. 이를 case file로 저장한다. (압축해서 `.cas.gz`로 내보내기를 추천) 그렇게 `.cas.gz` 파일이 생성된다.
4. mesh파일을 보통 case파일과 같은 폴더에 넣고 작업한다고 가정하고 2~3을 진행했다.
5. 서버를 쓰면, mesh 파일과 case file를 특정 디렉토리에 넣는다.
6. 다음과 같이 journal file을 작성한다.
      1. Stationary
          ```
          (set! *cx-exit-on-error* #t)

          ;; batch options
          /file/set-batch-options yes yes yes yes
          ;; read case file
          /file/read-case wst.cas.gz
          ;; disable HDF5
          /file/cff-files no
          ;; initialize the solution
          /solve/initialize/initialize/
          ;; save residuals plot as "residual-xxxx.jpg" at every 10 iteration, xxxx is a iteration number
          /solve/execute-commands/add-edit save_residuals 10 "iteration" "/solve/monitors/residual/plot? yes /display/set-window 1 /display/save-picture residual-%i.jpg"
          ;; iterate 100 step
          /solve/iterate 100
          ;; write data file as "wst.data"
          /file/write-case-data wst.data
          ;; exit FLUENT
          /exit yes
          ;; blank line at end

          ```
      2. Transient
          ```
          (set! *cx-exit-on-error* #t)

          /file/set-batch-options yes yes yes yes
          ;; read case file
          /file/read-case wst.cas.gz
          ;; disable HDF5 output
          /file/cff-files no
          ;; initialize the solution
          /solve/initialize/initialize/
          ;; save residuals plot as "residual.jpg" at every 10 time-step
          /solve/execute-commands/add-edit save_residuals 10 "time-step" "/solve/monitors/residual/plot? yes /display/set-window 1 /display/save-picture residual-%t.jpg"
          ;; time step interval for auto-save
          /file/auto-save/data-frequency 10
          ;; set suffix (in this case, time-step) for auto-saved files
          /file/auto-save append-file-name-with time-step 6
          ;; time step size (dt)
          /solve/set/transient-controls/time-step-size 8.33333e-5
          ;; iterate 10 time step, each time step has 20 iteration,
          /solve/dual-time-iterate 10 20
          ;; write data file
          /file/write-case-data wst.data
          ;; exit FLUENT
          /exit yes
          ;; blank line at end

          ```
  Line by Line으로 설명하자면
  * batch option
      ```
      (set! *cx-exit-on-error* #t)

      /file/set-batch-options yes yes yes yes
      ```
    이 부분은, overlap 되는 부분이 있긴 한데, ANSYS에서 File->Batch Options의 설정이다. `(set! *cx-exit-on-error* #t)` 이 커맨드는 Scheme문법의 ANSYS GUI 커맨드로 Exit on error를 체크하는 것이고, 아래 TUI 커맨드 `/file/set-batch-options yes yes yes yes`는 원래는 `Confirm File Overwrite`, `Hide Questions`, `Exit on Error` 이 세 가지를 묻는 옵션이었으나 20.1 기준으로는 실제 콘솔에서 `/file/set-batch-options`을 해봤을 경우 한 가지 더 묻는데, 지금 기억이 안나서 스킵.. 여튼 이걸 다 yes하는 이유는 job 실행시 저거 묻는다고 멈추는데 서버에서 그걸 interactive하게 대답하기 힘드므로 그걸 무시하기 위해서이다.

  * Read case file
      ```
      ;; read case file
      /file/read-case wst.cas.gz
      ```
      journal file의 주석은 Scheme을 따라 `;;`로 처리하였다. 단순하게 case file `wst.cas.gz`를 읽는 명령어
  * Disabling HDF5 output
      ```
      ;; disable HDF5 output
      /file/cff-files no
      ```
      개인적으로 시뮬레이션 output HDF5를 선호하는 편인데, ANSYS에서 끄는 이유는 [CFD-Post에서 지원을 안해서](https://forum.ansys.com/discussion/17423/reading-h5-file-into-ansys-for-post-processing), 근데 20.1부터 HDF5 output이 디폴트로 설정되어 있다. (..) 언젠가는 지원해주겠지만 일단은 지금은 끄자.
  * Solver initalization, 이것도 뭐 GUI에서 하는 그 intialization
      ```
      ;; initialize the solution
      /solve/initialize/initialize/
      ```
  * Plot residual
    * 우리 연구실에서 GUI를 선호하는 이유 중 하나가 실시간으로 residual 확인하면서 계산이 터지는지 안터지는지 확인하고 싶어서인데, 최근에서야 이걸 알아냈다. 처음에는 복잡하게 텍스트파일로 매번 출력해서 다운받고 다른 프로그램으로 그래프 그려서 보려고 했는데, 더 편하게 바로 plot 해줄 수 있는 명령어가 있다.
    * 기본적인 원리는 ANSYS에서 iteration 혹은 time-step별로 실행할 수 있는 커맨드를 추가할 수 있는 execute commands 기능을 사용하는 것이다. residual을 특정 iteration 혹은 time-step마다 출력하게 하고 이를 그림으로 출력하는게 그 원리
    * Stationary simulation에서는 다음 명령에서 iteration 기준으로 10번째마다 `residual-xxxx.png`를 출력하는 것. 여기서 xxxx는 iteration number를 말한다. *이렇게 하면 그림파일이 많이 나오겠지만, 이렇게 안하면 overwrite할거냐고 물어보면서 플랏이 제대로 그려지지 않는다.* 이를 `save_residuals`라는 커맨드로 저장한다.
        ```
        ;; save residuals plot as "residual-xxxx.jpg" at every 10 iteration, xxxx is a iteration number
        /solve/execute-commands/add-edit save_residuals 10 "iteration" "/solve/monitors/residual/plot? yes /display/set-window 1 /display/save-picture residual-%i.jpg"
        ```
    * Transient simulation의 경우는 time step 기준으로 10번째마다 위에서 설명한 것 같이 파일을 출력한다.
        ```
        ;; save residuals plot as "residual.jpg" at every 10 time-step
        /solve/execute-commands/add-edit save_residuals 10 "time-step" "/solve/monitors/residual/plot? yes /display/set-window 1 /display/save-picture residual-%t.jpg"
        ```
    * **너무 자주 plot하면 시뮬레이션이 느려진다** 적당히 조절하자
  * Auto-save
      ```
      ;; time step interval for auto-save
      /file/auto-save/data-frequency 10
      ;; set suffix (in this case, time-step) for auto-saved files
      /file/auto-save append-file-name-with time-step 6
      ```
    * Transient simultation은 시간이 오래 걸려서 중간 중간 저장하는게 중요한데 이 저장하는 frequency와 filename을 변경하는 옵션이다. 지금 같은 경우 10 step 마다 저장하고, 중간 저장 suffix을 time-step으로 지정하는 것. 그런데 이 옵션 제대로 테스트 안해봐서 확실하진 않다.
  * Simulation
    * Stationary simulation
      * 이건 단순하다. 100 stp의 iteration을 돌린다.
      ```
      ;; iterate 100 step
      /solve/iterate 100
      ```
    * Transient simulation
      * transient한 경우는 time step size(소위 말하는 dt)를 설정하고, 각 step 마다 몇 번 iteration을 돌리는 지 설정하고 돌려야하는데, 이렇게 돌리는게 `/solve/dual-time-iterate`에서 총 time step(10)과 각 iteration별 time step(20)을 매개변수로 넘기면 된다.
      ```
      ;; time step size (dt)
      /solve/set/transient-controls/time-step-size 8.33333e-5
      ;; iterate 10 time step, each time step has 20 iteration,
      /solve/dual-time-iterate 10 20
      ```
  * Write case
    * 계산이 끝나면 당연히 결과물을 저장해아하니깐, `/file/write-case-data` 명령어를 쓴다.
      ```
      ;; write data file
      /file/write-case-data wst.data
      ```
  * Exit Fluent
    * 다른 journal file에도 있길래 넣었는데, 이거 안넣으면 제대로 종료가 안되는 모양이다. 괜히 제대로 종료안되면 라이센스는 라이센스대로 점유하고 자원은 자원대로 못 쓸테니 써줘야한다.
    ```
    ;; exit FLUENT
    /exit yes
    ```
  * Blank line
    * 혹시 몰라 EOF(End of File)를 위해 넣었다.

7. Job scheduler file 작성 (SGE 기준)

    ```
    #!/bin/bash
    #$ -cwd
    #$ -V
    #$ -N 잡이름
    #$ -S /bin/sh
    #$ -j y
    #$ -q 큐이름
    #$ -pe ParallelEnvironment이름 코어수

    # Load module even you run jobs
    module purge
    module load ansys/20.1/fluent

    cpus=코어수

    # execute Fluent
    fluent 3ddp -rsh -t${cpus} -gu -i wst.in > wst.output
    ```

    각자 서버마다 사정이 있으므로 나머지는 알아서 하면 되지만, 맨 마지막 줄은 반드시 저렇게 해야한다.
    * 3ddp : 3D Double Precision
    * -rsh : rsh 방식으로 remote connection 구축, 이건 서버마다 환경이 다르므로 바꿔도 된다. 디폴트는 ssh방식
    * -t{cpus} : 굳이 이처럼 cpus변수 안만들고 직접 숫자 넣어줘도 된다. 다만 잡스케줄러의 코어수와는 맞춰주자.
    * -gu : GUI는 안쓰지만 그래픽은 쓰는 옵션. 일반적인 batch mode이면 `-g`를 써야하지만 residual plot때문에 그래픽이 필요하므로 `-gu`로 바꿔줬다.
    * `-i wst.in` : 위에서 저장한 journal file을 `wst.in`이라고 저장했다면, 여기서 실행할 때 'i'nput으로 넣어주는 것.
    * `> wst.output` : fluent 실행결과를 `wst.output`으로 저장하는 건데 어차피 SGE의 경우 job id에 따라 output이 따로 나오고, ANSYS transcript file(`.trn`)이 따로 생성되기 때문에 없애도 상관없을 것 같다.
8. Job submit : 위에서 저장한 잡스크립트 파일을 SGE의 job submit 명령어인 `qsub`을 통해 제출
   ```
   qsub 잡스크립트파일
   ```

이게 다이다.
복잡하지만, Journal file을 잘 만들면 GUI로 안해도 Text로 어느정도 대체할 수 있고 이를 fluent 실행시 input으로 넣어주면 된다. 사실 **계속 수정중이고, 테스트도 계속하고 있어서 완벽하다 할 수는 없다**. 그래서 보통은 내부 매뉴얼로 만들고 마는 문서인데, 너무나 문서를 찾기가 힘들어서 필요로 하시는 분들도 있을 것 같고, 정리도 할 겸 적어보았다.

# Reference
  * [KISTI Manual](https://www.ksc.re.kr/gsjw/jcs/sft#docout-7) : 국가슈퍼컴퓨팅센터 -> 기술지원 -> 소프트웨어
  * [Rescale 문서](https://docs.rescale.com/articles/ansys-start-here/) : FAQ에 Residual plot 내용이 있었다.
  * [ANSYS Fluent Getting Start guide](http://blog.ksc.re.kr/attachment/cfile9.uf@99D2954E5C0F6D9916BDE4.pdf)
  * [ANSYS Fluent TUI Command List](https://www.afs.enea.it/project/neptunius/docs/fluent/html/tuilist/main_pre.htm) 오래된 사이트라 그런지 TLS 에러가 나는데 크롬의 경우 고급눌러서 "안전하지 않음으로 이동"을 누르면 들어가진다. 참고로 12.0 기준이라 달라진 명령어가 있는데 이는 콘솔창에서 help 명령어 등을 통해 제대로 된 명령어 및 매개변수를 체크하고 사용하기 바란다.
  * [Fluent - Scheme 기초](https://davis68.github.io/me498cf-fa16/resources/flec08/handout-tui-scheme.pdf)
  * [Fluent - Scheme 문서](http://www.cfdresearch.com/wp-content/uploads/2018/08/scheme-programing-Javurek.pdf) 거의 유일한 Fluent에서의 Scheme 문서. 무려 ANSYS 5,6 기준이지만 아직도 적용되는게 많다. 이것도 독일어로 된 문서가 따로 있고, 그걸 번역한 것
  * [Fluent Troubleshooting](http://cfdyna.com/CFDHT/FluentErrors.html)