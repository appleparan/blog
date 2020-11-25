---
layout: post
title: Fortran의 미래
tags:
  - Fortran
  - Programming Language
  - Julia
  - Hardware
  - Apple Silicon
  - Scientific Computing
use_math: false
---

오랫동안 Fortran이 망하기만을 고대해왔다. 일단 표준 한번 나오기까지가 너무 오랜 시간 걸릴뿐더러, 그게 각 컴파일러마다 반영되기까지 시간이 걸리기도 하고, 그렇다고 사람들이 새 기능을 빠르게 흡수하지도 않는다. 많은 Fortran 코드는 90/95가 많고 Fortran 77 코드가 아직도 현업에서 쓰이는 곳도 많다.
Fortran도 나름 이런저런 개선을 한다고 하지만, 다른 언어에 비해 상당히 이질적인 문법을 가지고 있고 그럴 수 밖에 없어보인다. (i.e. OOP) 그리고 그 기능도 역시 다른 언어에 비해 제한적이다.

결국 답은 언어를 바꿔야한다라고 결론을 내렸지만, 실제 Legacy 코드를 뜯어고치기란 쉽지 않다. 나도 실패했다. 예전에는 Python과 C++, 최근에는 Julia를 해답으로 삼고 내가 쓰던 코드를 바꿔보려고 노력했고 남들에게도 권유했다.
그러나 Validation에서 문제가 생기거나 다른 사람들도 그다지 바꿀 메리트를 못 느껴서인지 적극적으로 바꾸려는 움직임을 보이지 않았다.

하지만 답은 하드웨어에 있었다. Apple Silicon이 발표되면서 본격적으로 PC도 ARM의 시대가 되었고, 10년쯤 지나면 서서히 ARM 서버도 대중화되지 않을까 싶다. GPGPU도 있다. 그러나 많은 Scientific Simulation Code들은 FP64 위주로 돌아가고 있지만, 머신러닝의 영향으로 FP32도 FP16에 비해 우선순위에서 밀리는 형국이다. 그래서 [Variable Precision Computing](https://www.osti.gov/servlets/purl/1573151)이 필요하거나 아니면 FP64에 특화된 가속기가 필요할수도 있다고 생각한다. 이미 GPGPU뿐만 아니라 다양한 가속기(Accelerator)의 시대가 도래할 것처럼 보인다. 이미 [Fast Stencil-Code Computation on a Wafer-Scale Processor](https://arxiv.org/abs/2010.03660?fbclid=IwAR0b9Vq1rPiHiPALk19yyv4UPQAkHLZ8PYhQitnuSmoZwrP5t6XMapu_g1s) 이런 식으로 새로운 가속기는 나오고 있고 테스트 되고 있다.

이런 다양한 하드웨어가 생겨나면서 기존의 Fortran은 빠르게 대처할 수 있을까? LLVM기반으로 바꾸면 좀 더 대처가 빠를 수 있겠지만, [flang](https://github.com/flang-compiler/flang)은 아직 나아갈 길이 많아 보인다. 잘 돌아간다 하더라도 HPC 유저들이 만족할만한 성능이 나올려면 시간이 좀 더 걸리지 않을까 생각한다. 결국 그나마 이쪽 업계에서 많이 쓰고 있는 C로 갈아타거나 (C나 Fortran이나...), Python(느리다)으로 오거나, Julia(속도와 편리성 둘 다 잡음)로 올 수 밖에 없지 않을까.
