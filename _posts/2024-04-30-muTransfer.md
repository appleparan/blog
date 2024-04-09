---
layout: post
title: μTransfer (WIP)
author: jongsukim
date: 2024-04-30 11:00:00 +0900
categories: [Deep Learning, LLM]
tags:
  - LLM
  - Foundation Model
  - Greg Yang
  - Hyperparameter Tuning
  - mup
  - μTransfer
math: true
mermaid: false
---

## Introduction
오늘은 {% cite yang2022tensor --file 2024-04-30-muTransfer %}와 {% cite yang2023spectral --file 2024-04-30-muTransfer %}에 대해 리뷰해보도록 하겠다. 내가 모델을 pretrain한 경험도 없고 앞으로도 그런 기회가 있을지는 잘 모르겠지만, 내가 알기로 이 두 논문이 foundation model을 pretrain하는데 있어 가장 중요한 논문이라고 알고 있다. 

이와 별개로 이 논문이 나의 관심을 끌게된 것은 중요한 이유가 있다.
### Reynolds Number and Wind Tunnel

유체 역학의 역사 중에서 중요한 발견 중 하나가 무차원 수(Nondimensional Number)의 발견이다.
Froude number가 먼저 발견되기는 했지만, Reynolds number가 아마 가장 유명한 무차원 수가 아닐까 생각한다.
{% cite 2023TaegeeMin --file 2024-04-30-muTransfer %}

Reynolds number는 다음과 같이 유체의 동적인 점성(kinematic viscosity)인 $\nu$와 유체의 속도 $U$, 그리고 대표적인 길이 $L$과의 관계로 정의된다.
Reynolds number가 낮으면 층류(laminar flow)로 분류되고, 높으면 난류 (turblent flow)로 분류되는게 중요한 특징이다.

$$
\begin{align}
Re = \dfrac{UL}{\nu}
\end{align}
$$

왜 갑자기 Reynolds number를 설명하냐면, Renolds number가 같다면 같은 유체의 특성(층류 or 난류)을 지닌다고 볼 수 있기 때문이다. 항공기를 개발하는데 있어 실제 사이즈로 항공기를 만들지 않더라도 소형 모형으로 공기역학적 특성을 파악할 수 있는 것이다. 
그리고 이를 현실에서 테스트하기 위한 장치가 풍동(wind tunnel)이다.

{% img align="center" style='background-color: #fff' caption='<a href="https://en.wikipedia.org/wiki/File:MD-11_12ft_Wind_Tunnel_Test.jpg">MD-11 12ft Wind Tunnel Test</a>' src='/assets/images/post/2024-04-30-muTransfer/01-MD-11_12ft_Wind_Tunnel_Test.jpg' %}

그리고 초거대모델(Extremely Large Deep Learning Model)에서 {% cite yang2022tensor --file 2024-04-30-muTransfer %} 논문은 Reynolds number와 같은 역할을 한다고 볼 수 있다. 스케일이 큰 모델의 하이퍼파라미터를 작은 모델의 하이퍼파라미터를 통해 추정할 수 있게 하기 때문이다. 이걸 깨닫는 순간 가슴이 두근거릴 수 밖에 없었고, 이 논문들에 빠져들 수 밖에 없었다.

## Neural Tangent Kernel (NTK)

먼저 알아야할 것은 Neural Tangent Kernel(NTK)이다. NTK는 이름 그대로 커널(Kernel)이지만, 딥러닝을 이해하는데 있어 중요한 개념이다.
머신러닝에서의 커널이란 고차원의 특성 공간(feature space)로 데이터를 변환하는 함수를 뜻한다. 아래의 그림처럼 2차원에서 linear함수로 분류가 되지 않는 데이터도 3차원으로 변환하면 hyperplane에 의해 분류가 될 수 있음을 알 수 있다. 또한 커널을 이용하면 고차원으로 매핑하지 않고도 내적(inner product)을 간단하게 계산할 수 있는 커널 트릭(kernel trick)을 사용할 수 있게 해준다.

{% img align="center" style='background-color: #fff' caption='<a href="https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d">What is the kernel trick? Why is it important?t</a>' src='/assets/images/post/2024-04-30-muTransfer/02-kernel.webp' %}

NTK는 테일러 전개(taylor expansion)를 통해 무한한 너비(infinite width)를 가지는 simple 2-hidden layer를 랜덤 초기값(initialization)이어도 결정론적(deterministic)인 선형 함수(linear function)로 변환해주는 역할을 수행하는 이론적인 틀이라고 요약할 수 있다. {% cite jacot2018neural --file 2024-04-30-muTransfer %}

### NTK: Mathematics

{% cite yang2020tensor --file 2024-04-30-muTransfer %}의 표현에 따르면 NTK는 수학적으로는 다음과 같이 표현한다. 
어떤 parameter $\theta$에 의존하는 함수 $f$ (추후에 모델이 되는 함수)에 대해서, 초기 파라미터 $\theta_0$ 기준으로 $f$를 $\theta$과 입력값 $x$ 대해 다음과 같이 확장할 수 있다. 
이 때, $\langle , \rangle$은 내적이며, 우변은 선형 모델처럼 작동한다. 

$$
\begin{align}
f(x; \theta) - f(x; \theta_0) \approx \langle \nabla_\theta f(x; \theta_0), \theta - \theta_0 \rangle
\end{align}
$$

이 식에서 


자세한 내용은 [이 블로그](https://rajatvd.github.io/NTK/)에 정리가 잘 되어있다.

## Maximal Update Parametrization (μP)

μTransfer 을 리뷰하기 앞서 리뷰해야할 논문은 {% cite yang2020feature --file 2024-04-30-muTransfer %} 이 논문이다.

## Reference

{% bibliography --cited --file 2024-04-30-muTransfer %}

