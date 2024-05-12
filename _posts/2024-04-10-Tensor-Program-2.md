---
layout: post
title: Tensor Program II (WIP)
author: jongsukim
date: 2024-04-10 09:00:00 +0900
categories: [Deep Learning, LLM]
tags:
  - Large Deep Learning Model
  - Greg Yang
  - Neural Tangent Kernel
  - Tensor Program 2
  - mup
  - μTransfer
math: true
mermaid: false
---

## Introduction
{% cite yang2022tensor --file 2024-04-10-Tensor-Program-2 %}와 {% cite yang2023spectral --file 2024-04-10-Tensor-Program-2 %}를
리뷰하기에 앞서 {% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}를 살펴보기로 하겠다.

이 논문의 핵심은 NTK를 확장하여 MLP뿐만 아니라 다른 어떤 아키텍처에서도 동일한 이론을 적용할 수 있음을 보인다.
NTK가 중요하다는 것은 알고있었지만, 너무 이상적인 이론이라고 생각하고 있었는데, 이 논문을 통해서 많은 궁금증이 풀린 경험이 있기에 소개한다.

{% cite yang2019wide --file 2024-04-10-Tensor-Program-2 %} 논문도 같이 보는게 맞으나, 다음 버전에서 잘 요약해주기도 했다고 생각하기도 하고, 무엇보다 양이 너무 많아서 생략한다.

### Neural Tangent Kernel (NTK)

먼저 알아야할 것은 Neural Tangent Kernel(NTK)이다. NTK를 통해 nonlinear한 모델을 linear하게 만들어서 training dynamics를 해석할 수 있는 길이 열렸다고 볼 수 있다.

NTK는 이름 그대로 커널(Kernel)이지만, 딥러닝을 이해하는데 있어 중요한 개념이다.
머신러닝에서의 커널이란 고차원의 특성 공간(feature space)로 데이터를 변환하는 함수를 뜻한다. 아래의 그림처럼 2차원에서 linear함수로 분류가 되지 않는 데이터도 3차원으로 변환하면 hyperplane에 의해 분류가 될 수 있음을 알 수 있다. 또한 커널을 이용하면 고차원으로 매핑하지 않고도 내적(inner product)을 간단하게 계산할 수 있는 커널 트릭(kernel trick)을 사용할 수 있게 해준다.

{% img align="center" style='background-color: #fff' caption='<a href="https://medium.com/@zxr.nju/what-is-the-kernel-trick-why-is-it-important-98a98db0961d">What is the kernel trick? Why is it important?t</a>' src='/assets/images/post/2024-04-10-Tensor-Program-2/01-kernel.webp' %}

NTK는 테일러 전개(taylor expansion)를 통해 무한한 너비(infinite width)를 가지는 simple 2-hidden layer를 랜덤 초기값(initialization)이어도 결정론적(deterministic)인 선형 함수(linear function)로 변환해주는 역할을 수행하는 이론적인 틀이라고 요약할 수 있다. {% cite jacot2018neural --file 2024-04-10-Tensor-Program-2 %}

#### NTK: Beyond Intuition

{% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}의 표현에 따르면 NTK는 수학적으로는 다음과 같이 표현한다.
어떤 parameter $\theta$에 의존하는 함수 $f$ (추후에 모델이 되는 함수)에 대해서, 초기 파라미터 $\theta_0$ 기준으로 $f$를 $\theta$과 입력값 $x$ 대해 다음과 같이 확장할 수 있다.
이 때, $\langle , \rangle$은 내적이며, 우변은 선형 모델(linear model)처럼 작동한다.

$$
\begin{align}
f(x; \theta) - f(x; \theta_0) \approx \langle \nabla_\theta f(x; \theta_0), \theta - \theta_0 \rangle
\end{align}
$$

위에서 언급했듯이 우변은 선형 모델처럼 작동한다. 위 식을 선형 모델의 형태로 다시 작성하면 다음과 같다. 이는 어떻게보면 $(fx; \theta)$의 $w$에 대한 1st order taylor expansion이라고 볼 수 있다.

$$
\begin{align}
f(x; \theta) \approx f(x; \theta_0) + \nabla_\theta f(x; \theta_0)^\mathsf{T} (\theta - \theta_0)
\end{align}
$$

위 식에서 $f(x; \theta_0)$는 초기값, $$\nabla_\theta f(x; \theta_0)$$는 $\theta_0$에 의존하므로 상수라고 생각할 수 있고, $\theta - \theta_0$는 정확히는 변수 $\theta$에 대한 선형 모델로 볼 수 있다.
하지만, 모델 함수 gradient를 찾는것은 선형(linear)이지 않기에 모델 함수는 비선형(nonlinear)이라고 생각할 수 있다.

그러나 NTK는 $$\nabla_\theta f(x; \theta_0)$$를 input featurizer 혹은 feature map이라고 불리우는 nonlinear 함수라고 증명한다.

물론 이 가정은 $\theta$와 $\theta_0$의 차이가 크지 않을 때만 잘 작동한다. 신경망(neural network)에서는 작은 learning rate로 매우 적은 시간에 훈련했을 때만 성립할 수 있다.

width가 클 수록 즉 무한 너비(infinite-width) 네트워크에서 $\theta_0$이 랜덤하게 잘 초기화되었다면, weight들이 변화량($\theta - \theta_0$에 대한 기대값이 거의 0에 가깝다. 여기에 대한 직관적인 설명은 width가 클 수록 모델의 output에 영향을 주는 weight들이 많아지기 떄문에, $\theta$의 작은 변화라도 $f$에는 영향이 클 수 있다는 점이다. 따라서 weight가 굉장히 조금만 움직이게 되고 이는 모델이 lienar하게 작동할 수 있게 된다.

내가 느끼기엔 이 가정은 수치해석에서의 [Euler Method](https://en.wikipedia.org/wiki/Euler_method)의 가정과 별로 차이가 없어보인다. Unstable하지만 않는다면 복잡한 식이어도 매우 작은 time step을 가정한다면 (비효율적이지만) linear하게 근사해서 풀 수 있기 때문이다.

NTK 논문 {% cite jacot2018neural --file 2024-04-10-Tensor-Program-2 %}은 이 직관을 infinite width 모델이기만 하면 어떤 데이터든간에 적용할 수 있음을 보였다. 이를 통해, 비선형 모델도 선형처럼 해석이 가능해지고, 이는 training dynamics를 해석할 수 있게 만들어준다.

#### NTK: Gradient Flow

NTK논문은 gradient flow라는 것을 제시하여 gradient descent에서의 training dynamics를 해석하고자 한다.

모델 weight $\theta$가 업데이트되는 과정을 다음과 같이 나타내면, 이를 learning rate $\eta$를 일종의 time처럼 생각하는 1D ODE로 표현할 수 있게 되고, ODE의 해를 구하면 언제나 해가 존재할 수 있음을 증명했다. 우선 $L$을 $f$에 대한 loss function라고 하자.

$$
\begin{align}
\theta_{k+1} &= \theta_k - \eta \nabla_\theta L(\theta_k) \\
\dfrac{\theta_{k+1} - \theta_k}{\eta} &= -\nabla_\theta L(\theta_k) \\
\dfrac{d \theta(t)}{dt} &= -\nabla_\theta L(\theta (t)) \\
\end{align}
$$

마지막 식은 1D ODE 가장 기본적인 형태 그 자체라고 할 수 있다. 하지만, 그 주체가 $\theta$일 뿐. 그래서 Gradient Flow라고 이름붙였다고 생각한다.

여기서 loss function $\mathcal{L}$을 MSE function이라고 가정하고, $f^*$를 정답 레이블이라고 하자. Loss function을 풀어써서 미분을 적용하면 다음과 같다.

$$
\begin{align}
\dfrac{d \theta(t)}{dt} &= -\nabla_\theta \mathcal{L}(\theta (t)) \\
\dot{\theta}(t) &= -\nabla_\theta (f(\theta) - f^*)^2\\
\dot{\theta} &= -(\nabla_\theta f(\theta)) (f(\theta) - f^*)
\end{align}
$$

이를 $f$에 적용하기 위해 Chain rule를 적용한다.

$$
\begin{align}
\dot{f}(\theta) &= \dfrac{d f(\theta(t))}{d \theta(t)} \dfrac{d \theta(t)}{dt} = \nabla_\theta f(\theta)^\mathsf{T} \dot{\theta} \\
\dot{f}(\theta) &= \nabla_\theta f(\theta)^\mathsf{T} \dot{\theta} = -\nabla_\theta f(\theta)^\mathsf{T} \nabla_\theta f(\theta) (f(\theta) - f^*) \\
\end{align}
$$

여기서 나온 $$\nabla_\theta f(\theta)^\mathsf{T} \nabla_\theta$$를 **NTK(Neural Tangent Kernel)**이라고 정의한다.

좀 더 자세한 내용은 원 논문과 {% cite jacot2018neural --file 2024-04-10-Tensor-Program-2 %} [이 블로그](https://rajatvd.github.io/NTK/)에 정리가 잘 되어있다. 개인적으로는 논문은 어려워서 이해가 잘 안됐지만, 해당 블로그가 정말 쉽게 잘 설명되어 있어서 읽기 좋았다.

#### NTK: NTK INIT

위의 {% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}의 표현으로 다시 바꾸고 정리하면 다음과 같다. $f(x; \theta)$를 파라미터 $\theta$와 input $x$에 대한 신경망이라고 할 때, $\mathcal{L}$을 Loss, $y$를 label라고 하자. 서로 다른 input $x$와 $\bar{x}$에 대해서 NTK $\Theta$를 다음과 같이 정의할 수 있다.

$$
\begin{align}
f_t - f_{t-1} &\approx -\eta \mathcal{\Theta} \mathcal{L}' (f_t, y) \\
\Theta (x, \bar{x}) &\stackrel{\text{def}}{=} \langle \nabla_\theta f(x; \theta_0), \nabla_\theta  f(\bar{x}; \theta_0) \rangle
\end{align}
$$

또한 {% cite jacot2018neural --file 2024-04-10-Tensor-Program-2 %}에서 보여줬듯이 $\theta$가 랜덤하게 잘 intialized되었고, $f$의 width가 충분히 크다면 (infinite-width), $\Theta$는 deterministic한 $\mathring{\Theta}$로 수렴한다.

이를 수학적으로 표현하면, $L$개의 hidden layer를 가지며, layer $l$의 width를 $n^l$이라고 할 때, NTK $\Theta (x, \bar{x})$는 $\theta$가 랜덤이어도 deterministic한 kernel $\mathring{\Theta} (x, \bar{x})$으로 수렴한다.

$$
\begin{align}
\Theta \stackrel{p}{\rightarrow} \mathring{\Theta} \textrm{ as } n^1, \dots, n^L \rightarrow \infty \textrm{ in that sequence}
\end{align}
$$

#### NTK: NTK TRAIN

수렴 여부뿐만 아니라 이 MLP $f$의 훈련과정을 생각해보자. Loss function $\mathcal{L}$을 사용하여 gradient descent로 train하는 하는 시간을 $t$라고 정의하자. 처음 가우시안 랜덤변수로 초기화한 MLP를 $f_0$, 시간에 따른 MLP를 $f_t$라고 했을 때, 어떤 고정된 시간 $T$에 대해서 width가 충분히 크다면 MLP 모델은 $\mathring{f}$로 수렴한다.

$$
\begin{align}
f_t &\rightarrow \mathring{f}_t \textrm{ for all } t < T, \mathrm{ where } f_0 \rightarrow \mathring{f}_0 \\
\partial_t \mathring{f}_t &= -\eta \mathring{\Theta} \cdot \nabla_f \mathcal{L}(\mathring{f}_t)
\end{align}
$$

위에서도 유도했듯이 위 식은 true label $f^*$에 대해 1D ODE로 변환될 수 있다.

$$
\begin{align}
\mathring{f}_t - f^* = e^{-\eta t \mathring{\Theta}} (f_0 - f^*)
\end{align}
$$


## NTK Decomposition

MLP용 NTK를 다른 모델(RNN, transformer 등)에 확장하기 위해서는 기존 MLP 표현법에 조금 변화가 필요하다. 왜냐하면, {% cite jacot2018neural --file 2024-04-10-Tensor-Program-2 %} 원 논문의 방법으로는 MLP가 귀납적(inductive)으로 표현되어 있어서 확장하기가 어렵기 때문이다. 이렇게 변형된 표현의 의미를 이해하는 것이 {% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}의 핵심적인 내용이다.

원래의 방식을 NTK parameterization이라고 하는데 다음과 같이 정의한다.

input $\xi \in \mathbb{R}^{n^0}$, output dimension $n^{L+1}=1$이라고 할 떄, MLP를 $f(\xi; \theta) = W^{L+1} x^L(\xi)$라고 표현하면, $l=2, \dots, L$에 대해서 재귀적으로 다음과 같이 정의할 수 있다.

$$
\begin{align}
h^l(\xi) &= W^l x^{l-1}(\xi) + b^l \in \mathbb{R}^{n^l} \\
x^l(\xi) &= \phi(h^l(\xi)) \\
h^1(\xi) &= W^1 \xi + b^1 \in \mathbb{R}^{n^1}
\end{align}
$$

{% img align="center" style='background-color: #fff' caption='NTK Parameterization' src='/assets/images/post/2024-04-10-Tensor-Program-2/02-NTK-parameterization.png' %}

MLP Parameter는 $$\theta = \{ w^l \in \mathbb{R}^{n^l \times n^{l-1}}\}_{l=1}^{L+1} \cup \{ b^l \in \mathbb{R}^{n^l }\}_{l=1}^{L}$$로 정의되고, $W^l$은 $w^l$을 $$\sqrt{n^{l-1}}$$로 나눠준 값으로 정의한다. $$W^l= \dfrac{1}{\sqrt{n^{l-1}}} w^l$$ 여기서 $\phi$는 activation function이다. 이는 {% cite poole2016exponential --file 2024-04-10-Tensor-Program-2 %}로부터 내려오는 유구한 notation이다.

이제 NTK parameterization을 NTK의 정의에 결합시킨다.

$$
\begin{align}
\Theta (\xi, \bar{\xi}) &= \langle \nabla_\theta f(\xi; \theta_0), \nabla_\theta  f(\bar{\xi}; \theta_0) \rangle \\
&= \sum_{l=1}^{L+1} \langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle + \sum_{l=1}^L \langle \nabla_{b^{l}} f(\xi),\nabla_{b^{l}} f(\bar{\xi}) \rangle
\end{align}
$$

$$W^l= \dfrac{1}{\sqrt{n^{l-1}}} w^l$$와 chain rule을 고려하면, $\nabla_{w^{l}} f(\xi)$는 다음과 같이 두 matrix의 곱으로 표현할 수 있으며 이는 $n^l \times 1 $와 $1 \times n^{l-1}$의 곱인 $ n^l \times n^{l-1}$ matrix이다.

$$
\begin{align}
\nabla_{w^{l}} f(\xi) = \left( \dfrac{1}{\sqrt{n^{l-1}}} \nabla_{h^l} f(\xi) \right) \left( x^{l-1}(\xi)^\mathsf{T} \right)
\end{align}
$$

편의를 위해 논문에 나온 abbreviation($\bullet = \bullet (\xi)$,$\bar{\bullet} = \bullet (\bar{\xi})$)을 사용하고,
$$dh^l = \sqrt{n^{l}} \nabla_{h^{l}} f(\xi)$$와 $$d\bar{h}^l = \sqrt{n^{l}} \nabla_{h^{l}} f(\bar{\xi})$$이라는 것을 정의하면,

$$
\begin{align*}
\nabla_{w^{l}} f(\xi) &= \dfrac{1}{\sqrt{n^{l-1}}} \nabla_{h^l} f(\xi)  x^{l-1}(\xi)^\mathsf{T}  \\
&= \dfrac{1}{\sqrt{n^{l-1}} \sqrt{n^{l}}} \sqrt{n^{l}} \nabla_{h^l} f(\xi)  x^{l-1}(\xi)^\mathsf{T} \\
&= \dfrac{1}{\sqrt{n^{l} n^{l-1}}} dh^l x^{l-1}(\xi)^\mathsf{T}
\end{align*}
$$

$$
\begin{align*}
\nabla_{w^{l}} f(\bar{\xi}) &= \dfrac{1}{\sqrt{n^{l-1}}} \nabla_{h^l} f(\bar{\xi})  x^{l-1}(\bar{\xi})^\mathsf{T} \\
&= \dfrac{1}{\sqrt{n^{l-1}} \sqrt{n^{l}}} \sqrt{n^{l}} \nabla_{h^l} f(\bar{\xi})  x^{l-1}(\bar{\xi})^\mathsf{T} \\
&= \dfrac{1}{\sqrt{n^{l} n^{l-1}}} d\bar{h}^l x^{l-1}(\bar{\xi})^\mathsf{T}
\end{align*}
$$

자 이제 원래 내적(inner product)에 넣어서 계산해보자. 내적을 trace inner product로 표현하고, cyclic property of trace inner product ($Tr(ABC) = Tr(BCA) = Tr(CBA)$)를 사용하면 다음과 같다.

$$
\begin{align*}
\langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle &= \dfrac{1}{n^{l} n^{l-1}} \langle dh^l x^{l-1 \mathsf{T}}, d\bar{h}^l \bar{x}^{l-1 \mathsf{T}} \rangle \\
&=\dfrac{1}{n^{l} n^{l-1}} Tr\left( \left(dh^l x^{l-1 \mathsf{T}} \right)^\mathsf{T} d\bar{h}^l \bar{x}^{l-1 \mathsf{T}} \right) \\
&=\dfrac{1}{n^{l} n^{l-1}} Tr\left( x^{l-1} dh^{l \mathsf{T}} d\bar{h}^l \bar{x}^{l-1 \mathsf{T}} \right) \\
&=\dfrac{1}{n^{l} n^{l-1}} Tr\left( x^{l-1} \left(dh^{l \mathsf{T}} d\bar{h}^l \right) \bar{x}^{l-1 \mathsf{T}} \right) \\
&=\dfrac{1}{n^{l} n^{l-1}} Tr\left( \left(dh^{l \mathsf{T}} d\bar{h}^l \right) \bar{x}^{l-1 \mathsf{T}} x^{l-1}  \right) \\
&=\left(\dfrac{dh^{l \mathsf{T}} d\bar{h}^l}{n^{l}} \right)  \left( \dfrac{\bar{x}^{l-1 \mathsf{T}} x^{l-1}}{n^{l-1}} \right)\\
\end{align*}
$$

마지막에 $Tr$이 사라지는 것은 $dh^{l \mathsf{T}} \in \mathbb{R}^{1\times n^l}, d\bar{h}^l \in \mathbb{R}^{n^l \times 1}$이고, $\bar{x}^{l-1 \mathsf{T}} \in \mathbb{R}^{1\times n^{l-1}}, x^{l-1} \in  \mathbb{R}^{n^{l-1} \times 1}$이라서 각각 scalar 값이 나오기 때문이다.

결론적으로 NTK를 두 입력 $\xi, \bar{\xi}$에 대해 decompose하면 $$\bar{x}^{l-1 \mathsf{T}} x^{l-1}$$와 $$dh^{l \mathsf{T}} d\bar{h}^{l \mathsf{T}} / n^l$$의 곱으로 표현할 수 있고, 이는 각각 forward와 backward quantity라고 간주할 수 있다. 다음 두 섹션은 각 quantity에 대해 자세하게 분석해볼 예정이다.

### Limits of Forward Quantities $x^{l \mathsf{T}} \bar{x}^{l} / n^l$

기존에 {% cite poole2016exponential --file 2024-04-10-Tensor-Program-2 %}, {% cite schoenholz2016deep --file 2024-04-10-Tensor-Program-2 %}에서 딥러닝을 mean-field theory로 설명하고자 했다. 이 논문들에서 나온 아이디어를 바탕으로 $\bar{x}^{l \mathsf{T}} x^{l}$를 분석할 수 있다.

평균장 이론(mean field theory)은 원래 통계물리학에서 각 개별입자가 전체 시스템의 평균적 효과에 의해 영향을 받는다고 가정하여 계의 거동을 설명하는 이론이다. 물리학에서의 복잡계를 딥러닝이라고 간주하면, 각 뉴런을 입자에 대응시킬 수 있고, 입자의 상호작용을 평균적인 필드에 대해서 설명하는 mean field theory를 딥러닝에 적용할 수 있게 된다. 유체역학에서 control volume을 사용하는것과 아이디어는 비슷하다고 생각된다. 자세한 내용은 다음 영상을 확인하면 좋다.

{% youtube "https://www.youtube.com/watch?v=FlR8CvyaE4I" %}

여튼, 여기서 중요한 것은 mean field theory를 적용할때는 각 요소는 독립적으로 행동한다고 가정하기 때문에, weight와 bias는 각각 가우시안 분포를 따른다고 가정한다. 레이어 $l$의 요소 $\alpha \in [n^l]$가 있다고 가정할 때, $\alpha$에 따른 좌표 $(W^l x^{l-!})$는 다음과 같이 표현할 수 있다.

$$
\begin{align}
(W^l x^{l-1})_\alpha = \sum_{\beta=1}^n W_{\alpha \beta}^l x_\beta^{l-1}
\end{align}
$$

이는 $$h^l(\xi) = W^l x^{l-1}(\xi) + b^l$$을 각 원소 좌표 $\alpha, \beta$에 대해 풀어쓴거라고 볼 수 있다. $$W_{\alpha \beta}^l x_\beta^{l-1}$$는 roughly i.i.d. random variable 이다. 처음 레이어에서는 완벽하게 i.i.d.를 맞춰서 샘플링하더라도 훈련이 진행되면서 weight의 분포가 달라지거나 레이어 간의 상관관계가 생겨서 i.i.d.가정이 깨지기 때문이다. 하지만 i.i.d.처럼 취급한다. 그런 이유로 Gaussian distribution을 따르는 roughly i.i.d. 랜덤변수의 평균은 0이고, $$W_{\alpha \beta}^l x_\beta^{l-1}$$의 분산은 다음과 같이 구할 수 있다. ($Var(X)=E(X^2) - [E(X)]^2$)

$$
\begin{align*}
\mathbb{E}(W^l x^{l-1})^2_\alpha &= \mathbb{E}((W^l x^{l-1})^2_\alpha)\\
&= \mathbb{E}\left(\left(\sum_{\beta=1}^n W_{\alpha \beta}^l x_\beta^{l-1}\right)^2\right) \\
&= \mathbb{E}\left(\left(W_{\alpha 1}x_1 + W_{\alpha 2}x_2 + \cdots + W_{\alpha n}x_n\right)^2\right) \\
&= \mathbb{E}(W_{\alpha 1}^2 x_1^2 + W_{\alpha 2}^2 x_2^2 + \cdots + W_{\alpha n}^2 x_n^2 + 2 W_1 x_1 W_2 x_2 ) \\
&= \mathbb{E}(W_{\alpha 1}^2 x_1^2 + W_{\alpha 2}^2 x_2^2 + \cdots + W_{\alpha n}^2 x_n^2 ) \\
&= \mathbb{E}(W_{\alpha 1}^2 x_1^2) + \mathbb{E}(W_{\alpha 2}^2 x_2^2) + \cdots + \mathbb{E}(W_{\alpha n}^2 x_n^2 ) \\
&= \mathbb{E}(W_{\alpha 1}^2) \mathbb{E}(x_1^2) + \mathbb{E}(W_{\alpha 2}^2) \mathbb{E}(x_2^2) + \cdots + \mathbb{E}(W_{\alpha n}^2) \mathbb{E}(x_n^2 ) \\
&= \sum_{\beta=1}^n \mathbb{E}((W_{\alpha \beta}^l)^2) \mathbb{E}((x_{\beta}^{l-1})^2) \\
&= \lVert x \rVert^2 / n^{l-1} \\
&\approx C^{l-1} (\xi, \xi)
\end{align*}
$$

위 식에 필요한 정보는 roughly i.i.d. 특성에 의해 $$ \mathbb{E}(2 W_1 x_1 W_2 x_2 ) = 2 \mathbb{E}(W_1) \mathbb{E}(x_1) \mathbb{E}(W_2) \mathbb{E}(x_2) = 0$$이 된다는 점과, NTK의 Lecun initialization에 의해 $$W_{\alpha \beta}^l \sim \mathcal{N} \left(0, \dfrac{1}{n^l}\right)$$ 라는 점이다. 여기서 $C$는 어떤 deterinstic한 scalar값이다.

종합하면 $$(W^l x^{l-1})_\alpha \sim \mathcal{N} (0, C^{l-1} (\xi, \xi))$$이 되며 마찬가지로 $\bar{x}$에 대해서도 $$(W^l \bar{x}^{l-1})_\alpha \sim \mathcal{N} (0, C^{l-1} (\bar{\xi}, \bar{\xi}) )$$ 로 나타낼 수 있다. 두 랜덤 변수 $$((W^l x^{l-1})_\alpha, (W^l \bar{x}^{l-1})_\alpha )$$, 즉 $$(x^{l-1}_\alpha, \bar{x}^{l-1}_\alpha)$$는 jointly Gaussian이며 이들의 covariance는 $$C^{l-1} (\xi, \bar{\xi})$$ 이다.

$$x^{l} (\xi) = \phi (h^l (\xi))$$ 이고, Lecun initialization은 bias $b \sim \mathcal{N} (0, 1)$이라고 정의하며, 이를 종합하면 이 전 layer의 scalar $$C^{l-1}$$에 의존하는
{% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}의 Eq. (6)이 나오게 된다.

$$
\begin{align}
C^{l} (\xi, \bar{\xi}) = \mathbb{E} (\phi(\xi) \phi(\bar{\xi})), \textrm{ where } \\
(\xi, \bar{\xi}) \sim \mathcal{N} \left(0, \begin{pmatrix}
C^{l-1} (\xi, \xi) & C^{l-1} (\xi, \bar{\xi}) \\
C^{l-1} (\xi, \bar{\xi}) & C^{l-1} (\bar{\xi}, \bar{\xi})
\end{pmatrix}
  + 1 \right)
\end{align}
$$

다시 정리하자면 forward quantities의 covariance는 다음과 같이 $C$로 수렴한다.

$$
\begin{align}
\dfrac{x^{l \mathsf{T}} \bar{x}^{l}}{n^l} \rightarrow C^l (\xi, \bar{xi})
\end{align}
$$

### Limits of Backward Quantities $dh^{l \mathsf{T}} d\bar{h}^{l} / n^l$

우선 각 layer의 크기가 다르면 복잡하므로 $$n^1 = \cdots = n^L$$이라고 가정한 뒤 시작한다. 그리고 $$dh^l = \sqrt{n^{l}} \nabla_{h^{l}} f(\xi)$$을 정의했던 것처럼 $$dx^l_\alpha$$를 $$(W^{l+1 \mathsf{T}}dh^{l+1})_\alpha$$이라고 정의한다. 이는 레이어 $l+1$에서 $l$로 전파되는 gradient이다.

1. Backpropagation을 돌이켜보면 forward propagation은 다음과 같이 전파된다.
  $$
  \begin{align*}
  h^l (\xi) = W^l x^{l-1} (\xi) + b^l
  \end{align*}
  $$
2. 각 레이어의 결과 $h^l$은 activation function $\phi$를 통해 최종 출력 $x^l$로 변환된다.
  $$
  \begin{align*}
  x^l = \phi(h^l)
  \end{align*}
  $$
3. 최종 출력 레이어에서 loss function $\mathcal{L}$에 대한 출력 값의 gradient를 계산한다.
  $$
  \begin{align*}
  \nabla_{x^L} \mathcal{L}
  \end{align*}
  $$
4. 여기서 나온 $\nabla_{x^L} \mathcal{L}$의 backpropagation 과정의 첫번째 gradient이다.
각 레이어 l에 대해 activation function의 미분 $\phi'$를 적용해서 기울기를 구한다. 이 때 $\odot$은 Hadamard product를 의미한다.
  $$
  \begin{align*}
  \nabla_{h^l} \mathcal{L} = \nabla_{x^l} \mathcal{L} \odot \phi'(h^l)
  \end{align*}
  $$
5. MLP 레이어의 weight와 bias에 대해 loss에 대한 gradient를 계산한다. 이 때 bias는 batch 전체의 weight를 더해주어야 한다.
  $$
  \begin{align*}
  \nabla_{W^l} \mathcal{L} = \nabla_{h^l} \mathcal{L} \cdot (x^{l-1})^{\mathsf{T}}, \nabla_{b^l} \mathcal{L} &= \sum \nabla_{h^l} \mathcal{L}
  \end{align*}
  $$
6. 현재 레이어 $l$의 weight $W$의 trace를 사용하여 이전 레이어 $l-1$의 출력에 대한 gradient를 계산한다.
  $$
  \begin{align*}
  \nabla_{x^{l-1} \mathcal{L}} = (W^l)^{\mathsf{T}} \nabla_{h^l} \mathcal{L}
  \end{align*}
  $$

이 과정을 새로운 $$d x^l_\alpha$$의 정의와 결합하면 다음과 같다. 위에서 사용한 $\nabla$대신 $d$를 써준다고 생각하면 이해하기 쉽다.

$$
\begin{align}
d x^l_\alpha &\stackrel{\text{def}}{=} (W^{l+1 \mathsf{T}}dh^{l+1})_\alpha \\
&= (W^{l+1 \mathsf{T}} (dx^{l+1} \odot \phi'(h^{l+1})))_\alpha \\
&= \sum_\beta W^{l+1}_{\beta \alpha} dx^{l+1}_{\beta} \phi' (h^{l+1}_\beta)
\end{align}
$$

전부가 i.i.d.라서 central limit theorem을 사용하면 좋겠지만, $$h^{l+1}_\beta$$는 모든 $\gamma$에 대해 $$W^{l+1}_{\beta \gamma}$$에 의존하기 때문에 i.i.d. 를 만족하지 못한다.
그러나, {% cite poole2016exponential --file 2024-04-10-Tensor-Program-2 %}와 {% cite poole2schoenholz2016deep016exponential --file 2024-04-10-Tensor-Program-2 %}에 따르면, 이 의존성은 무시해도 된다고 한다. 이는 mean field theory에서 뉴런의 크기 $n^l$이 크다면, weight와 bias는 독립적으로 작용하여 이전 레이어의 영향을 받지 않고 $h$는 이런 $W$와 $b$의 weighted sum이라고 표현할 수 있기 때문이다.

이를 통해 {% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}에서는 다음과 같은 아주 재미있는 Heuristic을 사용하게 된다. {% cite schoenholz2016deep --file 2024-04-10-Tensor-Program-2 %}의 Section 4와  {% cite yang2017mean --file 2024-04-10-Tensor-Program-2 %}의 Axiom 3.2를 참고하면 되겠다. Forward와 Backward pass의 weight가 서로 독립적이라니 이 얼마나 재밌는 가정이지 않은가!

> Heuristic 4.1 (gradient independent assumption, or GIA), For any matrix $$W$$, we assume $$W^{\mathsf{T}}$$ used in backprop is independent from $$W$$ used in forward pass.

여튼, 위 가정과 함께 Limits of Forward Quantities $$\bar{x}^{l-1 \mathsf{T}} x^{l-1}$$에서 했던 방식을 똑같이 적용할 수 있다. Backward pass의 weight는 forward pass때의 weight와 독립적이니 동일한 방식을 의심없이 적용할 수 있게 되었다.
그러면 $x$대신에 $dh$로 기호를 바꾼셈이 되어버려서 $$d x^{l}_\alpha$$는 $$\mathcal{N}(0, \lVert d h^{l+1} \rVert^2 / n^{l+1})$$ 분포를 따르며 $$\alpha$$에 대해 roughly i.i.d.를 만족한다고 할 수 있다.
또한 pair $$(dx^l_\alpha, d \bar{x}^l_\alpha) \stackrel{\text{def}}{=} ((W^{l+1 \mathsf{T}}dh^{l+1})_\alpha, (W^{l+1 \mathsf{T}}d \bar{h}^{l+1})_\alpha) $$ 역시 zero mean과 $$\lVert d h^{l+1} d \bar{h}^{l+1} \rVert^2 / n^{l+1}$$를 만족하는 $$\alpha$$에 대한 i.i.d.분포라고 할 수 있다. (jointly Gaussian)
$$dx^l_\alpha$$뿐만 아니라 $$h$$로 확장하면 $$(h^{l}, \bar{h}^{l})$$ 역시 roughly i.i.d이기 때문에  $$(dh^{l}_\alpha, \bar{h}^{l}_\alpha) = (d x_\alpha^{l} \phi'(h^l_\alpha), d \bar{x}_\alpha^{l} \phi'(\bar{h}^l_\alpha) )$$ 도 비슷한 결과를 가진다고 얘기할 수 있다.

이로써 Backward quantities $dh^{l \mathsf{T}} d\bar{h}^{l} / n^l$에 도달하였다.
이전 섹션에서의 $C$대신 Scalar $D^l (\xi, \bar{\xi})$를 도입하여 variance를 표현하면 다음과 같은 backward quantities의 covariance는 $D$로 수렴한다.

$$
\begin{align}
\dfrac{dh^{l \mathsf{T}} d\bar{h}^{l}}{n^l} \rightarrow D^l (\xi, \bar{\xi})
\end{align}
$$

그리고 $D^l (\xi, \bar{\xi})$ 도 다음과 같은 재귀함수로 정의된다.

$$
\begin{align}
D^l (\xi, \bar{xi}) &= \mathbb{E}_{\eta \bar{\eta}} \mathbb{E} \phi'(\xi) \phi'(\bar{\xi}) = D^{l+1} (\xi, \bar{\xi}) \mathbb{E} \phi'(\xi) \phi'(\bar{\xi}) \\
\textrm{ where } (\eta, \bar{\eta}) &\sim \mathcal{N} \left(0, \begin{pmatrix}
D^{l+1} (\xi, \xi) & D^{l+1} (\xi, \bar{\xi}) \\
D^{l+1} (\xi, \bar{\xi}) & D^{l+1} (\bar{\xi}, \bar{\xi})
\end{pmatrix}\right) \\
(\xi, \bar{\xi}) &\sim \mathcal{N} \left(0, \begin{pmatrix}
C^{l} (\xi, \xi) & C^{l} (\xi, \bar{\xi}) \\
C^{l} (\xi, \bar{\xi}) & C^{l} (\bar{\xi}, \bar{\xi})
\end{pmatrix}
  + 1 \right)
\end{align}
$$

### Foward quantities $x^{l \mathsf{T}} \bar{x}^{l} / n^l$ + Backward quantities $dh^{l \mathsf{T}} d\bar{h}^{l} / n^l$

이전에 NTK Decomposition을 사용하여 다음과 같이 유도하였다.

$$
\begin{align*}
\langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle = \left(\dfrac{dh^{l \mathsf{T}} d\bar{h}^l}{n^{l}} \right)  \left( \dfrac{\bar{x}^{l-1 \mathsf{T}} x^{l-1}}{n^{l-1}} \right)
\end{align*}
$$

지금까지 각 항에 대해 정의한 $C$와 $D$에 대해 표현하면 다음과 같다.

$$
\begin{align*}
\langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle = C^{l-1} (\xi, \bar{\xi}) D^{l} (\xi, \bar{\xi}), \forall l \in [2, L]
\end{align*}
$$

마찬가지로, bias에 대해서도 $\nabla_{b^l} f(\xi) = \nabla_{h^l} f(\xi) = dh^l / \sqrt{n^l}$이므로,

$$
\begin{align*}
\langle \nabla_{b^{l}} f(\xi),\nabla_{b^{l}} f(\bar{\xi}) \rangle = D^{l} (\xi, \bar{\xi}), \forall l \in [2, L]
\end{align*}
$$

기존 NTK 정의와 결합하면
$$
\begin{align}
\Theta (\xi, \bar{\xi}) &= \langle \nabla_\theta f(\xi; \theta_0), \nabla_\theta  f(\bar{\xi}; \theta_0) \rangle \\
&= \sum_{l=1}^{L+1} \langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle + \sum_{l=1}^L \langle \nabla_{b^{l}} f(\xi),\nabla_{b^{l}} f(\bar{\xi}) \rangle \\
&= \sum_{l=1}^{L+1} C^{l-1} (\xi, \bar{\xi}) D^{l} (\xi, \bar{\xi}) + \sum_{l=1}^L D^{l} (\xi, \bar{\xi}) \\
\end{align}
$$

이를 통해 MLP에 대해서 기존 NTK보다 훨씬 심플하게 $C$와 $D$라는 Scalar의 곱으로 표현하였다. 이 논문의 키 포인트는 NTK로부터 해당 식을 유도하고, 이를 다른 아키텍처 즉 CNN이나 RNN으로 확장하고자 하는 것이다.

## NTK -> Any Architecture

{% cite yang2020tensor --file 2024-04-10-Tensor-Program-2 %}은 지금까지의 과정이 과연 generalized될 수 있는가에 대해 독자들이 생각할 수 있는 질문에 대한 답을 미리 준비해 놓았다. 지금까지의 과정은 MLP였기 때문에 가능한 것이 아닌가라는 의심은 당연한 것이기 때문이다.

### NTK Decomposition을 의미있게 일반화할 수 있는가?

다음 분해는 MLP라는 가정하에서 이루어졌다. 그러나 $$\frac{dh^{l \mathsf{T}} d\bar{h}^l}{n^l}$$같은 값들이 수렴할지 어떻게 알 수 있으며, 발산하지 않는다는 보장이 어디 있는가?
$$
\begin{align}
\Theta (\xi, \bar{\xi}) &= \langle \nabla_\theta f(\xi; \theta_0), \nabla_\theta  f(\bar{\xi}; \theta_0) \rangle \\
&= \sum_{l=1}^{L+1} \langle \nabla_{w^{l}} f(\xi),\nabla_{w^{l}} f(\bar{\xi}) \rangle + \sum_{l=1}^L \langle \nabla_{b^{l}} f(\xi),\nabla_{b^{l}} f(\bar{\xi}) \rangle \\
\end{align}
$$

이에 대해 저자는 NTK를 NTK Decomposition, 즉 inner product의 곱셈의 합으로 분해할 수 있도록 일반화할 수 있다고 생각한다. NTK가 수렴한다면 말이다. 이는 다음 섹션 Strategy for Computing the Infinite-Width NTK에서 어떻게 일반화할지 보여줄 예정이다.

### GIA를 계속 가정해도 되는가?

아까도 재밌다고 언급했는데, forward pass와 backward pass의 gradient가 서로 독립적이라는 가정이 과연 지속적으로 유효한 가정인가에 대한 의문은 있을 수 있다.

### 현대의 복잡한 신경망에 대해 적용할 수 있는가?

CNN, RNN, LSTM 등 뿐만 아니라 ResNet, transformer에도 NTK Decompositon을 적용할 수 있는가에 대해서 당연히 의문이 들 수 밖에 없다.

## Reference

{% bibliography --cited --file 2024-04-10-Tensor-Program-2 %}

