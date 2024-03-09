---
layout: post
title: Maximal Overlap Discrete Wavelet Transform
author: jongsukim
date: 2021-09-10 12:00:00 +0900
categories: [Science]
tags:
  - Wavelet
  - MODWT
math: true
---

# Introduction
주파수 영역(frequency domain)은 어떤 신호(signal)의 숨겨진 특성을 드러낼 때 유용한 도구이다.
푸리에 변환(Fourier Transform)을 사용하면 시간 영역(time domain)과 주파수 영역(frequency domain)을 서로 변환할 수 있다.
그러나 이런 푸리에 변환에도 몇 가지 단점이 있다.
우선, 푸리에 변환은 사각 함수(Rectangular function)와 같은 특정 시간이나 위치에만 나타나는 함수(local function)를 표현하기 쉽지 않다. 푸리에변환은 sine과 cosine함수를 사용하기 때문에, 사각 함수를 표현하기 위해서는 수 많은 sine 및 cosine 항(term)이 필요하다. 푸리에 변환은 사각 함수의 사각형 모양은 local한 특성이지만 주파수 영역에서는 global한 특성이 될 수 있어서 약점을 드러낸다.

그리고, 문제에 따라 시간 영역과 주파수 영역이 모두 필요한 경우가 있다. 예를 들면, 이미지 압축을 하는 경우 주파수 영역 뿐만 아니라 위치(시간 영역에 대응)에 따른 정보도 필요하다. 또한 시계열(time series)의 경우 주파수 영역과 더불어 시간의 진행정보도 필요한 경우가 많다.

이를 위해서 웨이블릿 변환(Wavelet transform)이 탄생하였다. 웨이블릿 변환은 주파수 영역의 정확도를 약간 희생시키는 대신, 시간 영역의 정보를 함께 다룰 수 있는 장점이 있다. 이를 통해 앞서 언급한 단점들을 해결할 수 있다.
또한, FFT를 쓰더라도 \\(\mathcal{O}(N\log{N})\\)의 시간 복잡도를 가지는 푸리에 변환과는 달리, \\(\mathcal{O}(N)\\)의 선형복잡도를 지니고 있어 계산시간이 빠를 뿐더러, sparse한 데이터가 나오기 때문에 압축 등에 많이 쓰인다.

[Stackexchange: Difference between Fourier transform and Wavelets](https://math.stackexchange.com/questions/279980/difference-between-fourier-transform-and-wavelets)

> The goal is a new way to represent functions-especially functions that are local in time and frequency (or space and wave number). Compare with Fourier series. Sines and cosines are perfectly local in frequency, but global in \\(x\\) or \\(t\\). A short pulse has slowly decaying coefficients that are hard to measure. To reconstruct the pulse, a Fourier series depends heavily on cancellation. The whole of Fourier analysis, relating properties of functions to properties of coefficients, is made difficult (some say interesting) by the nonlocal support of \\(\sin{x}\\).

{% cite strangWaveletsDilationEquations1989 --file 2021-09-10-MODWT %}

> This global support is the one drawback to sines and cosines; otherwise, Fourier is virtually unbeatable. To represent a local function, vanishing outside a short interval of space or time, a global basis requires extreme cancellation. Reasonable accuracy needs many terms of the Fourier series. Wavelets give a local basis.

{% cite strangWaveletTransformsFourier1993 --file 2021-09-10-MODWT %}


# Haar wavelet

웨이블릿 변환(Wavelet transform)에서 가장 기본적으로 쓰이는 wavelet은 Haar wavelet이다. Haar wavelet은 sine과 cosine 함수를 basis 로 사용하는 푸리에 변환과는 달리 심플한 사각 함수를 basis로 사용한다.

![Haar Wavelet by [Omegatron](https://commons.wikimedia.org/wiki/File:Haar_wavelet.svg) / [CC](https://creativecommons.org/licenses/by-sa/3.0/) ](/assets/images/post/2021-09-10-Wavelet/Haar_wavelet.svg)
* Haar Wavelet by [Omegatron](https://commons.wikimedia.org/wiki/File:Haar_wavelet.svg) / [CC](https://creativecommons.org/licenses/by-sa/3.0/)


$$
\langle f, h_{-1} \rangle h_{-1}(x) + \langle f, h_{0} \rangle h_{0}(x) + \langle f, h_{10} \rangle h_{10}(x) + \langle f, h_{11} \rangle h_{11}(x) \cdots
$$

Haar wavelet은 두 함수의 조합을 기초로 이루어진다. 첫번째는 특성 함수 (characteristic function)의 역할을 하는 \\(h_{-1} (x)\\)이며 이를 scaling function라고 한다.

$$
h_{-1} (x) = \begin{cases}
                1 \; & 0 \leq x < 1 \\
                0 \; & \textrm{otherwise.}
            \end{cases}
$$

두번째는 위에서 정의한 \\(h_{-1} (x)\\)을 바탕으로 정의되는 \\(h_{0} (x)\\)이며 이를 wavelet function이라고 한다.

$$
h_{0} (x) = \begin{cases}
                1 \; & 0 \leq x < \dfrac{1}{2} \\
                -1 \; & \dfrac{1}{2} \leq x < 1 \\
                0 \; & \textrm{otherwise.}
            \end{cases}
$$

나머지 항(term)은 이 두 wavlet의 조합으부터 translation \\((x \rightarrow x + k) \\) 과 dyadic dilation \\( (x \rightarrow 2^j x ) \\)를 통해 구성한다. (i.e. \\(h_{10}(x), h_{11}(x), \cdots \\))

\\( h_0 (x) \\)는 2의 지수승으로 표현되는 기본적인 dilation function이라고 했을 때, \\(h_{0}(x)\\) 과 \\(h_{-1}(x)\\) 을 조합하여 scale \\(j\\)와 translation \\(j\\)에 대한 일반적인 Haar wavelet term (Haar function) 을 구성할 수 있다.
$$
h_{jk} (x) = 2^{j/2} h_0 (2^{j}x - k)
$$

각 basis들은 서로 orthonormal하지만, 미분가능하지는 않다. 그러나, 갑자기 나타나는 이벤트 (sudden transition)와 같은 신호를 변환하는데에는 좋은 효과를 보여준다.

{% cite roweDaubechiesWaveletsMathematica1995 --file 2021-09-10-MODWT %}

# Daubechies wavelet transform

Daubechies wavelet transform은 [Haar wavelet transform](
https://en.wikipedia.org/wiki/Haar_wavelet#Haar_transform)처럼 orthonormal한 basis를 유지하지만, dillation equation을 통해 scaling function과 wavelet function을 정의한다.

먼저, 가장 베이스라고 할 수 있는 scaling function(Haar wavelet transform에서의 \\(h_{-1} (x)\\))은 Daubechies basis의 order N에 따라 다음과 같이 정의한다.

$$
\phi (x) = \sqrt{2} \sum_{k=0}^{N-1} c_k \phi (2x - k)
$$

그리고 다음과 같이 normalize한다.

$$
\int \phi (x) dx = 1
$$

scaling function에 따라 정의되는 wavelet function은 다음과 같다.

$$
\psi (x) = \sqrt{2} \sum_{k=0}^{N-1} (-1)^k c_{N-1-k} \phi (2x - k)
$$

\\(c_k\\)는 filter coefficient라고 불리우며, normalization을 만족하도록 위의 두 \\(\phi (x) \\)의 식을 결합하면 다음 조건을 얻을 수 있다.

$$
\sum_{k=0}^{N-1} c_k = \sqrt{2}
$$

예를 들어, 4차 order인 D4의 \\(c_k\\) 계수는 위 조건을 바탕으로 구했을 때 다음과 같다.

[Wikipedia: Daubechies_wavelet](https://en.wikipedia.org/wiki/Daubechies_wavelet#Construction)

$$
\begin{align}
c_0 &= \dfrac{1+\sqrt{3}}{4\sqrt{2}} \\
c_1 &= \dfrac{3+\sqrt{3}}{4\sqrt{2}} \\
c_2 &= \dfrac{3-\sqrt{3}}{4\sqrt{2}} \\
c_3 &= \dfrac{1-\sqrt{3}}{4\sqrt{2}}
\end{align}
$$


{% cite roweDaubechiesWaveletsMathematica1995 --file 2021-09-10-MODWT %}.
{% cite whitcherWaveletAnalysisCovariance2000 --file 2021-09-10-MODWT %}.

# MODWT and Conclusion

DWT(Discrete Wavelet Transform)는 DFT와 같이 Wavelet Transform의 discrete한 버전이다. MODWT는 DWT와는 달리
orthonormal하지 않고, 모든 샘플링 사이즈에 대해 정의할 수 있으며, circular shift를 하더라도 power spectrum이 변하지 않는다. 그리고, 다차원 분석(MRA, multiresolution analysis)을 진행하는 경우 DWT에 비해 MODWT가 shift에 대해 추가적인 정보를 제공한다.

[Lecture note: Maximal Overlap Discrete Wavelet Transform](https://faculty.washington.edu/dbp/s530/PDFs/05-MODWT-2018.pdf)

대기과학에서의 비정상성 시계열(non-stationary time series)에는 wavelet method가 푸리에 변환에 비해 직관적이면서 다차원(multiresolution) 분산 분석(variance analysis)에 강점이 드러난다. 웨이블릿 변환 방법 중에서도 일반적인 DWT보다 MODWT가 circular shift에 민감할 뿐더러, MODWT로부터 계산된 detail(high-frequency)과 smooth(low-frequency)는 실제 발생하는 이벤트와 매칭되는 선형필터 결과라고 해석할 수 있다. 위 결과를 통해, 자연에서 실제 발생한 이벤트와 결합한 해석이 용이한 MODWT가 대기과학 시계열 분석에서 유용하다고 할 수 있다.

{% cite percivalAnalysisSubtidalCoastal1997 --file 2021-09-10-MODWT %}.

# Reference

{% bibliography --cited --file 2021-09-10-MODWT %}
