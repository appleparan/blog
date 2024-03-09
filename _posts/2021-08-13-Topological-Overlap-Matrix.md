---
layout: post
title: Topology Overlap Matrix
author: jongsukim
date: 2021-08-13 12:00:00 +0900
categories: [Science]
tags:
  - Topology Overlap Matrix
  - WGCNA
  - Ravasz algorithm
math: true
---

# Introduction

WGCNA(WeiGhted Correlation Network Analysis) 논문을 보다가 Topology Overlap Matrix의 이해를 돕고자 간단하게 메모하면서 정리하는 글이다. 다음 논문들을 참고하였고, 실제 내용은 {% cite zhang_general_2005 --file 2021-08-13-Topological-Overlap-Matrix %}의 2.4절을 정리한 것이다.
{% cite langfelder_wgcna_2008 zhang_general_2005 --file 2021-08-13-Topological-Overlap-Matrix %}

# Measure of Node Dissimilarity

논문에 나온대로 Co-expression network analysis의 목적은 node이 tightly connected이 되었는지 감지하여 clustering하는 것이라고 할 수 있다. {% cite zhang_general_2005 --file 2021-08-13-Topological-Overlap-Matrix %}.
이를 위해 clustering method와 함께 node dissimilarity measure를 사용한다.

이중에서 Ravasz algorithm을 사용한다
{% cite ravasz_hierarchical_2002 --file 2021-08-13-Topological-Overlap-Matrix %}.
Ravasz algorithm은 similarity measure을 기준으로 쓰여져있지만,
WGCNA에서는 dissimlarity measure를 사용한다. 이는 simliarity measure를 먼저 정의한 다음 이를 반전시키는 방법을 쓰면 된다.

The topological overlap matrix (TOM), \\(\Omega = [\omega_{ij}]\\) 는 다음과 같이 정의한다.

# The Topological Overlap Matrix (TOM) in Ravasz Algorithm

Node simliarity는 어떻게 정의될 수 있을까?
위 식이 어떻게 정의가 되게 되었는지 이해가 안돼서 이 글을 쓰게 되었고, Ravasz algorithm을 찾아보았다 {% cite ravasz_hierarchical_2002 --file 2021-08-13-Topological-Overlap-Matrix %}.

* 노드의 connectivity가 높다면, 다시말하면 clustering이 이루어진다면 공유하는 이웃 노드(neighbor)들도 많을 것이다.
* 하지만 단순히 neighbor의 개수는 simlarity의 척도가 되지 못한다. normalization을 안했기 때문에 비교하기가 힘들기 때문이다.
* 따라서 노드의 각 페어를 \\(i, j\\)라 하면,
TOM은 neighbor의 개수를 connectivity로 나누어주어야 한다. 이게 Ravasz algorithm에서 정의하는 TOM이다. Ravasz 논문에서의 notation을 그대로 가져다 쓰면 다음과 같이 표현할 수 있다.

    $$
    \Omega_{ij} = \dfrac{J_{ij}}{\min{\{k_i, k_j\}}}
    $$

    \\(J_{ij}\\)는 노드 \\(i\\)와 \\(j\\)가 공유하는 neighbor의 개수,
    \\(k_i\\) 는 \\(i\\) 노드에서 다른 노드로의 direct connection의 개수라고 할 수 있다 (node connectivity).

# The Topological Overlap Matrix (TOM) in WGCNA

WGCNA에서는 위에서 정의한 TOM을 확장하여 다음과 같이 정의한다.

$$
\omega_{ij} = \dfrac{l_{ij} + a_{ij}}{\min{\{k_i,k_j\}}+1-a_{ij}}
$$

\\(l_{ij}=\sum_u a_{iu} a_{uj}\\)이며 \\(k_i = \sum_{u} a_{iu}\\)는 node connectivity를 나타낸다. \\(l_{ij}\\)는 Ravasz algorithm에서의 neighbor의 수, 즉 \\(J_{ij}\\)에 해당함을 알 수 있다. \\(a_{ij}\\)는 adjacency matrix의 weight이다. shared되는 neighbor수에 weight를 주고싶다면 \\(0<a_{ij}<1\\)의 값을 주면 되는 것이고, 그렇지 않다면 0 혹은 1을 주면 된다.

## Extreme of \\(\omega_{ij}\\)

unweighted network라고 할 때 \\(\omega_{ij}\\)의 극단적인 케이스는 논문에 나온 것처럼 다음과 같다.

* \\(\omega_{ij}=1\\)
  1. 노드 \\(i, j\\) 중에서 더 적은 노드를 \\(i\\)라고 할 때 (\\(\min{\\{k_i,k_j\\}}\\) 때문), 노드 \\(i\\)의 모든 이웃 노드는 노드 \\(j\\)의 이웃이다.
  2. 그리고, \\(i\\)와 \\(j\\)는 연결되어있다.

* \\(\omega_{ij}=0\\)
  * 노드 \\(i, j\\)는 서로 연결되어 있지 않다.

## Range of \\(\omega_{ij}\\)

\\(0 \leq \omega_{ij} \leq 1\\)인가? 그렇다.

Proof.
1. \\(l_{ij} \leq \min{\\{\sum_{u \neq j} a_{iu}, \sum_{u \neq i} a_{uj}\\}}\\) 이므로, \\(l_{ij} \leq \min{\\{k_i, k_j\\}} - a_{ij}\\) 이다. \\(l_{ij}\\)는 neighbor의 수이므로, 당연히 connectivity보다는 작을 수 밖에 없다.
2. 따라서 \\(0 \leq a_{ij} \leq 1\\)이므로 \\(0 \leq \omega_{ij} \leq 1\\)이다.  1.에서 \\(\\)\\(l_{ij} \leq \min{\\{k_i, k_j\\}} - a_{ij}\\)의 양변을 \\(\min{\\{k_i, k_j\\}}\\)로 나누면 자명하다.

# Dissimilarity measure

결론적으로 심플하게 Similarity measure를 opposite하게 만들면 된다.

$$
d_{ij}^\omega = 1 - \omega_{ij}
$$

# Reference

{% bibliography --cited --file 2021-08-13-Topological-Overlap-Matrix %}
