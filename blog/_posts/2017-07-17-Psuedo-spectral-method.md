---
layout: post
title: Navier Stokes Equation Solver for Homogeneous Isotropic Turbulence
tags:
  - turbulence
  - pseudo-spectral method
use_math: true
---

(This content is originally written by ‪Kyongmin Yeo's manual)

# Introduction

The spectral method is solving certain differential equation by some "basis function", typically sinusoids with Fourier method.
With the Navier-Stokes equation, it can remove presssure term in N-S equation and solve viscous term analytically.

Pros:
  * Removing pressure term is huge performance advantage
  * Accurate result because differential operator doesn't depends on grid size

Cons:
  * Only can be applied to periodic domain

# Governing Equation

## Navier-Stokes equation to rotational form
Original Navier-Stokes equation in **convection form** is

\begin{align}
\dfrac{\partial u_i}{\partial t} &= -\dfrac{\nabla p}{\rho} - (u \cdot \nabla) u + \nu \nabla^2 u \newline
\nabla \cdot u &= 0
\end{align}

Using following vector identity,

\begin{align}
\dfrac{1}{2} \nabla (A \cdot A) = (A \cdot \nabla) A + A \times (\nabla \times A)
\end{align}

The Navier-Stoke sequations in **rotational form** can be obatained.

\begin{align}
\dfrac{\partial u_i}{\partial t} &= -\dfrac{\partial P}{\partial x_i} + H_i + \nu \nabla^2 u \newline
\dfrac{\partial u_i}{\partial x_i} &= 0
\end{align}

where

\begin{align}
P &= \dfrac{p}{\rho} + \dfrac{1}{2} u_j u_j \newline
H_i &= \epsilon_{i,j,k} u_j \omega_k = u \times (\nabla \times u)
\end{align}

## Removing pressure term

The pressure Poisson equation can be obatained by taking divergence from Navier-Stokes equation in rotational form

\begin{align}
\nabla^2 P = \dfrac{\partial H_j}{\partial x_j}
\end{align}

Expanding N-S equation and Poisson equation to Fourier space gives

\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -i \kappa_i \hat{P} + \hat{H}_i - \nu \kappa^2 \hat{u}_i \newline
-\kappa^2 \hat{P} &= i \kappa_j \hat{H}_j
\end{align}

Combining two equation and then

\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -i \kappa_i \left( -i \dfrac{\kappa_j}{\kappa^2} \hat{H}_j \right ) + \hat{H}_i - \nu \kappa^2 \hat{u}_i \newline
\dfrac{d \hat{u}_i }{d t} &= -\dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i - \nu \kappa^2 \hat{u}_i
\end{align}

where \\( \kappa \\) is a wavenumber. Final Navier Stokes equation is obtained without pressure term

\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -\dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i - \nu \kappa^2 \hat{u}_i
\end{align}

## Treating viscous term analytically

To treat a viscous terms analytically, multiply following formula to Navier Stokes equation w/o pressure form

\begin{equation}
f(t) = e^{\nu \kappa^2 t}
\end{equation}

Then the equation becomes..

\begin{align}
\left[ \dfrac{d \hat{u}}{dt} + \nu \kappa^2 \hat{u}_j \right] \times f(t) &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] \times f(t) \newline
f(t) \dfrac{d \hat{u}}{dt} + (\nu \kappa^2 f(t))\hat{u}_j &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \newline
f(t) \dfrac{d \hat{u}}{dt} + (\nu \kappa^2 e^{\nu \kappa^2 t})\hat{u}_j  &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \newline
f(t) \dfrac{d \hat{u}}{dt} + (\dfrac{d e^{\nu \kappa^2 t}}{dt})\hat{u}_j  &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \newline
\dfrac{d \hat{u}_i f(t)}{dt} &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t)
\end{align}

this can be more simpler by introducing new term \\(\widehat{NL}\\)

\begin{equation}
\dfrac{d \hat{u}_i e^{\nu \kappa^2 t}}{dt} = \widehat{NL} e^{\nu \kappa^2 t}
\end{equation}

### Time Discretization by RK3 method

For low-storage RK3 method (2-register, 3-stage, 3rd order), the coefficients are given by following table 
{% cite Lundbladh:1999vy --file 2017-07-17-Psuedo-spectral-method  %}, {% cite Yu1992  --file 2017-07-17-Psuedo-spectral-method  %}, {% cite Wray1990 --file 2017-07-17-Psuedo-spectral-method %}


| order   | \\(a_n\\) | \\(b_n\\) | \\(c_n\\) |
|---------|-----------|-----------|-----------|
| 1st     | 8/15      |  0        | 0         |
| 2nd     | 5/12      |-17/60     |8/15       |
| 3rd     | 3/4       |-5/12      |2/3        |


Assume equations are given by following form,

\begin{equation}
\dfrac{\partial Q}{\partial t} = R(Q)
\end{equation}

The low-storage RK3 method applied to the above equation using RK3 coefficients.

\begin{align}
Q^1 &= Q^n + \Delta t \left( \dfrac{8}{15} R^n \right)  \newline
Q^2 &= Q^1 + \Delta t \left( \dfrac{5}{12} R^n  - \dfrac{17}{60} R^1\right) \newline
Q^{n+1} &= Q^2 + \Delta t \left( \dfrac{3}{4} R^n  - \dfrac{5}{12} R^2\right)
\end{align}

Before applying RK3 method to Navier-Stokes equation, apply low-storage RK3 method to reaction-diffusion equation

\begin{align}
\dfrac{\partial \psi}{\partial t} &= G + L \psi \newline
\psi^{n+1} &= \psi^{n}  + a_n \Delta t G^n + b_n \Delta t G^{n-1} + (a_n + b_n) \Delta t \left(\dfrac{L \psi^{n+1} + L \psi^n}{2} \right)
\end{align}

Then the Navier Stokes equation w/o pressure term can be discretized by above method

\\[
  \dfrac{d \hat{u} e^{\nu \kappa^2 t}}{dt} = \widehat{NL} e^{\nu \kappa^2 t}
\\]

Discretization of LHS

\begin{align}
LHS = \dfrac{\hat{u}^{n+1}_i e^{\nu \kappa^2 (t+a_n \Delta t + b_n \Delta t)} - \hat{u}^{n}_i e^{\nu \kappa^2 t}} {\Delta t}
\end{align}

Discretization of RHS  (denoting RHS as \\( G \\))

\begin{align}
RHS &= \widehat{NL}^n e^{\nu \kappa^2 t} \newline
&= a_n\widehat{NL}^n e^{\nu \kappa^2 t} \newline
&+ b_n\widehat{NL}^{n - 1} e^{\nu \kappa^2 (t - a_{n-1} \Delta t - b_{n-1} \Delta t)}
\end{align}

Compensating \\( e^{\nu \kappa^2 t} \\) on both sides

\begin{align}
\hat{u}^{n+1}\_i e^{\nu \kappa^2 (a\_n \Delta t + b\_n \Delta t)} - \hat{u}^{n}\_i &= a\_n \Delta t \widehat{NL}^n \newline \newline
&+ b\_n \Delta t \widehat{NL}^{n - 1} e^{\nu \kappa^2 (- a\_{n-1} \Delta t - b\_{n-1} \Delta t)}
\end{align}

Finally

\begin{align}
\hat{u}^{n+1}\_i &= \left[ a\_n \Delta t \widehat{NL}^n + \hat{u}^{n}\_i \right ] e^{-\nu \kappa^2 (a\_n + b\_n) \Delta t} \newline
& + b\_n \Delta t \widehat{NL}^{n - 1} e^{-\nu \kappa^2 (a\_n + b\_n + a\_{n-1}+ b\_{n-1}) \Delta t}
\end{align}

or

\begin{align}
\hat{u}^{n+1}\_i &= \left[ a\_n \Delta t \widehat{NL}^n + \hat{u}^{n}\_i \right ] e^{-\nu \kappa^2 (c\_n - c\_{n+1}) \Delta t} \newline
& + b\_n \Delta t \widehat{NL}^{n - 1} e^{-\nu \kappa^2 (c\_{n-1} - c\_{n+1}) \Delta t}
\end{align}

# Reference

{% bibliography --file 2017-07-17-Psuedo-spectral-method %}
