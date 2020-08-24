---
layout: post
title: (Draft) External Forcing of Homogeneous Isotropic Turbulence
tags:
  - turbulence
  - pseudo-spectral method
  - external forcing
use_math: true
---

(This content is originally written by ‪Kyongmin Yeo's manual)

# Introduction

To maintain stationary turbulence, external force term is added to Navier-Stokes equation
\\[
  \dfrac{d \hat{u}\_i}{dt} = - i \kappa\_{i} \hat{P} + \hat{H}\_i - \nu \kappa^2 \hat{u}\_i + \hat{f}\_i
\\]

where \\(\hat{f}\_i\\) is a external forcing term.

\\(\hat{f}\_i\\) is defined as the projection of a vector \\(\hat{\mathbf{b}}\\) onto the plane normal to the wavenumber vector \\(\mathbf{\kappa}\\) to ensure divergence-free condition.

\\[
  \hat{f}\_{i} = \hat{b}\_{i} - \dfrac{\kappa_i}{\kappa^2} \kappa_j \hat{b}\_{j}
\\]

So, how we define vector \\(\hat{\mathbf{b}}\\)?
Eswaran & Pope suggested stochastic forcing.
They define 3D complex vector \\(\hat{\mathbf{b}}\\) which is non-zero in the range \\(0 < \kappa < \kappa\_f \\), in which \\(\kappa\_f\\) is the maximum forcing wavenumber. This can be interpreted as forcing to sphere in wavenumber space.

\\(\hat{\mathbf{b}}\\) is composed of six independent Uhlenbeck-Ornstein process.

\\[
  \hat{\mathbf{b}} = \begin{bmatrix} UO1 \newline UO3 \newline UO5 \newline \end{bmatrix} + i \begin{bmatrix} UO2 \newline UO4 \newline UO6 \newline \end{bmatrix}
\\]

# Solving Uhlenbeck-Ornstein process

Each stochastic process, \\(UO1 ~ UO6\\), is chosen so as to satisfy hte Langevin equation with a time scale \\(T^f\_L\\) and stadnard deviation \\(\sigma_f\\),

\\[
  dUO = - \dfrac{UO}{T^f\_L} \Delta t + \left( \dfrac{2\sigma^2\_f}{T^f\_L} \right)^{1/2} dW\_t
\\]

in which \\(W_t\\) denotes a Wiener process satisfying
\\[
  dW_t \sim \mathcal{N} (0, \Delta t)
\\]

Analytical solution to Langevin equation is given by 

\\[
  UO(t) = UO(0) e^{-t/T^f\_L} + e^{-t/T^f\_L} \int^t\_{0} e^{s/T^f\_L} (2\sigma^2\_f/T^f\_L)^{1/2}dW_s
\\]

Above solution can be solved discretely by applying Itô integral. With RK3 method, UO process discretized solution is

\\[
  UO^{n+1} = e^{-(a\_n+b\_n)\Delta t / T^f\_L}\left[ UO^{n} + e^{s/T^f\_L} (2\sigma^2\_f/T^f\_L)^{1/2}dW_s dW^n \right]
\\]

in which discretized Wiener proces is

\\[
  dW^n \sim \mathcal{N} (0, (a\_n + b\_n) \Delta t)
\\]

# Estimating Re

## Given parameters
* \\(\nu\\) : Fluid viscosity
* \\(\beta\\) : constant (\\(=0.8\\))
* \\(\kappa\_0\\) : smallest wavenumber
* \\(\kappa\_f\\) : maximum forcing wavenumber
* \\(T_L\\) : Forcing time scale
* \\(\epsilon^\* \equiv \sigma^2 T\_L \\) where \\(\sigma\\) is a forcing amplitude, usually just given by constant

## Assumptions
* \\(\\epsilon \propto N\_f \epsilon^\*\\)
  \begin{align}
    \epsilon &= \dfrac{4\epsilon^\* T_e N\_f}{T_L + T_e} \newline
    T\_e &= \dfrac{\beta}{(N\_f \epsilon^\* \kappa^2\_{0})^{1/3}}
  \end{align}
* \\(\kappa^{-1}\_{0}\\) : Integral length scales

## Computed parameters
* \\(N\_f\\) : The number of forced modes, \\(\kappa < \kappa\_f>\\), counted manually
* Predicted value of the energy dissipation, replace \\(T\_e\\) by \\(\beta / (N\_f \epsilon^\* \kappa^2\_{0})^{1/3}\\)
  \\[
    \epsilon^\*\_{T} \equiv \dfrac{4 \epsilon^\* N\_f}{1 + T_\{L} N^{1/3}\_{F}/\beta}
  \\]
* Predicted Kolmogorov microscale
  \\[
  \eta\_T \equiv (\nu^3 / \epsilon_T)
  \\]

## Predicted Re
Using above parameters Taylor Reynolds number is estimated by 

\\[
  Re \simeq \dfrac{8.5}{(\eta\_T \kappa_0)^{5/6} N^{2/9}\_{F}}
\\]


# Reference
