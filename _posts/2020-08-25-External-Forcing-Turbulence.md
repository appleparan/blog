---
layout: post
title: External Forcing of Homogeneous Isotropic Turbulence
author: jongsukim
date: 2020-08-25 12:00:00 +0900
categories: [Science]
tags:
  - turbulence
  - pseudo-spectral method
  - external forcing$$
math: true
---

(This content is originally written by [Kyongmin Yeo](https://scholar.google.com/citations?user=8fMRupoAAAAJ&hl=ko)'s manual)

## Introduction

The small scale statistics of turbulence are important research topic. 

{% quote eswaran_examination_1988 --file 2020-08-25-External-Forcing %}
Small-scale behavior in turbulent flows tends to be characterized by statistical homoegenity, isotropy, and universality. Because of this universality we can hope to
understand small-scale behavior by studying the simplest turbulent flows, i.e. homoegeneous, isotropic turbulence.
{% endquote %} 

To maintain statistically stationary turbulence, adding force to low wavenumber (large scale) velocity components artificially. Therfore, external force term is added to Navier-Stokes equation

$$
  \dfrac{d \hat{u}_i}{dt} = - i \kappa_{i} \hat{P} + \hat{H}_i - \nu \kappa^2 \hat{u}_i + \hat{f}_i
$$

where \\(\hat{f}_i\\) is a external forcing term.

The forcing \\(\hat{f}_i\\) is applied to circle of low wavenumber band. \\(\hat{f}_i\\) is defined as the projection of a vector \\(\hat{\mathbf{b}}\\) onto the plane normal to the wavenumber vector \\(\mathbf{\kappa}\\) to ensure divergence-free condition.

$$
  \hat{f}_{i} = \hat{b}_{i} - \dfrac{\kappa_i}{\kappa^2} \kappa_j \hat{b}_{j}
$$

So, how we define vector \\(\hat{\mathbf{b}}\\)?
Eswaran & Pope suggested stochastic forcing {% cite eswaran_examination_1988 --file 2020-08-25-External-Forcing %}.
They define 3D complex vector \\(\hat{\mathbf{b}}\\) which is non-zero in the range \\(0 < \kappa < \kappa_f \\), in which \\(\kappa_f\\) is the maximum forcing wavenumber. This can be interpreted as forcing to sphere in wavenumber space.

They used Uhlenbeck-Ornstein process to generate \\(\hat{\mathbf{b}} = \hat{b} (\kappa, t) \\) with following properties, the average and the correlation. 

$$
\begin{align}
  \langle \hat{b} (\kappa, t) \rangle &= 0 \\
  \langle \hat{b} (\kappa, t) \hat{b}^* (\kappa, t + s) \rangle &=  2\sigma^2 \delta_{ij} \exp{(-s/T_L)}
\end{align}
$$

where an asterisk dentoes a complex conjugate, angle bracket is the ensemble average, \\( \delta_{ij} \\) is the Kronecker delta. \\( \sigma^2 \\) and \\( T_L \\) are the variance and time-scale of UO process. Obviously, if \\( T_L \\) increases with fixed \\(\sigma \\), the correlation will converge to zero. This is by no means the desired result, so \\( \epsilon^* \equiv \sigma^2 T_L \\) is fixed.

The three-dimensional vector \\(\hat{\mathbf{b}}\\) is composed of six independent Uhlenbeck-Ornstein process.

$$
  \hat{\mathbf{b}} = \begin{bmatrix} UO1 \\ UO3 \\ UO5 \\ \end{bmatrix} + i \begin{bmatrix} UO2 \\ UO4 \\ UO6 \\ \end{bmatrix}
$$

## Solving Uhlenbeck-Ornstein process

Each stochastic process, \\(UO1 \\) ~ \\( UO6\\), is chosen so as to satisfy the [Langevin equation](https://en.wikipedia.org/wiki/Langevin_equation) with a time scale \\(T^f_L\\) and stadnard deviation \\(\sigma_f\\). 

In {% cite wojnowicz_ornstein-uhlenbeck_2012 --file 2020-08-25-External-Forcing %}, UO process are defined as

$$
dx_t = \dfrac{(\mu - x_t)}{\tau} dt + \sqrt{\dfrac{2\nu}{\tau}} dW_t
$$

After applying zero mean property of forcing term and adjusting parameters makes above eqaution to

$$
  dUO = - \dfrac{UO}{T^f_L} \Delta t + \left( \dfrac{2\sigma^2_f}{T^f_L} \right)^{1/2} dW_t
$$

in which \\(W_t\\) denotes a Wiener process satisfying

$$
  dW_t \sim \mathcal{N} (0, \Delta t)
$$

[The analytical solution of the Langevin equation](http://physics.gu.se/~frtbm/joomla/media/mydocs/LennartSjogren/kap6.pdf) is given by following equation, which describes the Browninan motion of particle.

$$
\begin{align}
x(t) &= x_0 + \int_0^t v(s) ds \\
v(t) &= e^{-t/T^f_L} v_0 + \dfrac{1}{m} \int^t_0 e^{-(t-s)/T^f_L} dW(s)
\end{align}
$$

where \\(x_0 = x(0)\\) and \\(v_0 = v(0) \\). The forcing \\(\hat{f}_i\\) term can be viewed as forcing acclereration, then \\(UO \\) can be denoted to \\(v(t)\\).

$$
  UO(t) = UO(0) e^{-t/T^f_L} + e^{-t/T^f_L} \int^t_{0} e^{s/T^f_L} (2\sigma^2_f/T^f_L)^{1/2}dW_s
$$

Above solution can be solved discretely by applying Itô integral. With RK3 method, UO process discretized solution is

$$
  UO^{n+1} = e^{-(a_n+b_n)\Delta t / T^f_L}\left[ UO^{n} + e^{s/T^f_L} (2\sigma^2_f/T^f_L)^{1/2}dW_s dW^n \right]
$$

in which discretized Wiener process is

$$
  dW^n \sim \mathcal{N} (0, (a_n + b_n) \Delta t)
$$

This is the extension of [Euler-Maruyama method](https://en.wikipedia.org/wiki/Euler%E2%80%93Maruyama_method).

## Estimating Reynolds Number

### Input parameters

The input parameters are \\( \kappa_0 \\) (the lowest wavenumber), \\( \kappa_\textrm{max} \\) (the highest wavenumber), \\( K_F \\) (the maximum wavenumber of the forced modes), \\( \nu \\) (the kinematic viscosity), \\( T_L \\) (the forcing time scale, time scale in UO process), and \\( \epsilon^* = \sigma^2 T_L \\).

The nondimensional parameters are \\(\kappa_{\textrm{max}} / \kappa_0 \\), \\( K_{F} / \kappa_0\\),

$$
\begin{align}
Re^* &\equiv \epsilon^* \kappa_0^{-4/3} / \nu \\
T^*_L &\equiv T_L {\epsilon^{*}}^{1/3} \kappa_0^{2/3}
\end{align}
$$

### Given parameters
* \\( \nu \\) : Fluid viscosity
* \\( \beta \\) : constant (\\( \beta=0.8 \\))
* \\( \kappa_0 \\) : smallest wavenumber
* \\( \kappa_f \\) : maximum forcing wavenumber
* \\( T_L \\) : Forcing time scale
* \\( \epsilon^* \equiv \sigma^2 T_L \\) where \\( \sigma \\) is a forcing amplitude, usually just given by constant

### Assumptions
* \\( \\epsilon \propto N_f \epsilon^* \\) 
* \\( T_e \approx \dfrac{\beta}{(N_f \epsilon^* \kappa^{2}_0)^{1/3}}\\) (posteriori assumption)
* \\( \kappa^{-1}_{0}\\) : Integral length scales


### Computed parameters

* \\( N_f \\) : The number of forced modes, \\( \kappa < \kappa_f \\), counted manually
* Predicted energy dissipation, 

  $$
    \begin{align}
      T_e &= \dfrac{\beta}{(N_f \epsilon^* \kappa^2_{0})^{1/3}} \\
      T^*_L &\equiv T_L {\epsilon^{*}}^{1/3} \kappa_0^{2/3} \\
      \epsilon^*_{T} &= \epsilon \\
          &\equiv \dfrac{4\epsilon^* T_e N_f}{T_L + T_e} \\
          &= \dfrac{4 \epsilon^* N_f}{1 + T^*_{L} N^{1/3}_{F}/\beta}
    \end{align}
  $$

* Predicted Kolmogorov microscale
  $$
  \eta_{T} \equiv (\nu^3 / \epsilon^*_T)
  $$

### Predicted \\( Re \\)
Using above parameters Taylor Reynolds number is estimated by 

$$
  Re \simeq \dfrac{8.5}{(\eta_{T} \kappa_0)^{5/6} N^{2/9}_{F}}
$$

# References

{% bibliography --cited --file 2020-08-25-External-Forcing %}
