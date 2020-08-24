---
layout: post
title: Equation of Pseudo-spectral method for Homogeneous Isotropic Turbulence
tags:
  - turbulence
  - pseudo-spectral method
use_math: true
---

# Introduction

The spectral method is solving certain differential equation by some "basis function", typically sinusoids with Fourier method
With the Navier-Stokes equation, it can remove presssure term in N-S equation and solve viscous term analytically.  This is huge because solving Poisson equation (pressure term) usually takes very long time compared to other terms. Moreover, I can expect more accuracy  than simple numericla discretization. However, I can apply spectral method only to periodic domain, but it doesn't matter when solving isotropic turbulence.

# Governing Equtation

## Navier-Stokes equation to rotational form
Original Navier-Stokes equation in convection form is

$$
\begin{align}
\dfrac{\partial u_i}{\partial t} &= -\dfrac{\nabla p}{\rho} - (u \cdot \nabla) u + \nu \nabla^2 u \\
\nabla \cdot u &= 0
\end{align}
$$

Using following vector identical solution,

$$
\begin{align}
\dfrac{1}{2} \nabla (A \cdot A) = (A \cdot \nabla) A + A \times (\nabla \times A)
\end{align}
$$

I can convert original equation to rotational form.

$$
\begin{align}
\dfrac{\partial u_i}{\partial t} &= -\dfrac{\partial P}{\partial x_i} + H_i + \nu \nabla^2 u \\
\dfrac{\partial u_i}{\partial x_i} &= 0
\end{align}
$$

where

$$
\begin{align}
P &= \dfrac{p}{\rho} + \dfrac{1}{2} u_j u_j \\
H_i &= \epsilon_{i,j,k} u_j \omega_k = u \times (\nabla \times u)
\end{align}
$$

## Removing pressure term

If I take divergence from Navier-Stokes equation in rotational form,

$$
\begin{align}
\nabla^2 P = \dfrac{\partial H_j}{\partial x_j}
\end{align}
$$

By expanding both equation (N-S equation & Poisson equation) to Fourier space, I got following equation

$$
\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -i \kappa_i \hat{P} + \hat{H}_i - \nu \kappa^2 \hat{u}_i \\\
-\kappa^2 \hat{P} &= i \kappa_j \hat{H}_j
\end{align}
$$

Combining two equation and then..

$$
\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -i \kappa_i \left( -i \dfrac{\kappa_j}{\kappa^2} \hat{H}_j \right ) + \hat{H}_i - \nu \kappa^2 \hat{u}_i \\\
\dfrac{d \hat{u}_i }{d t} &= -\dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i - \nu \kappa^2 \hat{u}_i
\end{align}
$$

where $\kappa$ is a wavenumber. Now I have new N-S equation without pressure term

$$
\begin{align}
\dfrac{d \hat{u}_i }{d t} &= -\dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i - \nu \kappa^2 \hat{u}_i
\end{align}
$$

## Treating viscous term analytically

To treat a viscous terms analytically, I just multiply following formula to N-S equation without pressure form

$$
\begin{equation}
f(t) = e^{\nu \kappa^2 t}
\end{equation}
$$

Then the equation changed like this

$$
\begin{align}
\left[ \dfrac{d \hat{u}}{dt} + \nu \kappa^2 \hat{u}_j \right] \times f(t) &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] \times f(t) \\\
f(t) \dfrac{d \hat{u}}{dt} + (\nu \kappa^2 f(t))\hat{u}_j &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \\\
f(t) \dfrac{d \hat{u}}{dt} + (\nu \kappa^2 e^{\nu \kappa^2 t})\hat{u}_j  &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \\\
f(t) \dfrac{d \hat{u}}{dt} + (\dfrac{d e^{\nu \kappa^2 t}}{dt})\hat{u}_j  &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t) \\\
\dfrac{d \hat{u}_i f(t)}{dt} &= \left[ - \dfrac{\kappa_i \kappa_j}{\kappa^2} \hat{H}_j + \hat{H}_i \right ] f(t)
\end{align}
$$

this can be more simpler by introducing new term $\widehat{NL}$

$$
\begin{equation}
\dfrac{d \hat{u}_i e^{\nu \kappa^2 t}}{dt} = \widehat{NL} e^{\nu \kappa^2 t}
\end{equation}
$$

### Time Discretization by RK3 method

For low-storage RK3 method (2-register, 3-stage, 3rd order), the coefficients are following table


| order   |   $a_n$   |   $b_n$   |   $c_n$   |
|---------|-----------|-----------|-----------|
| 1st     | 8/15      |  0        | 0         |
| 2nd     | 5/12      |-17/60     |8/15       |
| 3rd     | 3/4       |-5/12      |2/3        |


If I assume to solve following equation,

$$
\begin{equation}
\dfrac{\partial Q}{\partial t} = R(Q)
\end{equation}
$$

The low-storage RK3 method applied to the equation using above coefficients.

$$
\begin{align}
Q^1 &= Q^n + \Delta t \left( \dfrac{8}{15} R^n \right)  \\\
Q^2 &= Q^1 + \Delta t \left( \dfrac{5}{12} R^n  - \dfrac{17}{60} R^1\right) \\\
Q^{n+1} &= Q^2 + \Delta t \left( \dfrac{3}{4} R^n  - \dfrac{5}{12} R^2\right)
\end{align}
$$

Before Navier-Stokes equation, I can apply low-storage RK3 method to reaction-diffusion equation

$$
\begin{align}
\dfrac{\partial \psi}{\partial t} &= G + L \psi \\\
\psi^{n+1} &= \psi^{n}  + a_n \Delta t G^n + b_n \Delta t G^{n-1} + (a_n + b_n)\Delta t\left(\dfrac{L \psi^{n+1} + L \psi^n}{2} \right)
\end{align}
$$

Then the N-S equation should be..

$$
\begin{align}
\dfrac{d \hat{u} e^{\nu \kappa^2 t}}{dt} &= \widehat{NL} e^{\nu \kappa^2 t} \\\
\dfrac{\hat{u}^{n+1}_i e^{\nu \kappa^2 (t+a_n \Delta t + b_n \Delta t)} - \hat{u}^{n}_i e^{\nu \kappa^2 t}}{\Delta t} &= a_n\widehat{NL}^n e^{\nu \kappa^2 t} + b_n\widehat{NL}^{n - 1} e^{\nu \kappa^2 (t - a_{n-1} \Delta t - b_{n-1} \Delta t)} \\\
\hat{u}^{n+1}_i e^{\nu \kappa^2 (a_n \Delta t + b_n \Delta t)} - \hat{u}^{n}_i &= a_n \Delta t \widehat{NL}^n + b_n \Delta t \widehat{NL}^{n - 1} e^{\nu \kappa^2 (- a_{n-1} \Delta t - b_{n-1} \Delta t)} \\\
\hat{u}^{n+1}_i  &= \left[a_n \Delta t \widehat{NL}^n + \hat{u}^{n}_i \right ] e^{-\nu \kappa^2 (a_n + b_n) \Delta t}  + b_n \Delta t \widehat{NL}^{n - 1} e^{\nu \kappa^2 -(a_n + b_n + a_{n-1}+ b_{n-1}) \Delta t} \\\
\end{align}
$$

# Reference

## Navier Stokes Equation
* Chevalier, M. et al., 2007, SIMSON - A Pseudo-Spectral Solver for Incompressible Boundary Layer Flows https://www.mech.kth.se/mech/info_tritamek.jsp?TritaMekID=315
* Canuto, Claudio, et al. Spectral methods in fluid dynamics. Springer Science & Business Media, 2012.

## RK3 method
* Lundbladh, Anders, et al. "An efficient spectral method for simulation of incompressible flow over a flat plate." Trita-mek. Tech. Rep 11 (1999).
* Wray, A. A. "Minimal storage time advancement schemes for spectral methods." NASA Ames Research Center, California, Report No. MS 202 (1990).
* Yu, Sheng-Tao. "Runge-Kutta methods combined with compact difference schemes for the unsteady Euler equations." (1992).
