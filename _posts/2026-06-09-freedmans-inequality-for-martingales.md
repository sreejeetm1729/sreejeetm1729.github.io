---
layout: post
title: "Freedman's Inequality for Martingales"
date: 2026-06-09
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [probability, concentration, martingales, freedman-inequality, stochastic-approximation, reinforcement-learning]
math: true
---

Freedman's inequality is a martingale concentration inequality that plays the same variance-sensitive role for martingales that Bernstein's inequality plays for independent sums. Azuma-Hoeffding controls martingales using a deterministic bound on the increments. Freedman improves this by also using the predictable conditional variance.

This makes Freedman's inequality especially useful in reinforcement learning and stochastic approximation, where the noise is adapted to the past and its variance can be much smaller than the worst-case increment bound.

## 1. Martingale difference setting

Let $$\{\mathcal{F}_k\}_{k\geq0}$$ be a filtration. Let $$D_1,D_2,\ldots,D_n$$ be a martingale difference sequence satisfying

$$
\mathbb{E}[D_k\mid \mathcal{F}_{k-1}]=0.
$$

Define the martingale

$$
S_k=\sum_{i=1}^kD_i,
qquad S_0=0.
$$

The predictable quadratic variation is

$$
V_k=\sum_{i=1}^k\mathbb{E}[D_i^2\mid \mathcal{F}_{i-1}].
$$

This is the martingale analogue of the variance proxy $$\sum_i\mathbb{E}[X_i^2]$$ in Bernstein's inequality.

## 2. Statement of Freedman's inequality

Assume that

$$
D_k\leq b
\qquad \text{almost surely for every }k.
$$

Then, for every $$t>0$$ and $$v>0$$,

$$
\mathbb{P}\left(
\exists k\leq n:
S_k\geq t \text{ and } V_k\leq v
\right)
\leq
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

A fixed-time consequence is

$$
\mathbb{P}(S_n\geq t \text{ and } V_n\leq v)
\leq
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

If $$|D_k|\leq b$$ almost surely, applying the result to both $$S_k$$ and $$-S_k$$ gives the two-sided form

$$
\mathbb{P}(|S_n|\geq t \text{ and } V_n\leq v)
\leq
2\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

## 3. Comparison with Azuma-Hoeffding

Azuma-Hoeffding says that if $$|D_k|\leq b$$, then

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(-\frac{t^2}{2nb^2}\right).
$$

Freedman says that if the predictable variance satisfies $$V_n\leq v$$, then

$$
\mathbb{P}(S_n\geq t)
\lesssim
\exp\left(-\frac{t^2}{2(v+bt)}\right).
$$

If $$v\ll nb^2$$, Freedman is much sharper.

## 4. Exponential supermartingale ingredient

The proof uses a conditional Bernstein-type mgf bound. Suppose $$D$$ satisfies

$$
\mathbb{E}[D\mid\mathcal{F}]=0,
\qquad
D\leq b.
$$

Then for $$\lambda\in(0,3/b)$$,

$$
\mathbb{E}[e^{\lambda D}\mid\mathcal{F}]
\leq
\exp\left(
\frac{\lambda^2}{2(1-\lambda b/3)}
\mathbb{E}[D^2\mid\mathcal{F}]
\right).
$$

Define

$$
\psi_b(\lambda)=\frac{\lambda^2}{2(1-\lambda b/3)}.
$$

Then

$$
\mathbb{E}[e^{\lambda D}\mid\mathcal{F}]
\leq
\exp\left(\psi_b(\lambda)\mathbb{E}[D^2\mid\mathcal{F}]
\right).
$$

## 5. Constructing the supermartingale

For fixed $$\lambda\in(0,3/b)$$, define

$$
L_k
=
\exp\left(
\lambda S_k-
\psi_b(\lambda)V_k
\right).
$$

We claim $$\{L_k\}$$ is a supermartingale. Indeed,

$$
L_k
=
\exp\left(
\lambda S_{k-1}-\psi_b(\lambda)V_{k-1}
\right)
\exp\left(
\lambda D_k-
\psi_b(\lambda)\mathbb{E}[D_k^2\mid\mathcal{F}_{k-1}]
\right).
$$

Taking conditional expectation given $$\mathcal{F}_{k-1}$$,

$$
\mathbb{E}[L_k\mid\mathcal{F}_{k-1}]
=
L_{k-1}
\mathbb{E}\left[
\exp\left(
\lambda D_k-
\psi_b(\lambda)\mathbb{E}[D_k^2\mid\mathcal{F}_{k-1}]
\right)
\middle|\mathcal{F}_{k-1}
\right].
$$

The variance term is $$\mathcal{F}_{k-1}$$-measurable, so

$$
\mathbb{E}[L_k\mid\mathcal{F}_{k-1}]
=
L_{k-1}
\exp\left(-\psi_b(\lambda)\mathbb{E}[D_k^2\mid\mathcal{F}_{k-1}]\right)
\mathbb{E}[e^{\lambda D_k}\mid\mathcal{F}_{k-1}].
$$

Using the conditional mgf bound,

$$
\mathbb{E}[L_k\mid\mathcal{F}_{k-1}]
\leq
L_{k-1}.
$$

Thus $$\{L_k\}$$ is a nonnegative supermartingale with $$L_0=1$$.

## 6. Maximal inequality step

Because $$L_k$$ is a nonnegative supermartingale, Ville's inequality gives

$$
\mathbb{P}\left(\sup_{k\leq n}L_k\geq a\right)
\leq
\frac{1}{a}.
$$

Take

$$
a=\exp(\lambda t-\psi_b(\lambda)v).
$$

If there exists $$k\leq n$$ such that

$$
S_k\geq t
\quad \text{and} \quad
V_k\leq v,
$$

then

$$
\lambda S_k-\psi_b(\lambda)V_k
\geq
\lambda t-\psi_b(\lambda)v.
$$

Hence

$$
L_k
\geq
\exp(\lambda t-\psi_b(\lambda)v).
$$

Therefore,

$$
\mathbb{P}\left(
\exists k\leq n:S_k\geq t, V_k\leq v
\right)
\leq
\exp(-\lambda t+\psi_b(\lambda)v).
$$

This holds for every $$\lambda\in(0,3/b)$$.

## 7. Optimizing the exponent

We now choose

$$
\lambda=\frac{t}{v+bt/3}.
$$

Then $$\lambda b<3$$ and

$$
1-\frac{\lambda b}{3}
=
\frac{v}{v+bt/3}.
$$

Thus

$$
\psi_b(\lambda)v
=
\frac{\lambda^2}{2(1-\lambda b/3)}v
=
\frac{\lambda^2}{2}\cdot(v+bt/3)
=
\frac{\lambda t}{2}.
$$

Therefore,

$$
-\lambda t+\psi_b(\lambda)v
=
-\frac{\lambda t}{2}
=
-\frac{t^2}{2(v+bt/3)}.
$$

Hence

$$
\mathbb{P}\left(
\exists k\leq n:S_k\geq t, V_k\leq v
\right)
\leq
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

This proves Freedman's inequality.

## 8. High-probability form

A convenient inverted form is: with probability at least $$1-\delta$$, whenever $$V_n\leq v$$,

$$
S_n
\leq
\sqrt{2v\log(1/\delta)}+rac{2b}{3}\log(1/\delta).
$$

For two-sided deviations under $$|D_k|\leq b$$, with probability at least $$1-\delta$$,

$$
|S_n|
\leq
\sqrt{2v\log(2/\delta)}+rac{2b}{3}\log(2/\delta),
$$

provided $$V_n\leq v$$.

More refined empirical versions replace the deterministic upper bound $$v$$ by the observed or predictable variance process itself, often with additional peeling arguments.

## 9. RL applications

Freedman's inequality is useful in RL whenever the noise is a martingale difference and its conditional variance is controlled.

A typical temporal-difference noise term has the form

$$
D_k = \alpha_k \left(g_k-\mathbb{E}[g_k\mid\mathcal{F}_{k-1}]\right),
$$

where $$g_k$$ is a random Bellman-type target. Then

$$
\mathbb{E}[D_k\mid\mathcal{F}_{k-1}]=0.
$$

If $$|D_k|\leq b_k$$ and

$$
\sum_{k=1}^n\mathbb{E}[D_k^2\mid\mathcal{F}_{k-1}]
\leq v,
$$

Freedman gives

$$
\left|\sum_{k=1}^nD_k\right|
\lesssim
\sqrt{v\log(1/\delta)}+igl(\max_k b_k\bigr)\log(1/\delta).
$$

This can be sharper than Azuma-Hoeffding, which would use

$$
\sum_{k=1}^n b_k^2
$$

instead of the conditional variance.

## 10. Freedman versus Bernstein

Bernstein is for independent sums:

$$
S_n=\sum_{i=1}^nX_i,
\qquad
\mathbb{E}[X_i]=0.
$$

Freedman is for martingale sums:

$$
S_n=\sum_{i=1}^nD_i,
\qquad
\mathbb{E}[D_i\mid\mathcal{F}_{i-1}]=0.
$$

The variance proxy changes from

$$
\sum_i\mathbb{E}[X_i^2]
$$

to

$$
\sum_i\mathbb{E}[D_i^2\mid\mathcal{F}_{i-1}].
$$

Thus Freedman's inequality is the martingale Bernstein inequality.

## 11. Summary

Freedman's inequality says that martingales with bounded increments concentrate according to their predictable variance:

$$
\mathbb{P}\left(
\exists k\leq n:S_k\geq t, V_k\leq v
\right)
\leq
\exp\left(-\frac{t^2}{2(v+bt/3)}\right).
$$

It improves Azuma-Hoeffding when the conditional variance is much smaller than the worst-case squared-increment sum. In finite-time RL and stochastic approximation, this distinction can translate directly into sharper sample-complexity bounds.

## References

- D. A. Freedman. "On tail probabilities for martingales." *Annals of Probability*, 1975.
- P. Hall and C. C. Heyde. *Martingale Limit Theory and Its Application*. Academic Press, 1980.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
- M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.
