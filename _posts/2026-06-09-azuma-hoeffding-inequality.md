---
title: "Azuma-Hoeffding Inequality"
date: 2026-06-09
categories: [rl-blogs]
tags: [probability, concentration, martingales, azuma-hoeffding, stochastic-approximation, reinforcement-learning]
math: true
---

Hoeffding's inequality is designed for independent bounded random variables. But in reinforcement learning and stochastic approximation, the relevant noise terms are often not independent. Instead, they are adapted to a filtration and have zero conditional mean. This is exactly the martingale-difference setting.

Azuma-Hoeffding's inequality is the martingale analogue of Hoeffding's inequality. It says that if a martingale has bounded increments, then it concentrates around its starting value with Gaussian-type tails.

This inequality is one of the basic tools behind finite-time analyses of stochastic approximation, temporal-difference learning, and Q-learning.

## 1. Filtrations and martingales

A filtration is an increasing sequence of $$\sigma$$-algebras

$$
\mathcal{F}_0\subseteq \mathcal{F}_1\subseteq \cdots \subseteq \mathcal{F}_n.
$$

Intuitively, $$\mathcal{F}_k$$ represents all information available up to time $$k$$.

A sequence $$\{M_k\}_{k=0}^n$$ is a martingale with respect to $$\{\mathcal{F}_k\}$$ if:

1. $$M_k$$ is $$\mathcal{F}_k$$-measurable;
2. $$\mathbb{E}[|M_k|]<\infty$$;
3. $$\mathbb{E}[M_k\mid \mathcal{F}_{k-1}]=M_{k-1}$$ for every $$k\geq1$$.

Define the martingale difference sequence

$$
D_k=M_k-M_{k-1}.
$$

Then

$$
\mathbb{E}[D_k\mid \mathcal{F}_{k-1}]=0.
$$

This conditional zero-mean property is what replaces independence.

## 2. Statement of Azuma-Hoeffding

Let $$\{M_k\}_{k=0}^n$$ be a martingale with respect to $$\{\mathcal{F}_k\}_{k=0}^n$$. Suppose that for deterministic constants $$c_1,\ldots,c_n$$,

$$
|M_k-M_{k-1}|\leq c_k
\qquad \text{almost surely for every }k=1,\ldots,n.
$$

Then, for every $$t>0$$,

$$
\mathbb{P}(M_n-M_0\geq t)
\leq
\exp\left(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\right).
$$

Similarly,

$$
\mathbb{P}(|M_n-M_0|\geq t)
\leq
2\exp\left(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\right).
$$

The constants differ slightly from Hoeffding's inequality because the bounded increment condition is stated as $$|D_k|\leq c_k$$, so each difference lies in an interval of length $$2c_k$$.

## 3. Conditional Hoeffding lemma

The key step is a conditional version of Hoeffding's lemma.

Suppose $$D_k$$ is $$\mathcal{F}_k$$-measurable, satisfies

$$
\mathbb{E}[D_k\mid \mathcal{F}_{k-1}]=0,
\qquad
|D_k|\leq c_k \quad \text{almost surely}.
$$

Then, for every $$\lambda\in\mathbb{R}$$,

$$
\mathbb{E}\left[e^{\lambda D_k}\mid \mathcal{F}_{k-1}\right]
\leq
\exp\left(\frac{\lambda^2 c_k^2}{2}\right).
$$

This follows by applying Hoeffding's lemma conditionally. Given $$\mathcal{F}_{k-1}$$, the random variable $$D_k$$ has conditional mean zero and lies in $$[-c_k,c_k]$$. The interval length is $$2c_k$$, so Hoeffding's lemma gives

$$
\mathbb{E}\left[e^{\lambda D_k}\mid \mathcal{F}_{k-1}\right]
\leq
\exp\left(\frac{\lambda^2(2c_k)^2}{8}\right)
=
\exp\left(\frac{\lambda^2c_k^2}{2}\right).
$$

## 4. Proof of Azuma-Hoeffding

Let

$$
D_k=M_k-M_{k-1},
\qquad
M_n-M_0=\sum_{k=1}^nD_k.
$$

Fix $$\lambda>0$$. By Markov's inequality,

$$
\mathbb{P}(M_n-M_0\geq t)
\leq
\exp(-\lambda t)
\mathbb{E}\left[\exp\left(\lambda\sum_{k=1}^nD_k\right)\right].
$$

We now control the exponential moment recursively. Using the tower property,

$$
\mathbb{E}\left[e^{\lambda\sum_{k=1}^nD_k}\right]
=
\mathbb{E}\left[
e^{\lambda\sum_{k=1}^{n-1}D_k}
\mathbb{E}\left[e^{\lambda D_n}\mid \mathcal{F}_{n-1}\right]
\right].
$$

By the conditional Hoeffding lemma,

$$
\mathbb{E}\left[e^{\lambda D_n}\mid \mathcal{F}_{n-1}\right]
\leq
\exp\left(\frac{\lambda^2c_n^2}{2}\right).
$$

Thus,

$$
\mathbb{E}\left[e^{\lambda\sum_{k=1}^nD_k}\right]
\leq
\exp\left(\frac{\lambda^2c_n^2}{2}\right)
\mathbb{E}\left[e^{\lambda\sum_{k=1}^{n-1}D_k}\right].
$$

Repeating the argument backward from $$n$$ to $$1$$ yields

$$
\mathbb{E}\left[e^{\lambda\sum_{k=1}^nD_k}\right]
\leq
\exp\left(\frac{\lambda^2}{2}\sum_{k=1}^n c_k^2\right).
$$

Therefore,

$$
\mathbb{P}(M_n-M_0\geq t)
\leq
\exp\left(
-\lambda t+rac{\lambda^2}{2}\sum_{k=1}^n c_k^2
\right).
$$

Let

$$
V_n=\sum_{k=1}^n c_k^2.
$$

The exponent is minimized at

$$
\lambda^*=\frac{t}{V_n}.
$$

Substituting this value gives

$$
-\lambda^*t+rac{(\lambda^*)^2V_n}{2}
=
-\frac{t^2}{V_n}+\frac{t^2}{2V_n}
=
-\frac{t^2}{2V_n}.
$$

Thus,

$$
\mathbb{P}(M_n-M_0\geq t)
\leq
\exp\left(-\frac{t^2}{2\sum_{k=1}^n c_k^2}\right).
$$

The lower-tail bound follows by applying the same result to $$-M_k$$, and the two-sided bound follows by a union bound.

## 5. High-probability form

With probability at least $$1-\delta$$,

$$
M_n-M_0
\leq
\sqrt{2\left(\sum_{k=1}^n c_k^2\right)\log(1/\delta)}.
$$

With probability at least $$1-\delta$$,

$$
|M_n-M_0|
\leq
\sqrt{2\left(\sum_{k=1}^n c_k^2\right)\log(2/\delta)}.
$$

## 6. Weighted martingale sums

A form that appears constantly in stochastic approximation is the weighted sum

$$
Z_t=\sum_{k=0}^{t} w_{k,t}\xi_k,
$$

where $$\{\xi_k\}$$ is a martingale difference sequence and the weights $$w_{k,t}$$ are deterministic or predictable. Suppose

$$
|\xi_k|\leq B
\quad \text{almost surely}.
$$

If the weights are deterministic, then

$$
D_k=w_{k,t}\xi_k
$$

is again a martingale difference sequence and

$$
|D_k|\leq |w_{k,t}|B.
$$

Azuma-Hoeffding gives

$$
\mathbb{P}\left(
|Z_t|\geq u
\right)
\leq
2\exp\left(
-\frac{u^2}{2B^2\sum_{k=0}^{t}w_{k,t}^2}
\right).
$$

Equivalently, with probability at least $$1-\delta$$,

$$
|Z_t|
\leq
B\sqrt{2\left(\sum_{k=0}^{t}w_{k,t}^2\right)\log(2/\delta)}.
$$

In Q-learning proofs, one often encounters weights such as

$$
w_{k,t}=\alpha(1-\alpha\lambda)^{t-k},
$$

where $$\alpha$$ is the stepsize and $$\lambda$$ is a visitation probability. Then

$$
\sum_{k=0}^{t}w_{k,t}^2
=
\alpha^2\sum_{k=0}^{t}(1-\alpha\lambda)^{2(t-k)}.
$$

Changing variables $$r=t-k$$,

$$
\sum_{k=0}^{t}w_{k,t}^2
=
\alpha^2\sum_{r=0}^{t}(1-\alpha\lambda)^{2r}
\leq
\alpha^2\sum_{r=0}^{\infty}(1-\alpha\lambda)^r
=
\frac{\alpha^2}{\alpha\lambda}
=
\frac{\alpha}{\lambda}.
$$

Hence

$$
|Z_t|
\lesssim
B\sqrt{\frac{\alpha}{\lambda}\log(1/\delta)}.
$$

This is a canonical stochastic-approximation noise bound.

## 7. Union bounds over time and coordinates

Suppose we want a martingale concentration event to hold for every time $$t\leq T$$ and every coordinate $$j\in[d]$$. A standard approach is to assign failure probability

$$
\delta' = \frac{\delta}{dT}
$$

to each coordinate-time pair. Then each individual event holds with probability at least $$1-\delta'$$, and the union bound gives simultaneous validity with probability at least $$1-\delta$$.

This is why RL finite-time bounds often contain logarithmic factors of the form

$$
\log\frac{|\mathcal{S}||\mathcal{A}|T}{\delta}.
$$

The $$|\mathcal{S}||\mathcal{A}|$$ term comes from coordinates. The $$T$$ term comes from uniformity over time. The $$\delta$$ term comes from high-probability control.

## 8. Limitations

Azuma-Hoeffding is robust and convenient, but it can be loose.

First, it uses the deterministic bound $$c_k$$ and ignores conditional variance. Freedman's inequality improves this by incorporating the predictable quadratic variation.

Second, if the deterministic bound $$c_k$$ is very large but the increment is usually much smaller, standard Azuma-Hoeffding becomes vacuous. This issue appears in analyses where an iterate is deterministically bounded by a crude quantity but is much better behaved on a high-probability event. A refined high-probability Azuma-Hoeffding inequality is useful in such cases.

Third, Azuma-Hoeffding does not handle heavy-tailed martingale differences without truncation, robustification, or additional assumptions.

## References

- K. Azuma. "Weighted sums of certain dependent random variables." *Tohoku Mathematical Journal*, 1967.
- W. Hoeffding. "Probability inequalities for sums of bounded random variables." *Journal of the American Statistical Association*, 1963.
- D. Williams. *Probability with Martingales*. Cambridge University Press, 1991.
- M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.
