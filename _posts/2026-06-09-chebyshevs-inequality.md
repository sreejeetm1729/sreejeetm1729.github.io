---
title: "Chebyshev's Inequality"
date: 2026-06-09
categories: [rl-blogs]
tags: [probability, concentration, chebyshev-inequality, variance, heavy-tails, reinforcement-learning]
math: true
---

Chebyshev's inequality is the most important concentration inequality when one only has finite variance. It is weaker than Hoeffding, Bernstein, Chernoff, Azuma-Hoeffding, and Freedman in light-tailed settings, but it applies under much weaker assumptions. This makes it fundamental in heavy-tailed learning problems.

In reinforcement learning, Chebyshev's inequality is often the first tool one reaches for when rewards are allowed to have only a finite second moment. It gives polynomial concentration rather than exponential concentration, but it is assumption-light and explains why robust mean estimators are needed when exponential tails are unavailable.

## 1. Statement

Let $$X$$ be a random variable with finite mean

$$
\mu=\mathbb{E}[X]
$$

and finite variance

$$
\operatorname{Var}(X)=\mathbb{E}[(X-\mu)^2]=\sigma^2<\infty.
$$

Then, for every $$t>0$$,

$$
\mathbb{P}(|X-\mu|\geq t)
\leq
\frac{\sigma^2}{t^2}.
$$

Equivalently, for every $$k>0$$,

$$
\mathbb{P}(|X-\mu|\geq k\sigma)
\leq
\frac{1}{k^2}.
$$

## 2. Proof

The proof is Markov's inequality applied to the nonnegative random variable $$(X-\mu)^2$$.

Observe that

$$
|X-\mu|\geq t
\quad \Longleftrightarrow \quad
(X-\mu)^2\geq t^2.
$$

Therefore,

$$
\mathbb{P}(|X-\mu|\geq t)
=
\mathbb{P}((X-\mu)^2\geq t^2).
$$

By Markov's inequality,

$$
\mathbb{P}((X-\mu)^2\geq t^2)
\leq
\frac{\mathbb{E}[(X-\mu)^2]}{t^2}
=
\frac{\sigma^2}{t^2}.
$$

Thus,

$$
\mathbb{P}(|X-\mu|\geq t)
\leq
\frac{\sigma^2}{t^2}.
$$

## 3. High-probability form

Set

$$
\frac{\sigma^2}{t^2}=\delta.
$$

Solving for $$t$$ gives

$$
t=\frac{\sigma}{\sqrt\delta}.
$$

Thus, with probability at least $$1-\delta$$,

$$
|X-\mu|
\leq
\frac{\sigma}{\sqrt\delta}.
$$

This is much weaker than sub-Gaussian concentration, where the typical dependence would be $$\sqrt{\log(1/\delta)}$$ rather than $$1/\sqrt\delta$$. But Chebyshev requires only finite variance.

## 4. Chebyshev for sample means

Let $$X_1,\ldots,X_n$$ be independent random variables with common mean $$\mu$$ and variance at most $$\sigma^2$$. Define

$$
\overline{X}_n=\frac{1}{n}\sum_{i=1}^nX_i.
$$

Then

$$
\mathbb{E}[\overline{X}_n]=\mu.
$$

Also, by independence,

$$
\operatorname{Var}(\overline{X}_n)
=
\operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^nX_i\right)
=
\frac{1}{n^2}\sum_{i=1}^n\operatorname{Var}(X_i)
\leq
\frac{\sigma^2}{n}.
$$

Applying Chebyshev's inequality to $$\overline{X}_n$$ gives

$$
\mathbb{P}(|\overline{X}_n-\mu|\geq t)
\leq
\frac{\sigma^2}{nt^2}.
$$

Equivalently, with probability at least $$1-\delta$$,

$$
|\overline{X}_n-\mu|
\leq
\frac{\sigma}{\sqrt{n\delta}}.
$$

The rate in $$n$$ is still $$1/\sqrt n$$, but the confidence dependence is poor.

## 5. Why this is not enough for sharp learning rates

For comparison, if the variables were bounded or sub-Gaussian, we could obtain

$$
|\overline{X}_n-\mu|
\lesssim
\sigma\sqrt{\frac{\log(1/\delta)}{n}}
$$

with probability at least $$1-\delta$$. Chebyshev gives

$$
|\overline{X}_n-\mu|
\lesssim
\sigma\sqrt{\frac{1}{n\delta}}.
$$

When $$\delta$$ is small, the difference is enormous. For example, if $$\delta=10^{-6}$$, then

$$
\sqrt{\log(1/\delta)}\approx 3.7,
\qquad
\frac{1}{\sqrt\delta}=1000.
$$

This is why ordinary sample means are not ideal for high-probability heavy-tailed estimation. One usually replaces them with robust estimators such as median-of-means, Catoni estimators, or trimmed means.

## 6. One-sided Chebyshev: Cantelli's inequality

A useful refinement is Cantelli's inequality. If $$X$$ has mean $$\mu$$ and variance $$\sigma^2$$, then for every $$t>0$$,

$$
\mathbb{P}(X-\mu\geq t)
\leq
\frac{\sigma^2}{\sigma^2+t^2}.
$$

This is stronger than the one-sided consequence of Chebyshev, which would give $$\sigma^2/t^2$$.

### Proof

For any $$a>0$$,

$$
\mathbb{P}(X-\mu\geq t)
=
\mathbb{P}(X-\mu+a\geq t+a).
$$

Since $$X-\mu\geq t$$ implies $$X-\mu+a\geq t+a$$, and the latter quantity is nonnegative on the event, we have

$$
\mathbb{P}(X-\mu\geq t)
\leq
\mathbb{P}((X-\mu+a)^2\geq (t+a)^2).
$$

By Markov's inequality,

$$
\mathbb{P}((X-\mu+a)^2\geq (t+a)^2)
\leq
\frac{\mathbb{E}[(X-\mu+a)^2]}{(t+a)^2}.
$$

Now

$$
\mathbb{E}[(X-\mu+a)^2]
=
\mathbb{E}[(X-\mu)^2]+2a\mathbb{E}[X-\mu]+a^2
=
\sigma^2+a^2.
$$

Thus,

$$
\mathbb{P}(X-\mu\geq t)
\leq
\frac{\sigma^2+a^2}{(t+a)^2}.
$$

Minimize over $$a>0$$. The optimal choice is

$$
a=\frac{\sigma^2}{t}.
$$

Substituting gives

$$
\frac{\sigma^2+\sigma^4/t^2}{(t+\sigma^2/t)^2}
=
\frac{\sigma^2}{\sigma^2+t^2}.
$$

Therefore,

$$
\mathbb{P}(X-\mu\geq t)
\leq
\frac{\sigma^2}{\sigma^2+t^2}.
$$

## 7. Chebyshev and heavy-tailed rewards

Suppose rewards satisfy only

$$
\mathbb{E}[r(s,a)]=R(s,a),
\qquad
\operatorname{Var}(r(s,a))\leq \sigma^2.
$$

No boundedness or sub-Gaussian assumption is imposed. For $$n$$ independent samples from the reward distribution at $$(s,a)$$, Chebyshev gives

$$
\mathbb{P}\left(
|\widehat R(s,a)-R(s,a)|\geq t
\right)
\leq
\frac{\sigma^2}{nt^2}.
$$

This is enough to prove consistency of the sample mean, but it is not enough to get sharp high-probability rates uniformly over many state-action pairs and time indices. If we union-bound over $$|\mathcal{S}||\mathcal{A}|T$$ events, the confidence level per event becomes very small, and Chebyshev's $$1/\sqrt\delta$$ dependence becomes too expensive.

This is the basic reason robust mean estimation is important in heavy-tailed RL. Robust estimators recover sub-Gaussian-like confidence dependence under only finite variance, often giving bounds of the form

$$
|\widehat R(s,a)-R(s,a)|
\lesssim
\sigma\sqrt{\frac{\log(1/\delta)}{n}}
$$

under finite-variance assumptions, up to constants and estimator-specific conditions.

## 8. Relationship to other inequalities

Chebyshev is weaker than Hoeffding and Bernstein when boundedness holds. It is weaker than Chernoff when the variables are Bernoulli or bounded in a multiplicative sense. It is weaker than Azuma-Hoeffding for bounded martingale differences. It is weaker than Freedman when conditional variances and bounded increments are available.

But Chebyshev is more general than all of them in one important sense: it requires only a second moment.

The hierarchy is roughly:

$$
\text{finite variance}
\Rightarrow
\text{Chebyshev},
$$

while stronger assumptions such as boundedness, sub-Gaussianity, or bounded martingale increments lead to exponential concentration.

## 9. Summary

Chebyshev's inequality says that variance controls deviations:

$$
\mathbb{P}(|X-\mathbb{E}X|\geq t)
\leq
\frac{\operatorname{Var}(X)}{t^2}.
$$

For sample means,

$$
|\overline X_n-\mu|
=O\left(\frac{\sigma}{\sqrt{n\delta}}\right)
$$

with probability at least $$1-\delta$$.

This is not sharp enough for many high-probability learning bounds, but it is the starting point for heavy-tailed analysis and robust estimation.

## References

- P. Billingsley. *Probability and Measure*. Wiley, 1995.
- R. Durrett. *Probability: Theory and Examples*. Cambridge University Press, 2019.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
