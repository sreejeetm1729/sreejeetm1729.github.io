---
layout: post
title: "Bernstein's Inequality"
date: 2026-06-09
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [probability, concentration, bernstein-inequality, variance-sensitive-bounds, reinforcement-learning]
math: true
---

Hoeffding's inequality controls sums of bounded independent random variables using only their ranges. Bernstein's inequality is sharper because it also uses the variance. This matters in reinforcement learning because many random quantities are bounded but not worst-case noisy at every step. Visit-count processes, bounded temporal-difference noise, empirical transition estimates, and Bernoulli corruption indicators often have variance much smaller than their crude deterministic range.

Bernstein's inequality is the correct tool when the proof needs both:

1. a sub-Gaussian term for moderate deviations;
2. a sub-exponential term for very large deviations.

The resulting bound has the characteristic form

$$
\exp\left(-\frac{t^2}{2(v+bt/3)}\right),
$$

where $$v$$ is a variance proxy and $$b$$ is a uniform bound on the summands.

## 1. The setting

Let $$X_1,\ldots,X_n$$ be independent random variables satisfying

$$
\mathbb{E}[X_i]=0,
\qquad
|X_i|\leq b \quad \text{almost surely}.
$$

Define

$$
S_n = \sum_{i=1}^n X_i,
\qquad
v = \sum_{i=1}^n \mathbb{E}[X_i^2].
$$

Then Bernstein's inequality states that for every $$t>0$$,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

The two-sided version is

$$
\mathbb{P}(|S_n|\geq t)
\leq
2\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

## 2. Interpretation

The denominator has two pieces:

$$
v+\frac{bt}{3}.
$$

When $$t$$ is not too large relative to $$v/b$$, the variance term dominates and the exponent behaves like

$$
-\frac{t^2}{2v}.
$$

This is sub-Gaussian behavior.

When $$t$$ is very large, the linear term $$bt$$ dominates and the exponent behaves like

$$
-\frac{3t}{2b}.
$$

This is sub-exponential behavior. The transition between the two regimes is exactly what makes Bernstein sharper than Hoeffding in variance-sensitive analyses.

## 3. A useful exponential moment bound

The proof rests on the following elementary inequality. For every real $$x$$ satisfying $$|x|<3$$,

$$
e^x \leq 1+x+\frac{x^2}{2(1-|x|/3)}.
$$

Applying this to $$x=\lambda X_i$$, with $$0<\lambda<3/b$$, gives

$$
e^{\lambda X_i}
\leq
1+\lambda X_i+
\frac{\lambda^2X_i^2}{2(1-\lambda b/3)}.
$$

Taking expectations and using $$\mathbb{E}[X_i]=0$$,

$$
\mathbb{E}[e^{\lambda X_i}]
\leq
1+
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}.
$$

Since $$1+u\leq e^u$$,

$$
\mathbb{E}[e^{\lambda X_i}]
\leq
\exp\left(
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}
\right).
$$

This is the Bernstein moment-generating-function estimate.

## 4. Proof of Bernstein's inequality

Fix $$\lambda\in(0,3/b)$$. By Markov's inequality,

$$
\mathbb{P}(S_n\geq t)
=
\mathbb{P}(e^{\lambda S_n}\geq e^{\lambda t})
\leq
e^{-\lambda t}\mathbb{E}[e^{\lambda S_n}].
$$

By independence,

$$
\mathbb{E}[e^{\lambda S_n}]
=
\prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}].
$$

Using the moment bound from the previous section,

$$
\mathbb{E}[e^{\lambda S_n}]
\leq
\prod_{i=1}^n
\exp\left(
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}
\right).
$$

Therefore,

$$
\mathbb{E}[e^{\lambda S_n}]
\leq
\exp\left(
\frac{\lambda^2}{2(1-\lambda b/3)}
\sum_{i=1}^n\mathbb{E}[X_i^2]
\right)
=
\exp\left(
\frac{\lambda^2v}{2(1-\lambda b/3)}
\right).
$$

Hence,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(
-\lambda t+
\frac{\lambda^2v}{2(1-\lambda b/3)}
\right).
$$

Now choose

$$
\lambda = \frac{t}{v+bt/3}.
$$

This choice satisfies $$\lambda b<3$$. Indeed,

$$
\lambda b
=
\frac{bt}{v+bt/3}
<3.
$$

Next observe that

$$
1-\frac{\lambda b}{3}
=
1-\frac{bt}{3v+bt}
=
\frac{3v}{3v+bt}
=
\frac{v}{v+bt/3}.
$$

Thus,

$$
\frac{\lambda^2v}{2(1-\lambda b/3)}
=
\frac{1}{2}\lambda^2v\cdot \frac{v+bt/3}{v}
=
\frac{1}{2}\lambda^2(v+bt/3)
=
\frac{1}{2}\lambda t.
$$

Therefore the exponent becomes

$$
-\lambda t+rac{1}{2}\lambda t
=
-\frac{1}{2}\lambda t
=
-\frac{t^2}{2(v+bt/3)}.
$$

This proves

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

The lower-tail bound follows by applying the same argument to $$-S_n$$. The two-sided bound follows by a union bound.

## 5. High-probability form

Bernstein's inequality can be inverted. With probability at least $$1-\delta$$,

$$
S_n
\leq
\sqrt{2v\log(1/\delta)}+rac{2b}{3}\log(1/\delta).
$$

A similar two-sided statement is: with probability at least $$1-\delta$$,

$$
|S_n|
\leq
\sqrt{2v\log(2/\delta)}+rac{2b}{3}\log(2/\delta).
$$

The precise constants depend on the inversion used, but the important structure is

$$
|S_n|
\lesssim
\sqrt{v\log(1/\delta)}+b\log(1/\delta).
$$

This separates the variance-sensitive term from the worst-case boundedness term.

## 6. Bernstein for Bernoulli visit counts

A common use in asynchronous RL is to control the number of visits to a state-action pair. Suppose

$$
I_k(s,a)=\mathbf{1}\{(s_k,a_k)=(s,a)\},
$$

and under an i.i.d. sampling model,

$$
\mathbb{P}((s_k,a_k)=(s,a))=\lambda(s,a).
$$

Define

$$
N_t(s,a)=\sum_{k=1}^t I_k(s,a).
$$

Then

$$
\mathbb{E}[N_t(s,a)]=\lambda(s,a)t.
$$

Set

$$
X_k=I_k(s,a)-\lambda(s,a).
$$

Then $$\mathbb{E}[X_k]=0$$ and $$|X_k|\leq 1$$. Also,

$$
\sum_{k=1}^t\mathbb{E}[X_k^2]
=t\lambda(s,a)(1-\lambda(s,a))
\leq
\lambda(s,a)t.
$$

By Bernstein's inequality,

$$
\mathbb{P}\left(N_t(s,a)-\lambda(s,a)t\leq -u\right)
\leq
\exp\left(
-\frac{u^2}{2(\lambda(s,a)t+u/3)}
\right).
$$

Taking $$u=\lambda(s,a)t/2$$ gives

$$
\mathbb{P}\left(N_t(s,a)\leq \frac{\lambda(s,a)t}{2}\right)
\leq
\exp\left(
-\frac{\lambda(s,a)t}{10}
\right),
$$

up to an absolute constant. Hence after

$$
t\gtrsim \frac{1}{\lambda(s,a)}\log\frac{1}{\delta},
$$

the pair $$(s,a)$$ has been visited at least a constant fraction of its expected number of visits with probability at least $$1-\delta$$.

For all state-action pairs, a union bound gives the familiar burn-in scale

$$
t
\gtrsim
\frac{1}{\lambda_{\min}}
\log\frac{|\mathcal{S}||\mathcal{A}|}{\delta},
$$

where

$$
\lambda_{\min}=\min_{(s,a)}\lambda(s,a).
$$

This kind of argument is essential in asynchronous Q-learning proofs because the number of samples available for a specific state-action pair is itself random.

## 7. Bernstein versus Hoeffding

Hoeffding gives

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(-\frac{2t^2}{\sum_i(b_i-a_i)^2}\right).
$$

Bernstein gives

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(-\frac{t^2}{2(v+bt/3)}\right).
$$

If the variance $$v$$ is comparable to the crude range scale, the two inequalities are of similar order. But if $$v$$ is much smaller, Bernstein can be dramatically sharper.

This is why Bernstein's inequality is usually the better inequality for:

- Bernoulli visit counts;
- rare event indicators;
- empirical transition probabilities with small mass;
- variance-sensitive stochastic approximation bounds.

## 8. RL proof pattern

A typical Bernstein step in RL looks like this:

1. identify a random count or bounded noise sum;
2. center it by subtracting its conditional or unconditional mean;
3. compute or upper-bound the variance proxy;
4. apply Bernstein;
5. union-bound over state-action pairs and time indices.

For example, to ensure that every pair has enough data after a burn-in time, one proves an event such as

$$
\mathcal{E}_{\mathrm{visit}}
=
\left\{
N_t(s,a)\geq \frac{1}{2}\lambda(s,a)t
\text{ for all }t\geq \overline{T}, \text{ and all }(s,a)
\right\}.
$$

Then Bernstein's inequality is used to choose $$\overline{T}$$ so that

$$
\mathbb{P}(\mathcal{E}_{\mathrm{visit}})
\geq 1-\delta.
$$

Once this event holds, empirical estimators for every state-action pair have enough samples, and the rest of the proof can proceed conditionally on this good event.

## References

- S. Bernstein. "On a modification of Chebyshev's inequality and of the error formula of Laplace." 1924.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
- M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.
