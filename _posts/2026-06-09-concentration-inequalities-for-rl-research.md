---
title: "Concentration Inequalities: A Researcher's Guide from Markov to Freedman"
date: 2026-06-09
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [markov inequality, chebyshev inequality, hoeffding inequality, chernoff bound, bernstein inequality, azuma-hoeffding inequality, freedman inequality, martingales, reinforcement learning]
math: true
---

Concentration inequalities are among the most useful tools in modern probability, learning theory, stochastic approximation, and reinforcement learning. They convert local information about a random process into global high-probability control. In the analysis of algorithms, this is often the difference between saying

$$
\mathbb{E}[\mathrm{error}] \text{ is small}
$$

and saying

$$
\mathrm{error} \text{ is small with probability at least } 1-\delta.
$$

The second statement is usually what we need in finite-time reinforcement learning. A typical proof has to control random visitation counts, empirical transition probabilities, noisy Bellman targets, martingale noise terms, and sometimes adversarial or heavy-tailed reward perturbations. Each of these terms asks for a different concentration inequality.

This post is a coherent guide to the inequalities I repeatedly use in research:

1. Markov's inequality.
2. Chebyshev's inequality.
3. Hoeffding's inequality.
4. Chernoff and multiplicative Chernoff bounds.
5. Bernstein's inequality.
6. Azuma-Hoeffding inequality.
7. A high-probability refinement of Azuma-Hoeffding in the spirit of Shamir and Spencer.
8. Freedman's inequality for martingales.

The goal is not just to list statements. The goal is to understand the logic connecting them.

At a high level:

- Markov uses only one moment.
- Chebyshev uses a second moment.
- Hoeffding uses boundedness.
- Chernoff uses the moment generating function.
- Bernstein uses both variance and boundedness.
- Azuma-Hoeffding extends Hoeffding to martingales.
- Freedman extends Bernstein to martingales.
- The refined high-probability Azuma-Hoeffding principle handles processes whose increments are usually small, but only crudely bounded on rare events.

This last point is especially important in robust reinforcement learning. In reward-agnostic robust Q-learning, one may only have a crude deterministic bound on the iterates, while a much sharper bound holds with high probability. Standard Azuma-Hoeffding sees only the crude bound and can become vacuous. A refined inequality can exploit the sharper typical behavior.

## 1. Basic notation

Let $X$ be a real-valued random variable. We write

$$
\mathbb{E}[X]
$$

for its expectation and

$$
\operatorname{Var}(X)
=
\mathbb{E}\left[(X-\mathbb{E}[X])^2\right]
$$

for its variance whenever these quantities exist.

For independent random variables $X_1,\ldots,X_n$, define

$$
S_n = \sum_{i=1}^n X_i.
$$

For martingales, let

$$
\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \cdots \subseteq \mathcal{F}_n
$$

be a filtration. A stochastic process $M_0,M_1,\ldots,M_n$ is a martingale with respect to $(\mathcal{F}_i)_{i=0}^n$ if

$$
M_i \text{ is } \mathcal{F}_i\text{-measurable}
$$

and

$$
\mathbb{E}[M_i \mid \mathcal{F}_{i-1}] = M_{i-1}
\quad \text{for all } i\ge 1.
$$

The martingale difference sequence is

$$
D_i = M_i - M_{i-1}.
$$

Thus

$$
\mathbb{E}[D_i \mid \mathcal{F}_{i-1}] = 0.
$$

This zero conditional mean property is the martingale analogue of independence plus zero mean.

## 2. Markov's inequality: concentration from one moment

Markov's inequality is the most basic concentration inequality. It is weak, but it is extremely general.

### Theorem 1: Markov's inequality

Let $X\ge 0$ be a nonnegative random variable. Then, for every $a>0$,

$$
\mathbb{P}(X\ge a)
\le
\frac{\mathbb{E}[X]}{a}.
$$

### Proof

Since $X\ge 0$, we have the pointwise inequality

$$
X \ge a\mathbf{1}\{X\ge a\}.
$$

Taking expectations on both sides gives

$$
\mathbb{E}[X]
\ge
\mathbb{E}\left[a\mathbf{1}\{X\ge a\}\right].
$$

Since $a$ is deterministic,

$$
\mathbb{E}\left[a\mathbf{1}\{X\ge a\}\right]
=
a\mathbb{P}(X\ge a).
$$

Therefore

$$
\mathbb{E}[X]
\ge
a\mathbb{P}(X\ge a),
$$

and dividing by $a$ proves

$$
\mathbb{P}(X\ge a)
\le
\frac{\mathbb{E}[X]}{a}.
$$

### Intuition

Markov's inequality says: if a nonnegative random variable has small average size, then it cannot be large too often. It does not require independence, boundedness, variance, or tails. It only requires nonnegativity and a finite mean.

This is why Markov's inequality is often used as a first conversion step:

$$
\mathbb{E}[X] \text{ small}
\quad \Longrightarrow \quad
X \text{ small with some probability.}
$$

But the price is that the resulting tail decays only like $1/a$. This is usually too weak for sharp finite-time learning theory.

### A useful general form

If $g$ is nonnegative and increasing, then

$$
\mathbb{P}(X\ge a)
=
\mathbb{P}(g(X)\ge g(a))
\le
\frac{\mathbb{E}[g(X)]}{g(a)}.
$$

This is the bridge to many stronger inequalities. For example:

- taking $g(x)=x^2$ leads to Chebyshev-type bounds;
- taking $g(x)=e^{\lambda x}$ leads to Chernoff-type bounds.

Markov's inequality is therefore not just a standalone result. It is the engine underneath many concentration arguments.

## 3. Chebyshev's inequality: concentration from variance

Markov's inequality uses a first moment. Chebyshev's inequality applies Markov's inequality to a squared deviation.

### Theorem 2: Chebyshev's inequality

Let $X$ be a random variable with finite mean $\mu=\mathbb{E}[X]$ and finite variance $\sigma^2=\operatorname{Var}(X)$. Then, for every $t>0$,

$$
\mathbb{P}\left(\lvert X-\mu\rvert\ge t\right)
\le
\frac{\sigma^2}{t^2}.
$$

Equivalently, for every $k>0$,

$$
\mathbb{P}\left(\lvert X-\mu\rvert\ge k\sigma\right)
\le
\frac{1}{k^2}.
$$

### Proof

Consider the nonnegative random variable

$$
Y=(X-\mu)^2.
$$

The event $\lvert X-\mu\rvert\ge t$ is the same as the event $(X-\mu)^2\ge t^2$. Therefore,

$$
\mathbb{P}\left(\lvert X-\mu\rvert\ge t\right)
=
\mathbb{P}(Y\ge t^2).
$$

By Markov's inequality,

$$
\mathbb{P}(Y\ge t^2)
\le
\frac{\mathbb{E}[Y]}{t^2}.
$$

But

$$
\mathbb{E}[Y]
=
\mathbb{E}\left[(X-\mu)^2\right]
=
\sigma^2.
$$

Hence

$$
\mathbb{P}\left(\lvert X-\mu\rvert\ge t\right)
\le
\frac{\sigma^2}{t^2}.
$$

### Sample mean version

Let $X_1,\ldots,X_n$ be independent random variables with common mean $\mu$ and common variance $\sigma^2$. Define the sample mean

$$
\overline{X}_n = \frac{1}{n}\sum_{i=1}^n X_i.
$$

Then

$$
\mathbb{E}[\overline{X}_n]=\mu
$$

and, by independence,

$$
\operatorname{Var}(\overline{X}_n)
=
\frac{\sigma^2}{n}.
$$

Applying Chebyshev gives

$$
\mathbb{P}\left(\lvert \overline{X}_n-\mu\rvert\ge t\right)
\le
\frac{\sigma^2}{nt^2}.
$$

Equivalently, with probability at least $1-\delta$,

$$
\lvert \overline{X}_n-\mu\rvert
\le
\sqrt{\frac{\sigma^2}{n\delta}}.
$$

### Intuition

Chebyshev is robust because it only needs a second moment. This makes it useful when the distribution may be heavy-tailed. But the dependence on $\delta$ is weak:

$$
\sqrt{\frac{1}{\delta}}.
$$

In learning theory, one usually wants logarithmic dependence on $1/\delta$, such as

$$
\sqrt{\frac{\log(1/\delta)}{n}}.
$$

Chebyshev does not give that. To get logarithmic confidence dependence, we need stronger assumptions, such as boundedness, sub-Gaussianity, or variance plus boundedness.

## 4. The exponential method: the common backbone

Before Hoeffding, Chernoff, and Bernstein, it is useful to isolate the main proof pattern.

Let $S$ be a random variable. For any $\lambda>0$,

$$
\mathbb{P}(S\ge t)
=
\mathbb{P}\left(e^{\lambda S}\ge e^{\lambda t}\right).
$$

By Markov's inequality,

$$
\mathbb{P}\left(e^{\lambda S}\ge e^{\lambda t}\right)
\le
\frac{\mathbb{E}\left[e^{\lambda S}\right]}{e^{\lambda t}}.
$$

Thus

$$
\mathbb{P}(S\ge t)
\le
\exp(-\lambda t)\mathbb{E}\left[e^{\lambda S}\right].
$$

Since this holds for every $\lambda>0$, we can optimize over $\lambda$:

$$
\mathbb{P}(S\ge t)
\le
\inf_{\lambda>0}
\exp(-\lambda t)\mathbb{E}\left[e^{\lambda S}\right].
$$

This is the Chernoff method.

The quantity

$$
\mathbb{E}\left[e^{\lambda S}\right]
$$

is called the moment generating function of $S$. The reason exponential moments are powerful is that they transform sums into products under independence. If $S_n=\sum_{i=1}^n X_i$ and the $X_i$'s are independent, then

$$
\mathbb{E}\left[e^{\lambda S_n}\right]
=
\mathbb{E}\left[e^{\lambda\sum_{i=1}^n X_i}\right]
=
\mathbb{E}\left[\prod_{i=1}^n e^{\lambda X_i}\right]
=
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda X_i}\right].
$$

This factorization is the key to Hoeffding, Chernoff, and Bernstein.

## 5. Hoeffding's inequality: bounded independent noise

Hoeffding's inequality is one of the most common concentration inequalities in learning theory. It says that sums of independent bounded random variables concentrate around their means at a sub-Gaussian rate.

### Hoeffding's lemma

The main ingredient is Hoeffding's lemma.

Let $X$ be a random variable such that

$$
X\in[a,b]
\quad \text{almost surely}.
$$

Let

$$
\mu=\mathbb{E}[X].
$$

Then, for every $\lambda\in\mathbb{R}$,

$$
\mathbb{E}\left[e^{\lambda(X-\mu)}\right]
\le
\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
$$

This says that a centered bounded random variable behaves like a sub-Gaussian random variable with proxy variance $(b-a)^2/4$.

### Theorem 3: Hoeffding's inequality

Let $X_1,\ldots,X_n$ be independent random variables with

$$
X_i\in[a_i,b_i]
\quad \text{almost surely}.
$$

Let

$$
S_n = \sum_{i=1}^n X_i
$$

and

$$
\mathbb{E}[S_n] = \sum_{i=1}^n \mathbb{E}[X_i].
$$

Then, for every $t>0$,

$$
\mathbb{P}\left(S_n-\mathbb{E}[S_n]\ge t\right)
\le
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

Similarly,

$$
\mathbb{P}\left(\mathbb{E}[S_n]-S_n\ge t\right)
\le
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

Therefore, by the union bound,

$$
\mathbb{P}\left(\lvert S_n-\mathbb{E}[S_n]\rvert\ge t\right)
\le
2\exp\left(
-\frac{2t^2}{\sum_{i=1}^n (b_i-a_i)^2}
\right).
$$

### Proof

Define centered variables

$$
Y_i = X_i-\mathbb{E}[X_i].
$$

Then

$$
S_n-\mathbb{E}[S_n]=\sum_{i=1}^n Y_i.
$$

For any $\lambda>0$, Markov's inequality gives

$$
\mathbb{P}\left(\sum_{i=1}^n Y_i\ge t\right)
\le
\exp(-\lambda t)
\mathbb{E}\left[\exp\left(\lambda\sum_{i=1}^n Y_i\right)\right].
$$

By independence,

$$
\mathbb{E}\left[\exp\left(\lambda\sum_{i=1}^n Y_i\right)\right]
=
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda Y_i}\right].
$$

By Hoeffding's lemma,

$$
\mathbb{E}\left[e^{\lambda Y_i}\right]
\le
\exp\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right).
$$

Therefore,

$$
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda Y_i}\right]
\le
\exp\left(
\frac{\lambda^2}{8}
\sum_{i=1}^n (b_i-a_i)^2
\right).
$$

Thus

$$
\mathbb{P}\left(S_n-\mathbb{E}[S_n]\ge t\right)
\le
\exp\left(
-\lambda t
+
\frac{\lambda^2}{8}
\sum_{i=1}^n (b_i-a_i)^2
\right).
$$

Let

$$
V_H = \sum_{i=1}^n (b_i-a_i)^2.
$$

We need to minimize

$$
-\lambda t + \frac{\lambda^2V_H}{8}
$$

over $\lambda>0$. Differentiating gives

$$
-t+\frac{\lambda V_H}{4}=0,
$$

so the optimizer is

$$
\lambda^* = \frac{4t}{V_H}.
$$

Substituting $\lambda^*$ yields

$$
-\lambda^* t + \frac{(\lambda^*)^2V_H}{8}
=
-\frac{4t^2}{V_H}
+
\frac{16t^2}{V_H^2}\frac{V_H}{8}
=
-\frac{2t^2}{V_H}.
$$

Hence

$$
\mathbb{P}\left(S_n-\mathbb{E}[S_n]\ge t\right)
\le
\exp\left(-\frac{2t^2}{V_H}\right).
$$

The lower tail follows by applying the same argument to $-Y_i$. The two-sided bound follows from the union bound.

### Sample mean form

If $X_i\in[a,b]$ independently and all have mean $\mu$, then

$$
\mathbb{P}\left(\lvert \overline{X}_n-\mu\rvert\ge t\right)
\le
2\exp\left(-\frac{2nt^2}{(b-a)^2}\right).
$$

Equivalently, with probability at least $1-\delta$,

$$
\lvert \overline{X}_n-\mu\rvert
\le
(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

### Intuition

Hoeffding is the right first tool when:

- the samples are independent;
- each sample is bounded;
- we do not want to use the variance;
- we want exponential tails.

It gives the canonical rate

$$
\sqrt{\frac{\log(1/\delta)}{n}}.
$$

In RL, Hoeffding often appears when rewards are bounded, or when estimating transition probabilities for a fixed state-action pair using independent samples.

## 6. Chernoff bounds: concentration for sums of Bernoulli variables

Chernoff bounds are especially sharp for sums of Bernoulli random variables. They are central when controlling counts, such as the number of times a state-action pair is visited.

Let

$$
X_i\in\{0,1\}
$$

be independent Bernoulli random variables. Define

$$
S_n=\sum_{i=1}^n X_i
$$

and

$$
\mu=\mathbb{E}[S_n].
$$

The random variable $S_n$ counts how many successes occur.

### Theorem 4: Multiplicative Chernoff bound

For every $\varepsilon>0$,

$$
\mathbb{P}\left(S_n\ge (1+\varepsilon)\mu\right)
\le
\left(\frac{e^{\varepsilon}}{(1+\varepsilon)^{1+\varepsilon}}\right)^\mu.
$$

For every $\varepsilon\in(0,1)$,

$$
\mathbb{P}\left(S_n\le (1-\varepsilon)\mu\right)
\le
\exp\left(-\frac{\mu\varepsilon^2}{2}\right).
$$

A frequently used simplified upper-tail form is

$$
\mathbb{P}\left(S_n\ge (1+\varepsilon)\mu\right)
\le
\exp\left(-\frac{\mu\varepsilon^2}{2+\varepsilon}\right).
$$

In particular, for $\varepsilon\in(0,1)$,

$$
\mathbb{P}\left(S_n\ge (1+\varepsilon)\mu\right)
\le
\exp\left(-\frac{\mu\varepsilon^2}{3}\right).
$$

### Proof of the upper-tail bound

For $\lambda>0$, Markov's inequality gives

$$
\mathbb{P}(S_n\ge (1+\varepsilon)\mu)
\le
\exp(-\lambda(1+\varepsilon)\mu)
\mathbb{E}\left[e^{\lambda S_n}\right].
$$

By independence,

$$
\mathbb{E}\left[e^{\lambda S_n}\right]
=
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda X_i}\right].
$$

If $X_i\sim\operatorname{Bernoulli}(p_i)$, then

$$
\mathbb{E}\left[e^{\lambda X_i}\right]
=
1-p_i+p_i e^{\lambda}
=
1+p_i(e^{\lambda}-1).
$$

Using $1+x\le e^x$,

$$
1+p_i(e^{\lambda}-1)
\le
\exp\left(p_i(e^{\lambda}-1)\right).
$$

Therefore,

$$
\mathbb{E}\left[e^{\lambda S_n}\right]
\le
\exp\left((e^{\lambda}-1)\sum_{i=1}^n p_i\right)
=
\exp\left(\mu(e^{\lambda}-1)\right).
$$

Hence

$$
\mathbb{P}(S_n\ge (1+\varepsilon)\mu)
\le
\exp\left(
-\lambda(1+\varepsilon)\mu
+
\mu(e^{\lambda}-1)
\right).
$$

The optimal choice is

$$
\lambda = \log(1+\varepsilon),
$$

so $e^{\lambda}=1+\varepsilon$. Substituting gives

$$
-\lambda(1+\varepsilon)
+
e^{\lambda}-1
=
-(1+\varepsilon)\log(1+\varepsilon)+\varepsilon.
$$

Therefore

$$
\mathbb{P}(S_n\ge (1+\varepsilon)\mu)
\le
\exp\left(
\mu\left[\varepsilon-(1+\varepsilon)\log(1+\varepsilon)\right]
\right).
$$

This is exactly

$$
\mathbb{P}(S_n\ge (1+\varepsilon)\mu)
\le
\left(\frac{e^{\varepsilon}}{(1+\varepsilon)^{1+\varepsilon}}\right)^\mu.
$$

The simplified form follows from the elementary inequality

$$
(1+\varepsilon)\log(1+\varepsilon)-\varepsilon
\ge
\frac{\varepsilon^2}{2+\varepsilon}.
$$

### Proof of the lower-tail bound

Let $\lambda<0$. By Markov's inequality,

$$
\mathbb{P}(S_n\le (1-\varepsilon)\mu)
=
\mathbb{P}(e^{\lambda S_n}\ge e^{\lambda(1-\varepsilon)\mu}).
$$

Since $\lambda<0$, the event $S_n\le (1-\varepsilon)\mu$ becomes an upper-tail event for $e^{\lambda S_n}$. Thus

$$
\mathbb{P}(S_n\le (1-\varepsilon)\mu)
\le
\exp(-\lambda(1-\varepsilon)\mu)
\mathbb{E}\left[e^{\lambda S_n}\right].
$$

The same moment generating function bound gives

$$
\mathbb{E}\left[e^{\lambda S_n}\right]
\le
\exp\left(\mu(e^{\lambda}-1)\right).
$$

Choose

$$
\lambda = \log(1-\varepsilon) <0.
$$

Then

$$
\mathbb{P}(S_n\le (1-\varepsilon)\mu)
\le
\exp\left(
\mu\left[-\varepsilon-(1-\varepsilon)\log(1-\varepsilon)\right]
\right).
$$

Using the elementary inequality

$$
(1-\varepsilon)\log(1-\varepsilon)+\varepsilon
\ge
\frac{\varepsilon^2}{2},
\quad 0<\varepsilon<1,
$$

we obtain

$$
\mathbb{P}(S_n\le (1-\varepsilon)\mu)
\le
\exp\left(-\frac{\mu\varepsilon^2}{2}\right).
$$

### Intuition

Chernoff bounds are multiplicative. They say that if the expected number of visits is $\mu$, then the probability of seeing much fewer or much more than $\mu$ visits decays exponentially in $\mu$.

This is extremely useful in asynchronous RL. If a state-action pair $(s,a)$ has visitation probability $\lambda(s,a)$, then after $t$ i.i.d. draws, its visit count is roughly binomial with mean

$$
\lambda(s,a)t.
$$

A Chernoff or Bernstein bound can show that, after a burn-in time, every state-action pair has been visited often enough.

## 7. Bernstein's inequality: variance-sensitive concentration

Hoeffding uses only boundedness. Bernstein uses boundedness plus variance. It is sharper when the variance is much smaller than the worst-case squared range.

### Theorem 5: Bernstein's inequality

Let $X_1,\ldots,X_n$ be independent random variables such that

$$
\mathbb{E}[X_i]=0
$$

and

$$
\lvert X_i\rvert\le b
\quad \text{almost surely}.
$$

Let

$$
V = \sum_{i=1}^n \mathbb{E}[X_i^2].
$$

Then, for every $t>0$,

$$
\mathbb{P}\left(\sum_{i=1}^n X_i\ge t\right)
\le
\exp\left(
-\frac{t^2}{2(V+bt/3)}
\right).
$$

Consequently,

$$
\mathbb{P}\left(\left\lvert \sum_{i=1}^n X_i\right\rvert\ge t\right)
\le
2\exp\left(
-\frac{t^2}{2(V+bt/3)}
\right).
$$

### Proof

The proof is again based on the exponential method. We first use a moment generating function bound.

For $\lambda\in(0,3/b)$, one can show that

$$
\mathbb{E}\left[e^{\lambda X_i}\right]
\le
\exp\left(
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}
\right).
$$

This inequality follows from the Taylor expansion of the exponential and the bound $\lvert X_i\rvert\le b$. Indeed, for $k\ge 2$,

$$
\lvert X_i\rvert^k \le b^{k-2}X_i^2.
$$

Using $\mathbb{E}[X_i]=0$,

$$
\mathbb{E}[e^{\lambda X_i}]
=
1+
\sum_{k=2}^{\infty}\frac{\lambda^k\mathbb{E}[X_i^k]}{k!}.
$$

Bounding the higher moments by the second moment yields

$$
\mathbb{E}[e^{\lambda X_i}]
\le
1+
\mathbb{E}[X_i^2]
\sum_{k=2}^{\infty}\frac{\lambda^k b^{k-2}}{k!}.
$$

A standard estimate gives

$$
\sum_{k=2}^{\infty}\frac{\lambda^k b^{k-2}}{k!}
\le
\frac{\lambda^2}{2(1-\lambda b/3)}.
$$

Therefore,

$$
\mathbb{E}[e^{\lambda X_i}]
\le
1+
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}.
$$

Since $1+x\le e^x$,

$$
\mathbb{E}[e^{\lambda X_i}]
\le
\exp\left(
\frac{\lambda^2\mathbb{E}[X_i^2]}{2(1-\lambda b/3)}
\right).
$$

By independence,

$$
\mathbb{E}\left[e^{\lambda\sum_{i=1}^n X_i}\right]
=
\prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]
\le
\exp\left(
\frac{\lambda^2V}{2(1-\lambda b/3)}
\right).
$$

Thus, for every $\lambda\in(0,3/b)$,

$$
\mathbb{P}\left(\sum_{i=1}^n X_i\ge t\right)
\le
\exp\left(
-\lambda t+
\frac{\lambda^2V}{2(1-\lambda b/3)}
\right).
$$

Choosing

$$
\lambda=\frac{t}{V+bt/3}
$$

gives $\lambda b<3$ and yields

$$
\mathbb{P}\left(\sum_{i=1}^n X_i\ge t\right)
\le
\exp\left(
-\frac{t^2}{2(V+bt/3)}
\right).
$$

The two-sided version follows by applying the same argument to $-X_i$ and using the union bound.

### Two regimes

Bernstein has two regimes. When $t$ is small compared to $V/b$, the term $V$ dominates, and the exponent behaves like

$$
-\frac{t^2}{2V}.
$$

This is the sub-Gaussian regime.

When $t$ is large compared to $V/b$, the term $bt$ dominates, and the exponent behaves like

$$
-\frac{3t}{2b}.
$$

This is the sub-exponential regime.

Thus Bernstein smoothly interpolates between Gaussian-like and exponential-like tails.

### Why Bernstein is often sharper than Hoeffding

Suppose $X_i\in[-b,b]$. Hoeffding effectively uses the worst-case scale $b^2n$. Bernstein uses the actual variance

$$
V=\sum_{i=1}^n \mathbb{E}[X_i^2].
$$

If $V\ll nb^2$, Bernstein can be much sharper.

In RL proofs, Bernstein is useful when controlling empirical counts or transition estimates because Bernoulli variables may have small variance when the probability of the event is small.

## 8. Martingales: why independence is not always necessary

Many stochastic approximation algorithms are not sums of independent random variables. The iterate at time $t+1$ depends on the random data observed at time $t$, and also on the entire past through the current iterate.

This is where martingales enter.

A martingale difference sequence $(D_i)_{i=1}^n$ satisfies

$$
\mathbb{E}[D_i\mid \mathcal{F}_{i-1}]=0.
$$

It need not be independent. The random variable $D_i$ may depend strongly on the past. The only requirement is that, after conditioning on the past, its mean is zero.

This is precisely the structure of many noise terms in stochastic approximation. For example, a Bellman noise term often has the form

$$
D_t = \text{random Bellman target at time } t
-
\text{conditional mean Bellman target given the past}.
$$

Then

$$
\mathbb{E}[D_t\mid \mathcal{F}_{t-1}]=0.
$$

This is not independence, but it is enough for martingale concentration.

## 9. Azuma-Hoeffding inequality: Hoeffding for martingales

Azuma-Hoeffding is the martingale analogue of Hoeffding's inequality.

### Theorem 6: Azuma-Hoeffding inequality

Let
$$
\{M_i\}_{i=0}^{n}
$$
be a martingale with respect to
$$
\{\mathcal{F}_i\}_{i=0}^{n}.
$$
Suppose there exist deterministic constants $c_1,\ldots,c_n$ such that

$$
\lvert M_i-M_{i-1}\rvert\le c_i
\quad \text{almost surely for every } i.
$$

Then, for every $t>0$,

$$
\mathbb{P}(M_n-M_0\ge t)
\le
\exp\left(
-\frac{t^2}{2\sum_{i=1}^n c_i^2}
\right).
$$

Similarly,

$$
\mathbb{P}(M_0-M_n\ge t)
\le
\exp\left(
-\frac{t^2}{2\sum_{i=1}^n c_i^2}
\right).
$$

Therefore,

$$
\mathbb{P}(\lvert M_n-M_0\rvert\ge t)
\le
2\exp\left(
-\frac{t^2}{2\sum_{i=1}^n c_i^2}
\right).
$$

### Proof

Let

$$
D_i=M_i-M_{i-1}.
$$

Then

$$
\mathbb{E}[D_i\mid \mathcal{F}_{i-1}]=0
$$

and

$$
D_i\in[-c_i,c_i]
\quad \text{almost surely}.
$$

For $\lambda>0$, Markov's inequality gives

$$
\mathbb{P}(M_n-M_0\ge t)
=
\mathbb{P}\left(\sum_{i=1}^n D_i\ge t\right)
\le
\exp(-\lambda t)
\mathbb{E}\left[\exp\left(\lambda\sum_{i=1}^n D_i\right)\right].
$$

We now control the exponential moment iteratively using conditional expectation. Since $D_n$ is conditionally zero-mean and bounded in $[-c_n,c_n]$, Hoeffding's lemma gives

$$
\mathbb{E}\left[e^{\lambda D_n}\mid \mathcal{F}_{n-1}\right]
\le
\exp\left(\frac{\lambda^2c_n^2}{2}\right).
$$

Therefore,

$$
\mathbb{E}\left[\exp\left(\lambda\sum_{i=1}^n D_i\right)\right]
=
\mathbb{E}\left[
\exp\left(\lambda\sum_{i=1}^{n-1}D_i\right)
\mathbb{E}\left[e^{\lambda D_n}\mid \mathcal{F}_{n-1}\right]
\right]
$$

is at most

$$
\exp\left(\frac{\lambda^2c_n^2}{2}\right)
\mathbb{E}\left[
\exp\left(\lambda\sum_{i=1}^{n-1}D_i\right)
\right].
$$

Repeating this argument backward gives

$$
\mathbb{E}\left[\exp\left(\lambda\sum_{i=1}^n D_i\right)\right]
\le
\exp\left(
\frac{\lambda^2}{2}\sum_{i=1}^n c_i^2
\right).
$$

Hence

$$
\mathbb{P}(M_n-M_0\ge t)
\le
\exp\left(
-\lambda t+
\frac{\lambda^2}{2}\sum_{i=1}^n c_i^2
\right).
$$

The optimal choice is

$$
\lambda=\frac{t}{\sum_{i=1}^n c_i^2}.
$$

Substituting gives

$$
\mathbb{P}(M_n-M_0\ge t)
\le
\exp\left(
-\frac{t^2}{2\sum_{i=1}^n c_i^2}
\right).
$$

The lower-tail and two-sided bounds follow as before.

### Intuition

Azuma-Hoeffding says that a martingale with bounded increments cannot drift too far. The increments may be dependent, but they have no predictable direction.

In stochastic approximation, this is powerful because the noise terms are usually adapted to the past rather than independent. Azuma lets us control sums like

$$
\sum_{t=0}^{T-1}\alpha_t D_t,
$$

provided we can bound each weighted increment:

$$
\lvert \alpha_t D_t\rvert\le c_t.
$$

The resulting fluctuation scale is

$$
\sqrt{\sum_{t=0}^{T-1}c_t^2\log(1/\delta)}.
$$

## 10. A refined high-probability Azuma-Hoeffding principle

Standard Azuma-Hoeffding requires deterministic bounded differences. It only sees the worst-case increment bound.

This can be too crude.

Suppose a martingale difference $D_i$ satisfies both of the following:

1. A crude deterministic bound:

$$
\lvert D_i\rvert\le b_i
\quad \text{always}.
$$

2. A much sharper typical bound:

$$
\lvert D_i\rvert\le c_i
$$

with high probability, where $c_i\ll b_i$.

Standard Azuma-Hoeffding forces us to use $b_i$. But if the event $\lvert D_i\rvert>c_i$ is rare, the true fluctuations should be closer to the scale $c_i$, not $b_i$.

This is the setting where refined high-probability versions of Azuma-Hoeffding become useful.

### Theorem 7: Shamir-Spencer type probabilistic Azuma-Hoeffding inequality

Let $(X_i)_{i=0}^n$ be a martingale with $X_0$ constant. Suppose that for each $0\le i<n$, there exist deterministic numbers $c_i>0$, $b_i>0$, and $r\in(0,1)$ such that

$$
\mathbb{P}\left(\lvert X_{i+1}-X_i\rvert\le c_i\right)
\ge
1-r,
$$

and

$$
\lvert X_{i+1}-X_i\rvert\le b_i
\quad \text{deterministically}.
$$

Assume also that

$$
 b_i\sqrt{r}\le c_i
\quad \text{for all } i.
$$

Then the following bound holds:

$$
\mathbb{P}\left(
\lvert X_n-X_0\rvert
>
\sqrt{32\left(\sum_{i=0}^{n-1}c_i^2\right)\log\left(\frac{2}{\delta}\right)}
+
\sum_{i=0}^{n-1}b_i\sqrt{r}
\right)
<
\delta+2n\sqrt{r}.
$$

This form is adapted from a theorem of Shamir and Spencer. It is the kind of result needed when the increments are not uniformly small deterministically, but are uniformly small on a high-probability event.

### What the theorem says

The ordinary Azuma-Hoeffding bound with deterministic increments $b_i$ would give the scale

$$
\sqrt{\left(\sum_{i=0}^{n-1}b_i^2\right)\log(1/\delta)}.
$$

The refined inequality instead gives the scale

$$
\sqrt{\left(\sum_{i=0}^{n-1}c_i^2\right)\log(1/\delta)}
+
\sum_{i=0}^{n-1}b_i\sqrt{r}.
$$

If $c_i\ll b_i$ and $r$ is very small, this can be dramatically sharper.

The extra term

$$
\sum_{i=0}^{n-1}b_i\sqrt{r}
$$

is the price paid for rare bad increments. The additional failure probability

$$
2n\sqrt{r}
$$

is the price paid for ensuring that these rare events do not accumulate too badly.

### A clean conditional version and proof

The following slightly stronger conditional version is easier to prove and captures the main mechanism.

Let

$$
D_i=X_i-X_{i-1}.
$$

Assume

$$
\mathbb{E}[D_i\mid \mathcal{F}_{i-1}]=0,
$$

$$
\lvert D_i\rvert\le b_i
\quad \text{almost surely},
$$

and

$$
\mathbb{P}(\lvert D_i\rvert>c_i\mid \mathcal{F}_{i-1})
\le r
\quad \text{almost surely}.
$$

Define the bad event

$$
B_i=\{\lvert D_i\rvert>c_i\}.
$$

Now define the truncated and recentered increment

$$
\widetilde{D}_i
=
D_i\mathbf{1}_{B_i^c}
-
\mathbb{E}\left[D_i\mathbf{1}_{B_i^c}\mid\mathcal{F}_{i-1}\right].
$$

Then

$$
\mathbb{E}[\widetilde{D}_i\mid\mathcal{F}_{i-1}]=0,
$$

so

$$
\widetilde{M}_k=
\sum_{i=1}^k \widetilde{D}_i
$$

is a martingale.

Next, because $D_i$ is conditionally mean-zero,

$$
\mathbb{E}[D_i\mathbf{1}_{B_i^c}\mid\mathcal{F}_{i-1}]
=
-
\mathbb{E}[D_i\mathbf{1}_{B_i}\mid\mathcal{F}_{i-1}].
$$

Thus

$$
\left\lvert
\mathbb{E}[D_i\mathbf{1}_{B_i^c}\mid\mathcal{F}_{i-1}]
\right\rvert
\le
\mathbb{E}[\lvert D_i\rvert\mathbf{1}_{B_i}\mid\mathcal{F}_{i-1}]
\le
b_i r.
$$

On $B_i^c$, $\lvert D_i\rvert\le c_i$. Hence

$$
\lvert \widetilde{D}_i\rvert
\le
c_i+b_ir.
$$

If $b_i\sqrt{r}\le c_i$, then $b_ir\le c_i$, and therefore

$$
\lvert \widetilde{D}_i\rvert
\le
2c_i.
$$

Azuma-Hoeffding applied to $(\widetilde{M}_k)$ gives

$$
\mathbb{P}\left(
\left\lvert\sum_{i=1}^n\widetilde{D}_i\right\rvert
>
\sqrt{8\left(\sum_{i=1}^n c_i^2\right)\log\left(\frac{2}{\delta}\right)}
\right)
\le
\delta.
$$

Now compare the original sum and the truncated martingale. On the event that no bad event $B_i$ occurs,

$$
D_i
=
\widetilde{D}_i
+
\mathbb{E}\left[D_i\mathbf{1}_{B_i^c}\mid\mathcal{F}_{i-1}\right].
$$

Therefore, on that event,

$$
\left\lvert\sum_{i=1}^n D_i\right\rvert
\le
\left\lvert\sum_{i=1}^n\widetilde{D}_i\right\rvert
+
\sum_{i=1}^n b_i r.
$$

Since $r\le \sqrt{r}$ for $r\in(0,1)$,

$$
\sum_{i=1}^n b_i r
\le
\sum_{i=1}^n b_i\sqrt{r}.
$$

Finally, by the union bound,

$$
\mathbb{P}\left(\bigcup_{i=1}^n B_i\right)
\le
\sum_{i=1}^n \mathbb{P}(B_i)
\le
nr.
$$

Since $nr\le n\sqrt{r}$, we get the high-probability control

$$
\left\lvert\sum_{i=1}^nD_i\right\rvert
\le
\sqrt{8\left(\sum_{i=1}^n c_i^2\right)\log\left(\frac{2}{\delta}\right)}
+
\sum_{i=1}^n b_i\sqrt{r}
$$

with probability at least

$$
1-\delta-n\sqrt{r}.
$$

This conditional theorem is not exactly the Shamir-Spencer statement above, but it explains the mechanism:

1. truncate the rare large increments;
2. recenter to recover a martingale;
3. apply Azuma to the typical increments;
4. pay a small price for the rare increments.

### Why this matters in robust RL

In reward-agnostic robust Q-learning, the threshold may grow as a polynomial proxy for an unknown reward scale. This gives a crude deterministic bound on the iterates, for example of order

$$
O(T^p),
$$

but on a good event the iterates are actually bounded on the much smaller natural scale, for example

$$
O(1).
$$

If we use standard Azuma-Hoeffding, the martingale increment bound is forced to use the crude $O(T^p)$ scale. This can destroy the finite-time rate. The refined high-probability Azuma-Hoeffding principle allows the proof to exploit the fact that the large bound is only needed on rare events.

This is exactly the conceptual role of the Shamir-Spencer inequality in the analysis of reward-agnostic robust asynchronous Q-learning.

## 11. Freedman's inequality: Bernstein for martingales

Azuma-Hoeffding is the martingale analogue of Hoeffding. Freedman's inequality is the martingale analogue of Bernstein.

Instead of depending only on a uniform increment bound, Freedman also uses the conditional variance process.

Let $(M_i)_{i=0}^n$ be a martingale and define

$$
D_i=M_i-M_{i-1}.
$$

The predictable quadratic variation is

$$
V_n=
\sum_{i=1}^n
\mathbb{E}[D_i^2\mid\mathcal{F}_{i-1}].
$$

This is the martingale analogue of the variance sum

$$
\sum_{i=1}^n\mathbb{E}[X_i^2]
$$

in Bernstein's inequality.

### Theorem 8: Freedman's inequality

Let $(M_i)_{i=0}^n$ be a martingale with increments

$$
D_i=M_i-M_{i-1}.
$$

Assume

$$
D_i\le b
\quad \text{almost surely for every } i.
$$

Let

$$
V_n=
\sum_{i=1}^n
\mathbb{E}[D_i^2\mid\mathcal{F}_{i-1}].
$$

Then, for every $t>0$ and $v>0$,

$$
\mathbb{P}\left(M_n-M_0\ge t \text{ and } V_n\le v\right)
\le
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

A two-sided version follows if $\lvert D_i\rvert\le b$ almost surely:

$$
\mathbb{P}\left(\lvert M_n-M_0\rvert\ge t \text{ and } V_n\le v\right)
\le
2\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

### Proof idea with the exponential supermartingale

Freedman's proof is based on an exponential supermartingale.

For $\lambda\in(0,3/b)$, one can show that

$$
\mathbb{E}\left[e^{\lambda D_i}\mid\mathcal{F}_{i-1}\right]
\le
\exp\left(
\frac{\lambda^2}{2(1-\lambda b/3)}
\mathbb{E}[D_i^2\mid\mathcal{F}_{i-1}]
\right).
$$

This is the conditional analogue of Bernstein's moment generating function bound.

Define

$$
\phi(\lambda)
=
\frac{\lambda^2}{2(1-\lambda b/3)}.
$$

Then

$$
\mathbb{E}\left[
\exp\left(\lambda D_i-\phi(\lambda)\mathbb{E}[D_i^2\mid\mathcal{F}_{i-1}]\right)
\mid\mathcal{F}_{i-1}
\right]
\le 1.
$$

Therefore,

$$
Z_k
=
\exp\left(
\lambda(M_k-M_0)
-
\phi(\lambda)V_k
\right)
$$

is a nonnegative supermartingale. Hence

$$
\mathbb{E}[Z_n]
\le
\mathbb{E}[Z_0]
=
1.
$$

On the event

$$
\{M_n-M_0\ge t,\, V_n\le v\},
$$

we have

$$
Z_n
\ge
\exp(\lambda t-\phi(\lambda)v).
$$

By Markov's inequality applied to $Z_n$,

$$
\mathbb{P}(M_n-M_0\ge t,\, V_n\le v)
\le
\exp(-\lambda t+\phi(\lambda)v).
$$

Optimizing over $\lambda$ yields the Bernstein-type exponent

$$
-\frac{t^2}{2(v+bt/3)}.
$$

Thus

$$
\mathbb{P}\left(M_n-M_0\ge t \text{ and } V_n\le v\right)
\le
\exp\left(
-\frac{t^2}{2(v+bt/3)}
\right).
$$

### Intuition

Freedman's inequality says that martingale fluctuations are governed not only by the worst possible jump size $b$, but also by the accumulated conditional variance $V_n$.

This is extremely important in adaptive processes. Even if the increments are bounded by $b$, the process may usually have much smaller conditional variance. Freedman captures this. Azuma-Hoeffding does not.

In reinforcement learning, Freedman is often the right tool when the martingale noise has a variance structure that depends on the current state, action, or value function.

## 12. How the inequalities fit together

Here is the conceptual progression.

### Markov

Use when you only know

$$
\mathbb{E}[X].
$$

It gives

$$
\mathbb{P}(X\ge a)
\le
\frac{\mathbb{E}[X]}{a}.
$$

It is general but weak.

### Chebyshev

Use when you know

$$
\operatorname{Var}(X).
$$

It gives

$$
\mathbb{P}(\lvert X-\mathbb{E}[X]\rvert\ge t)
\le
\frac{\operatorname{Var}(X)}{t^2}.
$$

It handles heavy tails with finite variance but has weak confidence dependence.

### Hoeffding

Use for independent bounded variables. It gives

$$
\mathbb{P}\left(\lvert S_n-\mathbb{E}[S_n]\rvert\ge t\right)
\le
2\exp\left(-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}\right).
$$

It is simple and gives clean sub-Gaussian concentration.

### Chernoff

Use for sums of Bernoulli variables or nonnegative counting variables with tractable moment generating functions. It gives multiplicative control around the mean:

$$
S_n \approx \mu
$$

with exponentially high probability in $\mu$.

### Bernstein

Use for independent bounded variables when variance information matters. It gives

$$
\mathbb{P}\left(S_n\ge t\right)
\le
\exp\left(-\frac{t^2}{2(V+bt/3)}\right).
$$

It is sharper than Hoeffding when $V\ll nb^2$.

### Azuma-Hoeffding

Use for martingales with deterministic bounded increments. It gives

$$
\mathbb{P}(\lvert M_n-M_0\rvert\ge t)
\le
2\exp\left(-\frac{t^2}{2\sum_{i=1}^n c_i^2}\right).
$$

It is the default martingale concentration inequality.

### Refined high-probability Azuma-Hoeffding

Use when increments are always bounded by $b_i$, but are usually bounded by a much smaller $c_i$. It gives concentration at the typical scale $c_i$, plus a small penalty for rare bad increments.

This is useful when a deterministic worst-case bound exists only for technical safety, but the actual algorithm behaves much better on a good event.

### Freedman

Use for martingales when conditional variance matters. It gives

$$
\mathbb{P}\left(M_n-M_0\ge t \text{ and } V_n\le v\right)
\le
\exp\left(-\frac{t^2}{2(v+bt/3)}\right).
$$

It is the variance-sensitive martingale analogue of Bernstein.

## 13. A research-oriented decision guide

When facing a random error term, ask the following questions.

### Question 1: Is the object nonnegative and do I only know its expectation?

Use Markov.

Example:

$$
\mathbb{P}(X\ge a)
\le
\frac{\mathbb{E}[X]}{a}.
$$

### Question 2: Do I only know a second moment or variance?

Use Chebyshev.

Example:

$$
\mathbb{P}(\lvert X-\mu\rvert\ge t)
\le
\frac{\sigma^2}{t^2}.
$$

### Question 3: Is it a sum of independent bounded variables?

Use Hoeffding.

Example:

$$
\lvert \overline{X}_n-\mu\rvert
=
O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right).
$$

### Question 4: Is it a sum of indicators or a visit count?

Use Chernoff or Bernstein.

Example:

$$
\mathbb{P}\left(N_t(s,a)\le \frac{1}{2}\lambda(s,a)t\right)
\le
\exp(-c\lambda(s,a)t)
$$

for a numerical constant $c>0$.

### Question 5: Is it a martingale noise term with bounded increments?

Use Azuma-Hoeffding.

Example:

$$
\sum_{t=0}^{T-1}\alpha_t D_t
=
O\left(
\sqrt{\sum_{t=0}^{T-1}\alpha_t^2 c_t^2\log(1/\delta)}
\right)
$$

with high probability.

### Question 6: Is it a martingale noise term with small conditional variance?

Use Freedman.

Example:

$$
\sum_{t=0}^{T-1}D_t
=
O\left(
\sqrt{V_T\log(1/\delta)}+b\log(1/\delta)
\right)
$$

with high probability, where $V_T$ is the predictable quadratic variation.

### Question 7: Is the martingale increment usually small but rarely large?

Use a refined high-probability Azuma-Hoeffding inequality.

Example:

$$
\lvert D_t\rvert\le b_t \quad \text{always},
$$

but

$$
\lvert D_t\rvert\le c_t \quad \text{with high probability},
$$

where $c_t\ll b_t$. Then the refined inequality can preserve the $c_t$-scale rather than paying the full $b_t$-scale.

## 14. How these tools appear in reinforcement learning proofs

Finite-time RL proofs often contain several different random objects. No single concentration inequality is best for all of them.

### Visit counts

Suppose a state-action pair $(s,a)$ is visited with probability $\lambda(s,a)$. Under an i.i.d. sampling abstraction, the count

$$
N_t(s,a)
=
\sum_{k=1}^t \mathbf{1}\{(s_k,a_k)=(s,a)\}
$$

is a sum of Bernoulli random variables with mean

$$
\mathbb{E}[N_t(s,a)]
=
\lambda(s,a)t.
$$

Chernoff or Bernstein gives

$$
N_t(s,a)
\ge
\frac{1}{2}\lambda(s,a)t
$$

with high probability once $t$ is large enough. This kind of event is needed before empirical estimates for every state-action pair become reliable.

### Empirical rewards

If rewards are bounded and independent, Hoeffding controls the empirical reward mean:

$$
\left\lvert
\frac{1}{N}\sum_{i=1}^N r_i(s,a)-R(s,a)
\right\rvert
=
O\left(\sqrt{\frac{\log(1/\delta)}{N}}\right).
$$

If rewards are heavy-tailed but have finite variance, Chebyshev gives a weaker guarantee, and one often needs robust mean estimation to recover sub-Gaussian-type behavior.

### Bellman noise

A typical Q-learning update contains a random target and its conditional mean. The difference is a martingale difference. This is where Azuma-Hoeffding or Freedman enters.

If the noise increments are uniformly bounded, Azuma-Hoeffding is natural. If one can exploit conditional variance, Freedman can sharpen the result.

### Robust reward estimation and rare events

In robust RL under adversarial reward corruption, the analysis often constructs a good event on which robust reward estimates are accurate. Outside this event, the algorithm must still remain controlled. This creates two scales:

1. a deterministic safety scale;
2. a sharper high-probability scale.

Standard Azuma-Hoeffding only sees the deterministic safety scale. Refined high-probability Azuma-Hoeffding inequalities are designed precisely for this two-scale situation.

## 15. Summary

The inequalities in this post should be viewed as a hierarchy.

At the bottom, Markov and Chebyshev need very little structure, but give weak tails. Hoeffding and Chernoff give exponential tails by exploiting boundedness and independence. Bernstein improves Hoeffding by using variance. Azuma-Hoeffding moves from independence to martingales. Freedman adds variance sensitivity back into the martingale setting. Finally, refined high-probability Azuma-Hoeffding inequalities handle the subtle situation where the increments are usually small but only crudely bounded on rare events.

This hierarchy is exactly why concentration inequalities are so central in reinforcement learning theory. A finite-time RL proof is not just one concentration argument; it is usually a careful orchestration of several different concentration tools, each matched to the specific random object being controlled.

## References

- K. Azuma. Weighted sums of certain dependent random variables. *Tohoku Mathematical Journal*, 19:357--367, 1967.
- S. Bernstein. On a modification of Chebyshev's inequality and of the error formula of Laplace. 1924.
- H. Chernoff. A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations. *The Annals of Mathematical Statistics*, 23(4):493--507, 1952.
- W. Hoeffding. Probability inequalities for sums of bounded random variables. *Journal of the American Statistical Association*, 58(301):13--30, 1963.
- D. A. Freedman. On tail probabilities for martingales. *The Annals of Probability*, 3(1):100--118, 1975.
- E. Shamir and J. Spencer. Sharp concentration of the chromatic number on random graphs $G_{n,p}$. *Combinatorica*, 7(1):121--129, 1987.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
- P. Massart. *Concentration Inequalities and Model Selection*. Springer, 2007.
- S. Maity and A. Mitra. Corruption-Tolerant Asynchronous Q-learning with Near-Optimal Rates. ICML, 2026.
