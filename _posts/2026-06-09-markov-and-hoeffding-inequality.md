---
title: "Markov and Hoeffding's Inequality"
date: 2026-06-09
categories: [rl-blogs]
tags: [probability, concentration, markov-inequality, hoeffding-inequality, reinforcement-learning]
math: true
---

Concentration inequalities are tools for converting randomness into usable finite-sample statements. In reinforcement learning, optimization, statistics, and learning theory, we almost never need to know the entire distribution of a random variable. We often only need a statement of the following kind:

> with probability at least $$1-\delta$$, a random error term is no larger than a controlled deterministic quantity.

This post studies two of the most fundamental concentration inequalities: Markov's inequality and Hoeffding's inequality. Markov's inequality is the most primitive tail bound: it only needs nonnegativity and the first moment. Hoeffding's inequality is much sharper, but it requires independence and boundedness. In RL proofs, these two inequalities often appear at different levels of sophistication: Markov's inequality underlies the Chernoff method, while Hoeffding's inequality controls empirical averages of bounded noise, transition counts, and bounded martingale differences.

## 1. Markov's inequality

Let $$X$$ be a nonnegative random variable, i.e., $$X \geq 0$$ almost surely. Suppose $$\mathbb{E}[X] < \infty$$. Then, for every $$a>0$$,

$$
\mathbb{P}(X \geq a) \leq \frac{\mathbb{E}[X]}{a}.
$$

This is Markov's inequality.

### Proof

The proof is almost embarrassingly simple, but the idea is extremely powerful. Since $$X \geq 0$$, on the event $$\{X \geq a\}$$ we have $$X \geq a$$. Therefore,

$$
X \geq a\mathbf{1}\{X \geq a\}.
$$

Taking expectations on both sides gives

$$
\mathbb{E}[X]
\geq
\mathbb{E}\left[a\mathbf{1}\{X \geq a\}\right]
=
a\mathbb{P}(X \geq a).
$$

Dividing by $$a$$ gives

$$
\mathbb{P}(X \geq a)
\leq
\frac{\mathbb{E}[X]}{a}.
$$

That is the entire proof.

## 2. Why Markov's inequality matters

At first glance, Markov's inequality looks weak. It only gives a polynomial tail bound. For example, if $$\mathbb{E}[X]=1$$, then

$$
\mathbb{P}(X \geq 100) \leq \frac{1}{100}.
$$

This is useful, but not exponentially small. However, Markov's inequality becomes extremely powerful when applied to a transformed random variable. If $$\phi$$ is nonnegative and increasing, then

$$
\mathbb{P}(X \geq a)
=
\mathbb{P}(\phi(X) \geq \phi(a))
\leq
\frac{\mathbb{E}[\phi(X)]}{\phi(a)}.
$$

The most important choice is

$$
\phi(x)=e^{\lambda x}, \qquad \lambda>0.
$$

Then

$$
\mathbb{P}(X \geq a)
\leq
\frac{\mathbb{E}[e^{\lambda X}]}{e^{\lambda a}}
=
\exp(-\lambda a)\mathbb{E}[e^{\lambda X}].
$$

Optimizing over $$\lambda>0$$ gives the Chernoff method:

$$
\mathbb{P}(X \geq a)
\leq
\inf_{\lambda>0}\exp(-\lambda a)\mathbb{E}[e^{\lambda X}].
$$

Thus, many exponential concentration inequalities are Markov's inequality applied to the exponential random variable $$e^{\lambda X}$$.

## 3. From Markov to concentration of sums

Suppose $$X_1,\ldots,X_n$$ are independent random variables and define

$$
S_n = \sum_{i=1}^n (X_i-\mathbb{E}X_i).
$$

To bound $$\mathbb{P}(S_n \geq t)$$, Markov's inequality gives

$$
\mathbb{P}(S_n \geq t)
=
\mathbb{P}(e^{\lambda S_n} \geq e^{\lambda t})
\leq
\exp(-\lambda t)\mathbb{E}[e^{\lambda S_n}].
$$

By independence,

$$
\mathbb{E}[e^{\lambda S_n}]
=
\prod_{i=1}^n \mathbb{E}\left[e^{\lambda(X_i-\mathbb{E}X_i)}\right].
$$

So the problem reduces to controlling the moment generating functions of the centered variables $$X_i-\mathbb{E}X_i$$. Hoeffding's inequality does exactly this when each $$X_i$$ is bounded.

## 4. Hoeffding's lemma

Hoeffding's inequality is usually proved through the following lemma.

**Lemma.** Let $$X$$ be a random variable such that

$$
\mathbb{E}[X]=0, \qquad X\in[a,b] \quad \text{almost surely}.
$$

Then, for every $$\lambda\in\mathbb{R}$$,

$$
\mathbb{E}[e^{\lambda X}]
\leq
\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
$$

This says that every centered bounded random variable is sub-Gaussian with variance proxy $$(b-a)^2/4$$.

### Proof of Hoeffding's lemma

Because the exponential function is convex, for any $$x\in[a,b]$$ we can write $$x$$ as a convex combination of $$a$$ and $$b$$:

$$
x = \frac{b-x}{b-a}a + \frac{x-a}{b-a}b.
$$

Convexity of $$e^{\lambda x}$$ gives

$$
e^{\lambda x}
\leq
\frac{b-x}{b-a}e^{\lambda a}
+
\frac{x-a}{b-a}e^{\lambda b}.
$$

Taking expectations and using $$\mathbb{E}[X]=0$$,

$$
\mathbb{E}[e^{\lambda X}]
\leq
\frac{b}{b-a}e^{\lambda a}
-
\frac{a}{b-a}e^{\lambda b}.
$$

Define

$$
p = \frac{-a}{b-a}, \qquad 1-p = \frac{b}{b-a}.
$$

Since $$0\in[a,b]$$ because $$\mathbb{E}X=0$$ and $$X\in[a,b]$$ almost surely, we have $$p\in[0,1]$$. Also, if we define

$$
h = \lambda(b-a),
$$

then

$$
\frac{b}{b-a}e^{\lambda a}
-
\frac{a}{b-a}e^{\lambda b}
=
(1-p)e^{-ph}+pe^{(1-p)h}.
$$

Let

$$
L(h)=\log\left((1-p)e^{-ph}+pe^{(1-p)h}\right).
$$

One can verify that

$$
L(0)=0, \qquad L'(0)=0,
$$

and a direct calculation gives

$$
L''(h)\leq \frac{1}{4}
$$

for all $$h\in\mathbb{R}$$. Taylor's theorem with remainder then yields

$$
L(h)
=
L(0)+hL'(0)+\frac{h^2}{2}L''(\xi)
\leq
\frac{h^2}{8}
$$

for some $$\xi$$ between $$0$$ and $$h$$. Therefore,

$$
\mathbb{E}[e^{\lambda X}]
\leq
\exp(L(h))
\leq
\exp\left(\frac{h^2}{8}\right)
=
\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
$$

This proves the lemma.

## 5. Hoeffding's inequality

Let $$X_1,\ldots,X_n$$ be independent random variables such that

$$
X_i\in[a_i,b_i] \quad \text{almost surely}.
$$

Define

$$
S_n = \sum_{i=1}^n (X_i-\mathbb{E}X_i).
$$

Then, for every $$t>0$$,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(
-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}
\right).
$$

Similarly,

$$
\mathbb{P}(|S_n|\geq t)
\leq
2\exp\left(
-\frac{2t^2}{\sum_{i=1}^n(b_i-a_i)^2}
\right).
$$

### Proof

Fix $$\lambda>0$$. By Markov's inequality,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp(-\lambda t)\mathbb{E}[e^{\lambda S_n}].
$$

Using independence,

$$
\mathbb{E}[e^{\lambda S_n}]
=
\prod_{i=1}^n
\mathbb{E}\left[e^{\lambda(X_i-\mathbb{E}X_i)}\right].
$$

The variable $$X_i-\mathbb{E}X_i$$ is centered and lies in an interval of length $$b_i-a_i$$. By Hoeffding's lemma,

$$
\mathbb{E}\left[e^{\lambda(X_i-\mathbb{E}X_i)}\right]
\leq
\exp\left(\frac{\lambda^2(b_i-a_i)^2}{8}\right).
$$

Therefore,

$$
\mathbb{E}[e^{\lambda S_n}]
\leq
\exp\left(
\frac{\lambda^2}{8}\sum_{i=1}^n(b_i-a_i)^2
\right).
$$

Hence,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(
-\lambda t
+
\frac{\lambda^2}{8}\sum_{i=1}^n(b_i-a_i)^2
\right).
$$

Let

$$
V = \sum_{i=1}^n(b_i-a_i)^2.
$$

The exponent is

$$
f(\lambda)=-\lambda t+\frac{\lambda^2V}{8}.
$$

The minimizing value is obtained from

$$
f'(\lambda)=-t+\frac{\lambda V}{4}=0,
$$

so

$$
\lambda^*=\frac{4t}{V}.
$$

Substituting this value gives

$$
f(\lambda^*)
=-\frac{4t^2}{V}+\frac{16t^2}{V^2}\frac{V}{8}
=-\frac{4t^2}{V}+\frac{2t^2}{V}
=-\frac{2t^2}{V}.
$$

Thus,

$$
\mathbb{P}(S_n\geq t)
\leq
\exp\left(-\frac{2t^2}{V}\right).
$$

The lower-tail inequality follows by applying the same result to $$-S_n$$. The two-sided inequality follows from the union bound:

$$
\mathbb{P}(|S_n|\geq t)
\leq
\mathbb{P}(S_n\geq t)+\mathbb{P}(-S_n\geq t).
$$

## 6. Sample mean form

Suppose $$X_1,\ldots,X_n$$ are i.i.d. and $$X_i\in[a,b]$$ almost surely. Let

$$
\overline{X}_n = \frac{1}{n}\sum_{i=1}^n X_i,
\qquad
\mu=\mathbb{E}[X_1].
$$

Then

$$
\mathbb{P}(|\overline{X}_n-\mu|\geq \varepsilon)
\leq
2\exp\left(-\frac{2n\varepsilon^2}{(b-a)^2}\right).
$$

Equivalently, with probability at least $$1-\delta$$,

$$
|\overline{X}_n-\mu|
\leq
(b-a)\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

This is the form most commonly used in learning theory.

## 7. How this appears in RL

Hoeffding's inequality is useful whenever the stochastic object is bounded and independent. Examples include:

1. estimating bounded mean rewards from i.i.d. samples;
2. estimating transition probabilities under a generative model;
3. controlling the number of visits to a state-action pair under i.i.d. sampling;
4. bounding bounded martingale differences after conditioning on the past.

For example, suppose $$Y_1,\ldots,Y_n$$ are bounded reward samples with $$Y_i\in[0,R_{\max}]$$ and mean $$R(s,a)$$. Then with probability at least $$1-\delta$$,

$$
\left|
\frac{1}{n}\sum_{i=1}^nY_i-R(s,a)
\right|
\leq
R_{\max}\sqrt{\frac{\log(2/\delta)}{2n}}.
$$

This gives a high-probability reward-estimation error of order

$$
O\left(\sqrt{\frac{\log(1/\delta)}{n}}\right).
$$

In a finite state-action space, we usually need the guarantee to hold for all $$(s,a)\in\mathcal{S}\times\mathcal{A}$$. A union bound replaces $$\delta$$ by $$\delta/(|\mathcal{S}||\mathcal{A}|)$$, giving

$$
\left|
\widehat{R}(s,a)-R(s,a)
\right|
\lesssim
R_{\max}\sqrt{\frac{\log(|\mathcal{S}||\mathcal{A}|/\delta)}{n(s,a)}}.
$$

This is one of the basic reasons logarithmic factors involving $$|\mathcal{S}||\mathcal{A}|$$ appear in finite-time RL bounds.

## 8. Limitations

Hoeffding's inequality is powerful but not universal.

First, it requires bounded random variables. If rewards are heavy-tailed and only have finite variance, Hoeffding's inequality is no longer directly applicable.

Second, it does not exploit variance. If the variance is much smaller than the range, Bernstein's inequality can give sharper bounds.

Third, the classical version assumes independence. In stochastic approximation and Q-learning, the noise terms are often martingale differences rather than independent variables. This motivates Azuma-Hoeffding and Freedman's inequalities.

## References

- W. Hoeffding. "Probability inequalities for sums of bounded random variables." *Journal of the American Statistical Association*, 1963.
- M. J. Wainwright. *High-Dimensional Statistics: A Non-Asymptotic Viewpoint*. Cambridge University Press, 2019.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
