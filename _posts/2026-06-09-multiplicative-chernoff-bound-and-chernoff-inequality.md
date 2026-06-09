---
title: "Multiplicative Chernoff Bound and Chernoff Inequality"
date: 2026-06-09
categories: [rl-blogs]
tags: [probability, concentration, chernoff-bound, bernoulli, visit-counts, reinforcement-learning]
math: true
---

Chernoff bounds are exponential tail inequalities obtained by applying Markov's inequality to exponential moments. They are especially powerful for sums of independent Bernoulli random variables. In learning theory and reinforcement learning, multiplicative Chernoff bounds are frequently used to control random counts: number of visits to a state-action pair, number of corrupted samples, number of successful samples in a block, and majority events in robust estimators.

This post has two goals. First, we derive the general Chernoff method. Second, we specialize it to Bernoulli sums and prove the standard multiplicative Chernoff bounds.

## 1. The Chernoff method

Let $$S$$ be a real-valued random variable. For any $$\lambda>0$$ and any $$t\in\mathbb{R}$$,

$$
\mathbb{P}(S\geq t)
=
\mathbb{P}(e^{\lambda S}\geq e^{\lambda t}).
$$

By Markov's inequality,

$$
\mathbb{P}(S\geq t)
\leq
\frac{\mathbb{E}[e^{\lambda S}]}{e^{\lambda t}}
=
\exp(-\lambda t)\mathbb{E}[e^{\lambda S}].
$$

Since this is true for every $$\lambda>0$$,

$$
\mathbb{P}(S\geq t)
\leq
\inf_{\lambda>0}\exp(-\lambda t)\mathbb{E}[e^{\lambda S}].
$$

Similarly, for lower tails, for $$\lambda>0$$,

$$
\mathbb{P}(S\leq t)
=
\mathbb{P}(e^{-\lambda S}\geq e^{-\lambda t})
\leq
\exp(\lambda t)\mathbb{E}[e^{-\lambda S}].
$$

This is the Chernoff method.

## 2. Bernoulli sums

Let $$X_1,\ldots,X_n$$ be independent Bernoulli random variables, not necessarily identically distributed. Let

$$
X=\sum_{i=1}^n X_i,
\qquad
\mu=\mathbb{E}[X]=\sum_{i=1}^n p_i,
$$

where $$p_i=\mathbb{P}(X_i=1)$$.

We first bound the moment generating function of $$X$$. For a Bernoulli random variable $$X_i$$,

$$
\mathbb{E}[e^{\lambda X_i}]
=(1-p_i)+p_ie^\lambda
=1+p_i(e^\lambda-1).
$$

Using $$1+u\leq e^u$$,

$$
\mathbb{E}[e^{\lambda X_i}]
\leq
\exp(p_i(e^\lambda-1)).
$$

By independence,

$$
\mathbb{E}[e^{\lambda X}]
=
\prod_{i=1}^n \mathbb{E}[e^{\lambda X_i}]
\leq
\prod_{i=1}^n\exp(p_i(e^\lambda-1))
=
\exp(\mu(e^\lambda-1)).
$$

This is the key mgf estimate.

## 3. Upper-tail multiplicative Chernoff bound

Let $$\delta>0$$. We want to bound

$$
\mathbb{P}(X\geq (1+\delta)\mu).
$$

For any $$\lambda>0$$,

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\exp(-\lambda(1+\delta)\mu)\mathbb{E}[e^{\lambda X}].
$$

Using the mgf bound,

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\exp\left(
-\lambda(1+\delta)\mu+
\mu(e^\lambda-1)
\right).
$$

We minimize the exponent over $$\lambda>0$$. Define

$$
f(\lambda)
=-\lambda(1+\delta)\mu+
\mu(e^\lambda-1).
$$

Then

$$
f'(\lambda)=-(1+\delta)\mu+\mu e^\lambda.
$$

Setting $$f'(\lambda)=0$$ gives

$$
e^\lambda=1+\delta,
\qquad
\lambda=\log(1+\delta).
$$

Substituting,

$$
f(\log(1+\delta))
=-(1+\delta)\mu\log(1+\delta)+\mu\delta.
$$

Therefore,

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\exp\left(
\mu\delta-(1+\delta)\mu\log(1+\delta)
\right).
$$

Equivalently,

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\left(\frac{e^\delta}{(1+\delta)^{1+\delta}}\right)^\mu.
$$

This is the sharp multiplicative Chernoff upper-tail form.

## 4. Simplified upper-tail forms

The exact form is often simplified. For $$0\leq\delta\leq1$$,

$$
(1+\delta)\log(1+\delta)-\delta
\geq
\frac{\delta^2}{3}.
$$

Therefore,

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\exp\left(-\frac{\mu\delta^2}{3}\right),
\qquad 0\leq\delta\leq1.
$$

For all $$\delta\geq0$$, another common bound is

$$
\mathbb{P}(X\geq (1+\delta)\mu)
\leq
\exp\left(-\frac{\mu\delta^2}{2+\delta}\right).
$$

The latter captures both moderate and large deviation regimes.

## 5. Lower-tail multiplicative Chernoff bound

Let $$\delta\in(0,1)$$. We now bound

$$
\mathbb{P}(X\leq (1-\delta)\mu).
$$

For $$\lambda>0$$,

$$
\mathbb{P}(X\leq (1-\delta)\mu)
=
\mathbb{P}(e^{-\lambda X}\geq e^{-\lambda(1-\delta)\mu}).
$$

By Markov's inequality,

$$
\mathbb{P}(X\leq (1-\delta)\mu)
\leq
\exp(\lambda(1-\delta)\mu)
\mathbb{E}[e^{-\lambda X}].
$$

Using the Bernoulli mgf estimate with $$-\lambda$$,

$$
\mathbb{E}[e^{-\lambda X}]
\leq
\exp(\mu(e^{-\lambda}-1)).
$$

Thus,

$$
\mathbb{P}(X\leq (1-\delta)\mu)
\leq
\exp\left(
\lambda(1-\delta)\mu+
\mu(e^{-\lambda}-1)
\right).
$$

Minimize over $$\lambda>0$$. Define

$$
g(\lambda)=\lambda(1-\delta)\mu+
\mu(e^{-\lambda}-1).
$$

Then

$$
g'(\lambda)=(1-\delta)\mu-\mu e^{-\lambda}.
$$

Setting $$g'(\lambda)=0$$ gives

$$
e^{-\lambda}=1-\delta,
\qquad
\lambda=-\log(1-\delta).
$$

Substituting,

$$
g(-\log(1-\delta))
=-(1-\delta)\mu\log(1-\delta)-\mu\delta.
$$

Hence,

$$
\mathbb{P}(X\leq (1-\delta)\mu)
\leq
\exp\left(
-(\delta+(1-\delta)\log(1-\delta))\mu
\right).
$$

Using the inequality

$$
\delta+(1-\delta)\log(1-\delta)
\geq
\frac{\delta^2}{2},
\qquad 0<\delta<1,
$$

we get

$$
\mathbb{P}(X\leq (1-\delta)\mu)
\leq
\exp\left(-\frac{\mu\delta^2}{2}\right).
$$

## 6. Two-sided multiplicative Chernoff bound

Combining the upper and lower tails, for $$0\leq\delta\leq1$$,

$$
\mathbb{P}(|X-\mu|\geq \delta\mu)
\leq
2\exp\left(-\frac{\mu\delta^2}{3}\right).
$$

The constant $$3$$ is chosen to dominate both upper and lower tails.

Equivalently, with probability at least $$1-\eta$$,

$$
|X-\mu|
\leq
\sqrt{3\mu\log\frac{2}{\eta}}
$$

provided the corresponding relative deviation lies in the moderate regime.

## 7. Additive form for Bernoulli sums

Let $$X=\sum_{i=1}^nX_i$$ with mean $$\mu$$. Taking $$t=\delta\mu$$, the two-sided bound can be written as

$$
\mathbb{P}(|X-\mu|\geq t)
\leq
2\exp\left(-\frac{t^2}{3\mu}\right)
$$

for $$0\leq t\leq \mu$$.

This is especially useful for visit counts and corruption counts, where $$X$$ counts the number of occurrences of some event.

## 8. Application: majority of uncorrupted samples

Suppose each sample is corrupted independently with probability $$\varepsilon<1/2$$. Let

$$
C_n=\sum_{i=1}^nY_i,
\qquad
Y_i\sim\operatorname{Bernoulli}(\varepsilon),
$$

where $$C_n$$ is the number of corrupted samples. Then

$$
\mathbb{E}[C_n]=\varepsilon n.
$$

To show that corrupted samples do not form a majority, we want

$$
C_n<\frac{n}{2}.
$$

Let

$$
\frac{n}{2}=(1+\delta)\varepsilon n.
$$

Then

$$
1+\delta=\frac{1}{2\varepsilon},
\qquad
\delta=\frac{1}{2\varepsilon}-1.
$$

Since $$\varepsilon<1/2$$, we have $$\delta>0$$. The multiplicative Chernoff bound gives

$$
\mathbb{P}\left(C_n\geq \frac{n}{2}\right)
\leq
\left(
\frac{e^\delta}{(1+\delta)^{1+\delta}}
\right)^{\varepsilon n}.
$$

Thus the probability that corruptions form a majority decays exponentially in $$n$$. This type of argument is central when median-type estimators are used under Huber contamination.

## 9. Application: state-action visit counts

Let

$$
N_t(s,a)=\sum_{k=1}^t\mathbf{1}\{(s_k,a_k)=(s,a)\}
$$

under an i.i.d. sampling model with

$$
\mathbb{P}((s_k,a_k)=(s,a))=\lambda(s,a).
$$

Then

$$
\mu=\mathbb{E}[N_t(s,a)]=\lambda(s,a)t.
$$

The lower-tail Chernoff bound gives

$$
\mathbb{P}\left(N_t(s,a)\leq \frac{1}{2}\lambda(s,a)t\right)
\leq
\exp\left(-\frac{\lambda(s,a)t}{8}\right).
$$

Therefore, after

$$
t\gtrsim \frac{1}{\lambda(s,a)}\log\frac{1}{\delta},
$$

the pair $$(s,a)$$ has enough samples with high probability. Union-bounding over all state-action pairs gives the scale

$$
t\gtrsim
\frac{1}{\lambda_{\min}}
\log\frac{|\mathcal{S}||\mathcal{A}|}{\delta}.
$$

## 10. Chernoff versus Hoeffding and Bernstein

For Bernoulli sums, Chernoff and Bernstein are closely related. Both exploit the variance-like scale $$\mu$$ instead of the crude range scale $$n$$. Hoeffding would give a bound depending on $$n$$, while Chernoff gives a multiplicative bound depending on $$\mu$$.

When $$\mu\ll n$$, this difference matters. For rare events, Chernoff bounds are often much sharper.

## References

- H. Chernoff. "A measure of asymptotic efficiency for tests of a hypothesis based on the sum of observations." *Annals of Mathematical Statistics*, 1952.
- S. Boucheron, G. Lugosi, and P. Massart. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press, 2013.
- M. Mitzenmacher and E. Upfal. *Probability and Computing*. Cambridge University Press, 2017.
