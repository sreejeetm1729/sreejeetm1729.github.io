---
title: "A High-Probability Azuma-Hoeffding Inequality for Almost-Bounded Martingales"
date: 2026-06-09
categories: [rl-blogs]
tags: [probability, concentration, martingales, azuma-hoeffding, shamir-spencer, robust-rl]
math: true
---

The standard Azuma-Hoeffding inequality is extremely useful when martingale increments are deterministically bounded by good constants. But in many modern learning proofs, especially in reinforcement learning with robust or adaptive estimators, the following more delicate structure appears:

- the martingale increment is always bounded, but the deterministic bound is very crude;
- with high probability, the same increment is bounded by a much smaller and more meaningful quantity.

If we use standard Azuma-Hoeffding with the crude deterministic bound, the final estimate can be vacuous. The refined inequality discussed in this post addresses precisely this issue. The version below is based on the martingale concentration tool of Shamir and Spencer (1987), and it is especially useful for analyses where one has both a coarse deterministic envelope and a fine high-probability envelope.

This is the kind of situation that appears in robust Q-learning with reward-agnostic thresholding: the iterates may be crudely bounded by a growing polynomial threshold, while on a good event they behave as if they were uniformly bounded by a constant. A standard Azuma-Hoeffding step would see only the crude bound; the refined inequality sees the typical bound.

## 1. Why standard Azuma can fail

Let $$\{X_i\}_{i=0}^n$$ be a martingale and define

$$
\Delta_i=X_{i+1}-X_i.
$$

Suppose the deterministic bound is

$$
|\Delta_i|\leq b_i,
$$

but $$b_i$$ is very large. Standard Azuma-Hoeffding gives

$$
|X_n-X_0|
\lesssim
\sqrt{\left(\sum_{i=0}^{n-1}b_i^2\right)\log(1/\delta)}.
$$

If $$b_i$$ scales like $$T^p$$ in a finite-horizon proof, this produces a useless bound.

Now suppose, however, that with high probability each increment satisfies a much smaller bound:

$$
|\Delta_i|\leq c_i
$$

except on rare events. Ideally, we would like a concentration inequality whose leading term depends on $$\sum_i c_i^2$$, not $$\sum_i b_i^2$$. The price we should pay is an additional error term reflecting the rare exceptional events. The refined Azuma-Hoeffding inequality gives exactly this.

## 2. Statement of the refined inequality

Let $$X_0,X_1,\ldots,X_n$$ be a martingale with $$X_0$$ constant. Suppose that for constants $$c_i,b_i>0$$ and a parameter $$r\in(0,1)$$, the following two conditions hold for every $$0\leq i<n$$:

$$
\mathbb{P}(|X_{i+1}-X_i|\leq c_i)\geq 1-r,
$$

and

$$
|X_{i+1}-X_i|\leq b_i
\qquad \text{deterministically}.
$$

Assume also that

$$
b_i\sqrt{r}\leq c_i
\qquad \text{for every }i.
$$

Then, for every $$\delta\in(0,1)$$, with probability at least

$$
1-\delta-2n\sqrt{r},
$$

we have

$$
|X_n-X_0|
\leq
\sqrt{
32\left(\sum_{i=0}^{n-1}c_i^2\right)
\log\frac{2}{\delta}
}
+
\sum_{i=0}^{n-1}b_i\sqrt{r}.
$$

The constants are not the main point. The structure is:

$$
|X_n-X_0|
\lesssim
\sqrt{\left(\sum_i c_i^2\right)\log(1/\delta)}
+
\sum_i b_i\sqrt{r}.
$$

The leading concentration term uses the typical increment scale $$c_i$$, while the rare-event correction uses the deterministic scale $$b_i$$ multiplied by $$\sqrt r$$.

## 3. The truncation idea

The proof constructs a new martingale $$\{Y_i\}$$ that removes the dangerous rare increments of $$\{X_i\}$$ while remaining close to $$\{X_i\}$$ on a good event.

Define the bad event

$$
F_i=\{|X_{i+1}-X_i|>c_i\}.
$$

Let

$$
p_i=\mathbb{P}(F_i\mid \mathcal{F}_i).
$$

Although $$\mathbb{P}(F_i)\leq r$$, the conditional probability $$p_i$$ may occasionally be larger. We therefore separate indices into two cases:

1. If $$p_i\geq \sqrt r$$, the conditional chance of a bad increment is too large.
2. If $$p_i<\sqrt r$$, the bad increment is conditionally rare.

The construction terminates the auxiliary martingale in the first case and truncates the increment in the second case.

## 4. Constructing the auxiliary martingale

Set

$$
Y_0=X_0.
$$

Assume $$Y_i$$ has been constructed and the process has not terminated. If

$$
p_i\geq \sqrt r,
$$

then terminate the process by setting

$$
Y_j=Y_i
\qquad \text{for all }j\geq i+1.
$$

If instead

$$
p_i<\sqrt r,
$$

define a truncated version of $$X_{i+1}$$ by

$$
\overline{X}_{i+1}
=
\begin{cases}
X_i, & \text{if }F_i\text{ occurs},\\
X_{i+1}, & \text{otherwise}.
\end{cases}
$$

Thus, on a bad increment, we replace $$X_{i+1}$$ by the old value $$X_i$$. On a good increment, we leave it unchanged.

Now define the correction term

$$
A_i
=
\mathbb{E}[\overline{X}_{i+1}-X_{i+1}\mid \mathcal{F}_i].
$$

Since $$\{X_i\}$$ is a martingale,

$$
\mathbb{E}[X_{i+1}\mid \mathcal{F}_i]=X_i.
$$

Therefore,

$$
\mathbb{E}[\overline{X}_{i+1}\mid \mathcal{F}_i]
=
\mathbb{E}[X_{i+1}\mid \mathcal{F}_i]+A_i
=
X_i+A_i.
$$

Define

$$
Y_{i+1}=Y_i+\overline{X}_{i+1}-X_i-A_i.
$$

Then

$$
Y_{i+1}-Y_i
=
\overline{X}_{i+1}-X_i-A_i.
$$

Taking conditional expectation gives

$$
\mathbb{E}[Y_{i+1}-Y_i\mid \mathcal{F}_i]
=
\mathbb{E}[\overline{X}_{i+1}-X_i-A_i\mid \mathcal{F}_i]
=
(X_i+A_i)-X_i-A_i
=0.
$$

So $$\{Y_i\}$$ is a martingale until termination, and it remains a martingale after termination because its increments are zero.

## 5. Bounding the increments of the auxiliary martingale

On the event $$F_i^c$$,

$$
\overline{X}_{i+1}=X_{i+1},
$$

and therefore

$$
|\overline{X}_{i+1}-X_i|
=
|X_{i+1}-X_i|
\leq c_i.
$$

On the event $$F_i$$,

$$
\overline{X}_{i+1}=X_i,
$$

so

$$
|\overline{X}_{i+1}-X_i|=0.
$$

Thus, always in the non-terminated case,

$$
|\overline{X}_{i+1}-X_i|\leq c_i.
$$

Next, we bound $$A_i$$. Since $$\overline{X}_{i+1}=X_{i+1}$$ on $$F_i^c$$, the difference $$\overline{X}_{i+1}-X_{i+1}$$ is nonzero only on $$F_i$$. Using the deterministic bound $$|X_{i+1}-X_i|\leq b_i$$,

$$
|\overline{X}_{i+1}-X_{i+1}|
\leq b_i\mathbf{1}_{F_i}.
$$

Therefore,

$$
|A_i|
=
\left|\mathbb{E}[\overline{X}_{i+1}-X_{i+1}\mid \mathcal{F}_i]\right|
\leq
b_i\mathbb{P}(F_i\mid \mathcal{F}_i)
=b_ip_i.
$$

In the non-terminated case, $$p_i<\sqrt r$$. Hence

$$
|A_i|
\leq b_i\sqrt r.
$$

By assumption, $$b_i\sqrt r\leq c_i$$. Therefore,

$$
|Y_{i+1}-Y_i|
\leq
|\overline{X}_{i+1}-X_i|+|A_i|
\leq
c_i+b_i\sqrt r
\leq
2c_i.
$$

Thus the auxiliary martingale has deterministic increments bounded by $$2c_i$$.

## 6. Applying standard Azuma to the auxiliary martingale

Since $$\{Y_i\}$$ is a martingale with increments bounded by $$2c_i$$, Azuma-Hoeffding gives

$$
\mathbb{P}\left(
|Y_n-Y_0|
\geq u
\right)
\leq
2\exp\left(
-\frac{u^2}{2\sum_{i=0}^{n-1}(2c_i)^2}
\right).
$$

Equivalently, with probability at least $$1-\delta$$,

$$
|Y_n-Y_0|
\leq
\sqrt{
8\left(\sum_{i=0}^{n-1}c_i^2\right)
\log\frac{2}{\delta}
}.
$$

The stated theorem uses a slightly looser constant $$32$$. This is harmless and convenient when accounting for all good-event reductions.

## 7. Relating the auxiliary martingale back to the original martingale

Let $$\mathcal{G}$$ be the good event on which:

1. termination never occurs;
2. no bad increment $$F_i$$ occurs.

On this event, $$\overline{X}_{i+1}=X_{i+1}$$ for every $$i$$. Therefore,

$$
Y_{i+1}-Y_i
=
X_{i+1}-X_i-A_i.
$$

Summing from $$i=0$$ to $$n-1$$ gives

$$
Y_n-Y_0
=
X_n-X_0-
\sum_{i=0}^{n-1}A_i.
$$

Thus,

$$
X_n-X_0
=
Y_n-Y_0+
\sum_{i=0}^{n-1}A_i.
$$

Taking absolute values,

$$
|X_n-X_0|
\leq
|Y_n-Y_0|+
\sum_{i=0}^{n-1}|A_i|.
$$

Using $$|A_i|\leq b_i\sqrt r$$,

$$
|X_n-X_0|
\leq
|Y_n-Y_0|+
\sum_{i=0}^{n-1}b_i\sqrt r.
$$

Combining this with Azuma-Hoeffding for $$Y_n-Y_0$$ gives

$$
|X_n-X_0|
\leq
\sqrt{
32\left(\sum_{i=0}^{n-1}c_i^2\right)
\log\frac{2}{\delta}
}
+
\sum_{i=0}^{n-1}b_i\sqrt r
$$

on the intersection of the Azuma event and $$\mathcal{G}$$.

## 8. Probability of the good event

It remains to see why the failure probability contains $$2n\sqrt r$$.

First, by Markov's inequality applied to the conditional probability $$p_i=\mathbb{P}(F_i\mid\mathcal{F}_i)$$,

$$
\mathbb{P}(p_i\geq \sqrt r)
\leq
\frac{\mathbb{E}[p_i]}{\sqrt r}.
$$

By the tower property,

$$
\mathbb{E}[p_i]
=
\mathbb{E}[\mathbb{P}(F_i\mid\mathcal{F}_i)]
=
\mathbb{P}(F_i)
\leq r.
$$

Therefore,

$$
\mathbb{P}(p_i\geq \sqrt r)
\leq \sqrt r.
$$

A union bound over $$i=0,\ldots,n-1$$ gives probability at most $$n\sqrt r$$ that termination occurs.

Second, in the non-terminated case, $$p_i<\sqrt r$$. The probability that a bad event $$F_i$$ occurs under this condition is at most $$\sqrt r$$ at each step. Another union bound gives an additional failure probability at most $$n\sqrt r$$.

Hence

$$
\mathbb{P}(\mathcal{G}^c)
\leq 2n\sqrt r.
$$

Combining this with the Azuma failure probability $$\delta$$ gives the final probability

$$
1-\delta-2n\sqrt r.
$$

## 9. Why this is useful in robust RL

Suppose a stochastic approximation proof produces a martingale term whose increment has the form

$$
\Delta_i = \text{weight}_i \times \text{noise}_i.
$$

Because of an adaptive threshold, one may only be able to prove a crude deterministic bound

$$
|\Delta_i|\leq b_i,
$$

where $$b_i$$ grows polynomially with the horizon. Standard Azuma-Hoeffding would then produce a bound involving $$\sqrt{\sum_i b_i^2}$$, which may be far too large.

However, if a good event implies

$$
|\Delta_i|\leq c_i
$$

with high probability, where $$c_i$$ is of the correct constant order, the refined inequality gives a near-optimal martingale bound depending primarily on $$c_i$$. The rare failures contribute only through

$$
\sum_i b_i\sqrt r.
$$

Thus, by choosing the failure probability parameter $$r$$ sufficiently small, the rare-event correction becomes negligible.

This is exactly the conceptual role of the refined Azuma-Hoeffding inequality in reward-agnostic robust Q-learning: it allows the proof to exploit the fact that the iterates are typically well-controlled, even though their deterministic envelope is much larger.

## 10. Summary

The standard Azuma-Hoeffding inequality says:

$$
\text{deterministic bounded increments}
\quad \Longrightarrow \quad
\text{sub-Gaussian martingale concentration}.
$$

The refined high-probability version says:

$$
\text{small increments with high probability}
+
\text{crude deterministic increments always}
\quad \Longrightarrow \quad
\text{almost the same concentration as if increments were small}.
$$

This distinction is not cosmetic. In adaptive RL analyses, using the crude deterministic bound can destroy the rate, while using the refined inequality can preserve the correct finite-time behavior.

## References

- E. Shamir and J. Spencer. "Sharp concentration of the chromatic number on random graphs $$G_{n,p}$$." *Combinatorica*, 7(1):121--129, 1987.
- S. Maity and A. Mitra. "Corruption-Tolerant Asynchronous Q-learning with Near-Optimal Rates." ICML, 2026.
- K. Azuma. "Weighted sums of certain dependent random variables." *Tohoku Mathematical Journal*, 1967.
- W. Hoeffding. "Probability inequalities for sums of bounded random variables." *Journal of the American Statistical Association*, 1963.
