---
title: "When Rewards Lie: The Mathematics of Robust Q-Learning"
date: 2026-05-13
categories: [rl-blogs]
tags: [reinforcement-learning, q-learning, robust-statistics, stochastic-approximation, adversarial-rl]
math: true
---

In classical reinforcement learning, the reward is the learner's compass.  
The agent takes an action, receives a reward, and updates its belief about which actions are good.

But what if the compass lies?

Not just through ordinary noise, but through adversarial corruption. Suppose a small fraction of the rewards are replaced by arbitrary values. These corruptions may be huge, may depend on the past, and may be chosen specifically to mislead the learner.

This post develops the mathematics of robust \(Q\)-Learning under corrupted rewards. The story has three layers:

1. vanilla \(Q\)-Learning can fail catastrophically under reward corruption;
2. robust mean estimation can repair the Bellman update;
3. in asynchronous online learning, coverage and mixing determine the final finite-time rate.

The guiding principle is:

$$
\boxed{
\text{Do not trust raw rewards. Robustly estimate their means before using them in Bellman updates.}
}
$$

---

# 1. Discounted MDPs and the Bellman Fixed Point

Consider a finite discounted Markov decision process

$$
\mathcal M = (\mathcal S,\mathcal A,P,R,\gamma),
$$

where \(\mathcal S\) is a finite state space, \(\mathcal A\) is a finite action space, \(P(\cdot\mid s,a)\) is the transition kernel, \(R(s,a)\) is the mean reward, and \(\gamma\in(0,1)\) is the discount factor.

When the learner is at state \(s\) and takes action \(a\), the next state is sampled as

$$
s' \sim P(\cdot\mid s,a),
$$

and the learner receives a random reward

$$
r(s,a)
$$

satisfying

$$
\mathbb E[r(s,a)] = R(s,a).
$$

For a policy \(\pi\), the state-action value function is

$$
Q^\pi(s,a)
=
\mathbb E_\pi
\left[
\sum_{t=0}^\infty \gamma^t r(s_t,a_t)
\;\middle|\;
s_0=s,\ a_0=a
\right].
$$

The optimal \(Q\)-function is

$$
Q^\star(s,a)
=
\sup_\pi Q^\pi(s,a).
$$

Once \(Q^\star\) is known, an optimal policy is obtained by greedification:

$$
\pi^\star(s)
\in
\arg\max_{a\in\mathcal A} Q^\star(s,a).
$$

The central object is the Bellman optimality operator

$$
\mathcal T^\star:\mathbb R^{|\mathcal S||\mathcal A|}
\to
\mathbb R^{|\mathcal S||\mathcal A|},
$$

defined coordinate-wise as

$$
(\mathcal T^\star Q)(s,a)
=
R(s,a)
+
\gamma
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'\in\mathcal A} Q(s',a')
\right].
$$

The optimal \(Q\)-function is the unique fixed point:

$$
Q^\star = \mathcal T^\star Q^\star.
$$

Moreover, \(\mathcal T^\star\) is a \(\gamma\)-contraction in the sup norm:

$$
\|\mathcal T^\star Q_1-\mathcal T^\star Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty.
$$

This contraction is the mathematical reason \(Q\)-Learning works. If we could apply \(\mathcal T^\star\) exactly, then the deterministic recursion

$$
Q_{t+1}=\mathcal T^\star Q_t
$$

would converge geometrically to \(Q^\star\).

But in reinforcement learning, \(P\) and \(R\) are unknown. We only see samples.

---

# 2. Vanilla \(Q\)-Learning as a Noisy Bellman Iteration

In the synchronous sampling model, at each iteration \(t\), the learner obtains, for every \((s,a)\),

$$
s_t(s,a)\sim P(\cdot\mid s,a),
\qquad
r_t(s,a)\sim \mathcal R(s,a),
$$

where \(\mathbb E[r_t(s,a)] = R(s,a)\).

Define the empirical Bellman operator

$$
(\mathcal T_t Q)(s,a)
=
r_t(s,a)
+
\gamma
\max_{a'\in\mathcal A} Q(s_t(s,a),a').
$$

Then

$$
\mathbb E[\mathcal T_t Q \mid Q]
=
\mathcal T^\star Q.
$$

The synchronous \(Q\)-Learning recursion is

$$
Q_{t+1}
=
(1-\alpha_t)Q_t
+
\alpha_t \mathcal T_t Q_t.
$$

Equivalently,

$$
Q_{t+1}
=
Q_t
+
\alpha_t
\left(
\mathcal T_t Q_t-Q_t
\right).
$$

Thus, vanilla \(Q\)-Learning is stochastic approximation toward the Bellman fixed point.

To see the noise structure, write

$$
\mathcal T_t Q_t
=
\mathcal T^\star Q_t
+
W_t,
$$

where

$$
W_t
=
\mathcal T_t Q_t-\mathcal T^\star Q_t.
$$

Then

$$
\mathbb E[W_t\mid \mathcal F_t]=0.
$$

So the recursion becomes

$$
Q_{t+1}
=
(1-\alpha_t)Q_t
+
\alpha_t \mathcal T^\star Q_t
+
\alpha_t W_t.
$$

The term \(\mathcal T^\star Q_t\) contracts toward \(Q^\star\), while \(W_t\) is martingale noise.

This is the clean story.

The corrupted story is different.

---

# 3. Reward Corruption Model

Suppose the learner does not observe \(r_t(s,a)\). Instead, it observes

$$
y_t(s,a).
$$

A Huber-style corruption model is

$$
y_t(s,a)
=
(1-Y_t)r_t(s,a)+Y_t z_t(s,a),
$$

where

$$
Y_t\sim \mathrm{Bernoulli}(\varepsilon).
$$

With probability \(1-\varepsilon\), the reward is clean.  
With probability \(\varepsilon\), the reward is adversarial.

The corruption signal \(z_t(s,a)\) can be arbitrary. It can be unbounded, history-dependent, and chosen with knowledge of the learner.

Equivalently,

$$
y_t(s,a)
\sim
(1-\varepsilon)\mathcal R(s,a)
+
\varepsilon \mathcal Q_t(s,a),
$$

where \(\mathcal Q_t(s,a)\) is an adversarial distribution.

In the stronger contamination model, an adversary may corrupt an \(\varepsilon\)-fraction of the reward observations over time. The key common feature is:

$$
\boxed{
\text{A small fraction of rewards can be arbitrary.}
}
$$

This is exactly the setting where the empirical mean is fragile.

---

# 4. Why Vanilla \(Q\)-Learning Fails

Vanilla \(Q\)-Learning plugs the observed reward directly into the Bellman target. Under corruption, the empirical operator becomes

$$
(\widetilde{\mathcal T}_t Q)(s,a)
=
y_t(s,a)
+
\gamma
\max_{a'}Q(s_t(s,a),a').
$$

Taking expectation under a fixed Huber model gives

$$
\mathbb E[y_t(s,a)]
=
(1-\varepsilon)R(s,a)
+
\varepsilon C(s,a),
$$

where

$$
C(s,a)=\mathbb E[z_t(s,a)]
$$

if the adversarial mean exists.

Define the corrupted reward function

$$
\widetilde R_c(s,a)
=
(1-\varepsilon)R(s,a)+\varepsilon C(s,a).
$$

Then vanilla \(Q\)-Learning no longer tracks \(\mathcal T^\star\). It tracks the corrupted Bellman operator

$$
(\widetilde{\mathcal T}_c^\star Q)(s,a)
=
\widetilde R_c(s,a)
+
\gamma
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q(s',a')
\right].
$$

Let \(\widetilde Q_c^\star\) be its fixed point:

$$
\widetilde Q_c^\star
=
\widetilde{\mathcal T}_c^\star \widetilde Q_c^\star.
$$

Then vanilla \(Q\)-Learning converges, but to \(\widetilde Q_c^\star\), not to \(Q^\star\).

This distinction is crucial.

The Bellman contraction is not broken.  
The learner simply contracts toward the wrong fixed point.

---

# 5. Bellman Perturbation Calculation

Let

$$
\Delta_R(s,a)
=
\widetilde R_c(s,a)-R(s,a).
$$

Then for any \(Q\),

$$
(\widetilde{\mathcal T}_c^\star Q)(s,a)
-
(\mathcal T^\star Q)(s,a)
=
\Delta_R(s,a).
$$

Thus,

$$
\|\widetilde{\mathcal T}_c^\star Q-\mathcal T^\star Q\|_\infty
=
\|\Delta_R\|_\infty.
$$

Now compare the two fixed points:

$$
Q^\star=\mathcal T^\star Q^\star,
\qquad
\widetilde Q_c^\star=\widetilde{\mathcal T}_c^\star \widetilde Q_c^\star.
$$

Then

$$
\begin{aligned}
\|\widetilde Q_c^\star-Q^\star\|_\infty
&=
\|\widetilde{\mathcal T}_c^\star \widetilde Q_c^\star
-
\mathcal T^\star Q^\star\|_\infty \\
&\le
\|\widetilde{\mathcal T}_c^\star \widetilde Q_c^\star
-
\mathcal T^\star \widetilde Q_c^\star\|_\infty
+
\|\mathcal T^\star \widetilde Q_c^\star
-
\mathcal T^\star Q^\star\|_\infty \\
&\le
\|\Delta_R\|_\infty
+
\gamma
\|\widetilde Q_c^\star-Q^\star\|_\infty.
\end{aligned}
$$

Rearranging gives

$$
\|\widetilde Q_c^\star-Q^\star\|_\infty
\le
\frac{\|\Delta_R\|_\infty}{1-\gamma}.
$$

This upper bound also reveals the danger. If the adversary can make \(\|\Delta_R\|_\infty\) arbitrarily large, then the fixed-point error can be arbitrarily large.

Even small \(\varepsilon\) does not save us, because

$$
\Delta_R(s,a)
=
\varepsilon(C(s,a)-R(s,a)),
$$

and \(C(s,a)\) can be unbounded.

Therefore,

$$
\boxed{
\text{Small corruption fraction does not imply small bias when corruption magnitudes are unbounded.}
}
$$

---

# 6. A Simple Action-Flipping Example

Consider a state \(s=1\) with two actions \(L\) and \(R\). Suppose the clean immediate rewards are

$$
R(1,L)=d,
\qquad
R(1,R)=-d,
$$

where \(d>0\). So action \(L\) is better.

Now suppose the adversary corrupts only rewards at state \(1\):

$$
y(1,L)
=
\begin{cases}
d, & \text{with probability }1-\varepsilon,\\
-C, & \text{with probability }\varepsilon,
\end{cases}
$$

and

$$
y(1,R)
=
\begin{cases}
-d, & \text{with probability }1-\varepsilon,\\
C, & \text{with probability }\varepsilon.
\end{cases}
$$

Then

$$
\widetilde R_c(1,L)
=
(1-\varepsilon)d-\varepsilon C,
$$

and

$$
\widetilde R_c(1,R)
=
-(1-\varepsilon)d+\varepsilon C.
$$

The corrupted model prefers \(R\) over \(L\) whenever

$$
-(1-\varepsilon)d+\varepsilon C
>
(1-\varepsilon)d-\varepsilon C.
$$

Equivalently,

$$
2\varepsilon C > 2(1-\varepsilon)d,
$$

or

$$
C>\frac{1-\varepsilon}{\varepsilon}d.
$$

Thus, for any small \(\varepsilon>0\), the adversary can choose \(C\) large enough to flip the optimal action.

A sharper construction chooses

$$
C
=
\frac{(2-\varepsilon)d+\kappa}{\varepsilon},
$$

where \(\kappa>0\). Then the fixed-point gap can be made of order

$$
\|\widetilde Q_c^\star-Q^\star\|_\infty
=
2d+\kappa.
$$

Since \(\kappa\) is arbitrary,

$$
\boxed{
\text{vanilla \(Q\)-Learning can suffer arbitrarily large error.}
}
$$

The reason is not mysterious: the sample mean is non-robust.

---

# 7. Robust Mean Estimation: The Statistical Repair

The reward mean \(R(s,a)\) is the weak point. So we replace the empirical mean by a robust estimator.

Suppose we observe corrupted samples

$$
X_1,\dots,X_M
$$

of a scalar random variable \(X\) with mean \(\mu\) and variance \(\sigma^2\). An \(\varepsilon\)-fraction of the samples may be arbitrary.

A robust trimmed estimator proceeds as follows.

Split the data into two halves:

$$
D_1=\{X_1,\dots,X_{M/2}\},
\qquad
D_2=\{X_{M/2+1},\dots,X_M\}.
$$

Use \(D_1\) to compute empirical quantile cutoffs \(\alpha\) and \(\beta\). Define the clipping map

$$
\phi_{\alpha,\beta}(x)
=
\begin{cases}
\beta, & x>\beta,\\
x, & \alpha\le x\le \beta,\\
\alpha, & x<\alpha.
\end{cases}
$$

Then estimate

$$
\widehat \mu
=
\frac{2}{M}
\sum_{X_i\in D_2}
\phi_{\alpha,\beta}(X_i).
$$

The first half estimates a typical interval.  
The second half is averaged after clipping to that interval.

A representative robust mean guarantee is

$$
|\widehat\mu-\mu|
\le
C\sigma
\left(
\sqrt{\varepsilon}
+
\sqrt{\frac{\log(1/\delta)}{M}}
\right)
$$

with probability at least \(1-\delta\).

This bound is the mathematical heart of robust \(Q\)-Learning.

The term

$$
\sqrt{\frac{\log(1/\delta)}{M}}
$$

is the usual sampling error.

The term

$$
\sqrt{\varepsilon}
$$

is the adversarial contamination price.

Crucially, the bound does not depend on the magnitude of the corrupted samples.

So the adversary can inject values of size \(10^{10}\), but the trimmed estimator only pays through \(\varepsilon\), not through \(10^{10}\).

---

# 8. Robust Empirical Bellman Operator

For every state-action pair \((s,a)\), maintain a reward history

$$
\mathcal D_t(s,a)
=
\{y_0(s,a),y_1(s,a),\dots,y_t(s,a)\}
$$

in the synchronous case.

Define the robust reward estimate

$$
\widehat R_t(s,a)
=
\mathrm{TRIM}(\mathcal D_t(s,a),\varepsilon,\delta_1).
$$

Then construct the robust empirical Bellman operator

$$
(\widehat{\mathcal T}_t Q)(s,a)
=
\widehat R_t(s,a)
+
\gamma
\max_{a'}Q(s_t(s,a),a').
$$

The robust \(Q\)-Learning update is

$$
Q_{t+1}
=
(1-\alpha)Q_t
+
\alpha \widehat{\mathcal T}_t Q_t.
$$

Equivalently,

$$
Q_{t+1}(s,a)
=
(1-\alpha)Q_t(s,a)
+
\alpha
\left[
\widehat R_t(s,a)
+
\gamma
\max_{a'}Q_t(s_t(s,a),a')
\right].
$$

The clean Bellman target uses

$$
R(s,a).
$$

The robust Bellman target uses

$$
\widehat R_t(s,a).
$$

Thus the reward error is

$$
E_t(s,a)
=
\widehat R_t(s,a)-R(s,a).
$$

On the robust mean good event,

$$
\|E_t\|_\infty
\lesssim
\sigma
\left(
\sqrt{\varepsilon}
+
\sqrt{\frac{\log(|\mathcal S||\mathcal A|T/\delta)}{t}}
\right).
$$

This is the first robust Bellman error component.

---

# 9. Why Thresholding Is Needed

The trimmed mean bound is high probability. It does not hold deterministically.

On a rare failure event, \(\widehat R_t(s,a)\) may be very large. In a recursive algorithm, this is dangerous: one extreme update can pollute future \(Q\)-values.

Therefore, robust \(Q\)-Learning adds a threshold.

Define

$$
\widetilde R_t(s,a)
=
\begin{cases}
\widehat R_t(s,a), & |\widehat R_t(s,a)|\le G_t,\\
\mathrm{clip}(\widehat R_t(s,a),[-G_t,G_t]), & |\widehat R_t(s,a)|>G_t.
\end{cases}
$$

In some variants, out-of-threshold estimates are set to zero instead:

$$
\widetilde R_t(s,a)
=
\begin{cases}
\widehat R_t(s,a), & |\widehat R_t(s,a)|\le G_t,\\
0, & |\widehat R_t(s,a)|>G_t.
\end{cases}
$$

The update becomes

$$
Q_{t+1}(s,a)
=
(1-\alpha)Q_t(s,a)
+
\alpha
\left[
\widetilde R_t(s,a)
+
\gamma
\max_{a'}Q_t(s_t(s,a),a')
\right].
$$

The threshold \(G_t\) has to be chosen carefully.

If \(G_t\) is too small, we reject good estimates.  
If \(G_t\) is too large, extreme bad estimates can enter.

The right threshold follows the statistical confidence radius of robust mean estimation.

For bounded rewards with \(|r(s,a)|\le \bar r\), one may choose a threshold of the form

$$
G_t
=
C\bar r
\left(
\sqrt{\varepsilon}
+
\sqrt{\frac{\log(1/\delta_1)}{t}}
\right)
+
\bar r.
$$

For heavy-tailed rewards with finite variance proxy \(\bar\sigma\), the analogous scale is

$$
G_t
=
C\bar\sigma
\left(
\sqrt{\varepsilon}
+
\sqrt{\frac{\log(1/\delta_1)}{t}}
\right)
+
\widetilde \sigma,
$$

where \(\widetilde\sigma\) is a bound on the relevant reward scale.

---

# 10. The Fundamental Error Recursion

Let

$$
\Delta_t = Q_t-Q^\star.
$$

Using the robust update,

$$
Q_{t+1}
=
(1-\alpha)Q_t
+
\alpha \widehat{\mathcal T}_t Q_t.
$$

Subtract \(Q^\star=\mathcal T^\star Q^\star\):

$$
\Delta_{t+1}
=
(1-\alpha)\Delta_t
+
\alpha
\left(
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q^\star
\right).
$$

Add and subtract \(\mathcal T^\star Q_t\):

$$
\Delta_{t+1}
=
(1-\alpha)\Delta_t
+
\alpha
\left(
\mathcal T^\star Q_t-\mathcal T^\star Q^\star
\right)
+
\alpha
\left(
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t
\right).
$$

Taking sup norms and using contraction,

$$
\|\Delta_{t+1}\|_\infty
\le
(1-\alpha)\|\Delta_t\|_\infty
+
\alpha\gamma\|\Delta_t\|_\infty
+
\alpha
\|\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t\|_\infty.
$$

Thus,

$$
\|\Delta_{t+1}\|_\infty
\le
(1-\alpha(1-\gamma))\|\Delta_t\|_\infty
+
\alpha \mathcal E_t,
$$

where

$$
\mathcal E_t
=
\|\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t\|_\infty.
$$

Unrolling gives

$$
\|\Delta_T\|_\infty
\le
(1-\alpha(1-\gamma))^T\|\Delta_0\|_\infty
+
\alpha
\sum_{t=0}^{T-1}
(1-\alpha(1-\gamma))^{T-1-t}\mathcal E_t.
$$

This is the master inequality.

Everything reduces to bounding \(\mathcal E_t\).

---

# 11. Decomposing the Bellman Error

The robust empirical Bellman error is

$$
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t.
$$

For each \((s,a)\),

$$
\begin{aligned}
(\widehat{\mathcal T}_t Q_t)(s,a)
-
(\mathcal T^\star Q_t)(s,a)
&=
\widetilde R_t(s,a)-R(s,a) \\
&\quad+
\gamma
\left[
\max_{a'}Q_t(s_t(s,a),a')
-
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\max_{a'}Q_t(s',a')
\right].
\end{aligned}
$$

Define

$$
E_t(s,a)=\widetilde R_t(s,a)-R(s,a),
$$

and

$$
W_t(s,a)
=
\gamma
\left[
\max_{a'}Q_t(s_t(s,a),a')
-
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\max_{a'}Q_t(s',a')
\right].
$$

Then

$$
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t
=
E_t+W_t.
$$

So

$$
\mathcal E_t
\le
\|E_t\|_\infty+\|W_t\|_\infty.
$$

The term \(E_t\) is the robust reward estimation error.  
The term \(W_t\) is transition sampling noise.

The robust proof has two separate jobs:

1. control \(E_t\) using trimmed mean estimation and thresholding;
2. control \(W_t\) using martingale concentration and boundedness of \(Q_t\).

---

# 12. Synchronous Robust Rate

In the synchronous setting, every state-action pair receives one reward sample per iteration. Therefore

$$
N_t(s,a)=t
$$

for all \((s,a)\).

The robust reward estimation guarantee gives

$$
\|E_t\|_\infty
\lesssim
\bar r
\left(
\sqrt{\varepsilon}
+
\sqrt{
\frac{\log(|\mathcal S||\mathcal A|T/\delta)}{t}
}
\right).
$$

The transition noise term contributes the usual \(Q\)-Learning sampling rate. After summing the weighted martingale differences and choosing an appropriate constant step size

$$
\alpha
\asymp
\frac{\log T}{(1-\gamma)T},
$$

one obtains a high-probability bound of the form

$$
\|Q_T-Q^\star\|_\infty
\le
\frac{\|Q_0-Q^\star\|_\infty}{T}
+
\widetilde O
\left(
\frac{\bar r}
{(1-\gamma)^{5/2}\sqrt T}
\right)
+
O
\left(
\frac{\bar r\sqrt\varepsilon}
{1-\gamma}
\right).
$$

This expression is worth reading term by term.

The first term,

$$
\frac{\|Q_0-Q^\star\|_\infty}{T},
$$

is initialization bias.

The second term,

$$
\widetilde O
\left(
\frac{\bar r}
{(1-\gamma)^{5/2}\sqrt T}
\right),
$$

is the clean finite-sample statistical rate.

The third term,

$$
O
\left(
\frac{\bar r\sqrt\varepsilon}
{1-\gamma}
\right),
$$

is the adversarial corruption floor.

Thus,

$$
\boxed{
\text{robust \(Q\)-Learning recovers the clean rate up to an additive \(\sqrt\varepsilon\) term.}
}
$$

---

# 13. Why the Corruption Term Has the Form \(\sqrt\varepsilon/(1-\gamma)\)

The \(\sqrt\varepsilon\) part comes from robust mean estimation:

$$
|\widehat R(s,a)-R(s,a)|
\lesssim
\sigma\sqrt\varepsilon.
$$

The factor \(1/(1-\gamma)\) comes from Bellman amplification.

Indeed, if two MDPs differ only in their reward functions, with

$$
\|R_1-R_2\|_\infty\le \Delta,
$$

then their optimal \(Q\)-functions satisfy

$$
\|Q_1^\star-Q_2^\star\|_\infty
\le
\frac{\Delta}{1-\gamma}.
$$

Proof:

$$
\begin{aligned}
\|Q_1^\star-Q_2^\star\|_\infty
&=
\|\mathcal T_1^\star Q_1^\star-\mathcal T_2^\star Q_2^\star\|_\infty\\
&\le
\|\mathcal T_1^\star Q_1^\star-\mathcal T_1^\star Q_2^\star\|_\infty
+
\|\mathcal T_1^\star Q_2^\star-\mathcal T_2^\star Q_2^\star\|_\infty\\
&\le
\gamma\|Q_1^\star-Q_2^\star\|_\infty+\Delta.
\end{aligned}
$$

Rearranging,

$$
\|Q_1^\star-Q_2^\star\|_\infty
\le
\frac{\Delta}{1-\gamma}.
$$

Taking

$$
\Delta\asymp \sigma\sqrt\varepsilon,
$$

we obtain

$$
\frac{\sigma\sqrt\varepsilon}{1-\gamma}.
$$

Thus the corruption error has exactly the form one should expect.

---

# 14. Moving to Asynchronous \(Q\)-Learning

The synchronous setting is idealized. In ordinary online \(Q\)-Learning, the learner observes only one transition at a time:

$$
(s_t,a_t,s_{t+1},y_t).
$$

Only the visited state-action pair \((s_t,a_t)\) is updated.

The asynchronous update is

$$
Q_{t+1}(s,a)
=
Q_t(s,a),
\qquad
(s,a)\ne(s_t,a_t),
$$

and

$$
Q_{t+1}(s_t,a_t)
=
(1-\alpha)Q_t(s_t,a_t)
+
\alpha
\left[
\widetilde R_t(s_t,a_t)
+
\gamma
\max_{a'}Q_t(s_{t+1},a')
\right].
$$

The challenge is that the number of samples per state-action pair is now random.

Define

$$
N_t(s,a)
=
\sum_{k=0}^{t}
\mathbf 1\{(s_k,a_k)=(s,a)\}.
$$

If data are sampled from the stationary behavior distribution, define

$$
\lambda(s,a)=\pi(s)\mu(a\mid s),
$$

and

$$
\lambda_{\min}
=
\min_{(s,a)\in\mathcal S\times\mathcal A}
\lambda(s,a).
$$

Then

$$
\mathbb E[N_t(s,a)]
=
\lambda(s,a)t.
$$

The least visited state-action pair controls the sample complexity.

---

# 15. Visit Count Concentration

To use robust mean estimation, we need enough samples for every \((s,a)\). In the i.i.d. asynchronous setting,

$$
N_t(s,a)
\sim
\mathrm{Binomial}(t,\lambda(s,a)).
$$

A Bernstein-style argument gives, with high probability,

$$
N_t(s,a)
\ge
\frac34 \lambda(s,a)t
\ge
\frac34 \lambda_{\min}t
$$

simultaneously for all \((s,a)\) and all \(t\) beyond a burn-in time

$$
\overline T
\asymp
\frac{1}{\lambda_{\min}}
\log\left(
\frac{|\mathcal S||\mathcal A|T}{\delta}
\right).
$$

This burn-in is conceptually unavoidable.

You cannot robustly estimate \(R(s,a)\) before you have seen \((s,a)\) enough times.

After burn-in,

$$
N_t(s,a)
\gtrsim
\lambda_{\min}t.
$$

Therefore the robust reward error becomes

$$
|\widetilde R_t(s,a)-R(s,a)|
\lesssim
\bar\sigma
\left(
\sqrt\varepsilon
+
\sqrt{
\frac{\log(|\mathcal S||\mathcal A|T/\delta)}
{\lambda_{\min}t}
}
\right).
$$

This is the mathematical reason \(\lambda_{\min}\) appears in the asynchronous rate.

---

# 16. Robust Async-\(Q\): Algorithmic Form

For every \((s,a)\), maintain a history

$$
\mathcal D_t(s,a)
=
\{y_k(s_k,a_k): (s_k,a_k)=(s,a),\ 0\le k\le t\}.
$$

When \((s_t,a_t)\) is visited:

1. append \(y_t(s_t,a_t)\) to \(\mathcal D_t(s_t,a_t)\);
2. compute

$$
\overline r_t(s_t,a_t)
=
\mathrm{TRIM}(\mathcal D_t(s_t,a_t),\varepsilon,\delta_1);
$$

3. threshold:

$$
\widetilde r_t(s_t,a_t)
=
\begin{cases}
\overline r_t(s_t,a_t),
&
|\overline r_t(s_t,a_t)|\le G_t,\\
0,
&
|\overline r_t(s_t,a_t)|>G_t;
\end{cases}
$$

4. update

$$
Q_{t+1}(s_t,a_t)
=
(1-\alpha)Q_t(s_t,a_t)
+
\alpha
\left[
\widetilde r_t(s_t,a_t)
+
\gamma
\max_{a'}Q_t(s_{t+1},a')
\right].
$$

The threshold has the form

$$
G_t
=
C\bar\sigma
\left(
\sqrt\varepsilon
+
\sqrt{
\frac{\log(1/\delta_1)}
{\lambda_{\min}t}
}
\right)
+
\widetilde\sigma,
\qquad t>\overline T.
$$

For \(t\le \overline T\), the algorithm may use a conservative default threshold.

The key event is:

$$
\mathcal E_{\mathrm{good}}
=
\left\{
\forall t>\overline T,\ \forall(s,a):
N_t(s,a)\gtrsim \lambda_{\min}t
\ \text{and}\
|\overline r_t(s,a)-R(s,a)|
\lesssim
\bar\sigma
\left(
\sqrt\varepsilon+
\sqrt{
\frac{\log(1/\delta_1)}
{\lambda_{\min}t}
}
\right)
\right\}.
$$

On this event, no good estimate is rejected after burn-in, and all accepted reward estimates are accurate.

---

# 17. Asynchronous Error Recursion

In asynchronous \(Q\)-Learning, the update is coordinate-wise. Let

$$
D_t
=
\mathrm{diag}
\left(
\mathbf 1\{(s_t,a_t)=(s,a)\}
\right)_{(s,a)}.
$$

Then the update can be written vectorially as

$$
Q_{t+1}
=
Q_t
+
\alpha D_t
\left(
\widehat{\mathcal T}_t Q_t-Q_t
\right).
$$

Subtract \(Q^\star\):

$$
\Delta_{t+1}
=
(I-\alpha D_t)\Delta_t
+
\alpha D_t
\left(
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q^\star
\right).
$$

Add and subtract \(\mathcal T^\star Q_t\):

$$
\Delta_{t+1}
=
(I-\alpha D_t)\Delta_t
+
\alpha D_t
\left(
\mathcal T^\star Q_t-\mathcal T^\star Q^\star
\right)
+
\alpha D_t
\left(
\widehat{\mathcal T}_t Q_t-\mathcal T^\star Q_t
\right).
$$

The difficulty is that \(D_t\) is random and only updates one coordinate. In expectation,

$$
\mathbb E[D_t]
=
D
=
\mathrm{diag}(\lambda(s,a)).
$$

The smallest diagonal entry is \(\lambda_{\min}\). This means the contraction is weakened from roughly

$$
1-\alpha(1-\gamma)
$$

to roughly

$$
1-\alpha\lambda_{\min}(1-\gamma).
$$

This explains why the step size is chosen on the scale

$$
\alpha
\asymp
\frac{\log T}
{\lambda_{\min}(1-\gamma)T}.
$$

With this choice,

$$
\left(1-\alpha\lambda_{\min}(1-\gamma)\right)^T
\approx
\frac1T.
$$

So the initialization bias becomes

$$
\frac{\|\Delta_0\|_\infty}{T}.
$$

---

# 18. Robust Async-\(Q\) Finite-Time Rate

The high-probability bound has the schematic form

$$
\|Q_T-Q^\star\|_\infty
\le
\frac{\|Q_0-Q^\star\|_\infty}{T}
+
\widetilde O
\left(
\frac{\widetilde\sigma}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}\sqrt T}
\right)
+
O
\left(
\frac{\bar\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

This expression contains three effects.

The first term is initialization:

$$
\frac{\|Q_0-Q^\star\|_\infty}{T}.
$$

The second term is clean asynchronous sampling noise:

$$
\widetilde O
\left(
\frac{\widetilde\sigma}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}\sqrt T}
\right).
$$

The third term is corruption:

$$
O
\left(
\frac{\bar\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

Compared with synchronous robust \(Q\)-Learning, the asynchronous rate has extra dependence on \(\lambda_{\min}\).

This is not merely proof looseness. In asynchronous learning, a rare state-action pair receives only about

$$
\lambda_{\min}T
$$

samples. Reward estimation for that pair is harder, and the adversary can exploit the least-covered coordinate.

Thus,

$$
\boxed{
\lambda_{\min}
\text{ is the coverage bottleneck of asynchronous robust \(Q\)-Learning.}
}
$$

---

# 19. Reward-Agnostic Robust Async-\(Q\)

The threshold \(G_t\) requires knowledge of reward scale parameters, such as \(\bar\sigma\) or \(\widetilde\sigma\). But in practice these may be unknown.

A reward-agnostic version replaces the unknown scale by a growing proxy

$$
m(t)=t^p,
\qquad p\ge 1.
$$

Then the threshold becomes

$$
\widetilde G_t
=
C m(t)
\left(
\sqrt\varepsilon
+
\sqrt{
\frac{\log(1/\delta_1)}
{\lambda_{\min}t}
}
\right)
+
m(t).
$$

The intuition is simple.

Since the true reward scale \(\widetilde\sigma\) is fixed, eventually

$$
m(t)\ge \widetilde\sigma.
$$

After that point, \(\widetilde G_t\) dominates the ideal known-scale threshold. Thus, the algorithm eventually behaves like the known-scale robust algorithm.

The price is a more delicate proof.

If one only uses the deterministic crude bound

$$
\|Q_t\|_\infty
\lesssim
\frac{T^p}{1-\gamma},
$$

then Azuma-Hoeffding gives a vacuous concentration bound.

The refined analysis instead uses two bounds:

1. a crude deterministic bound, valid always;
2. a sharper high-probability bound, valid on the good event.

This is where refined almost-martingale concentration becomes important.

The resulting reward-agnostic rate is of the form

$$
\|Q_T-Q^\star\|_\infty
\le
\frac{\|Q_0-Q^\star\|_\infty}{T}
+
\widetilde O
\left(
\frac{\widetilde\sigma^{1+1/(2p)}}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}\sqrt T}
\right)
+
O
\left(
\frac{\bar\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

As \(p\) grows,

$$
\widetilde\sigma^{1+1/(2p)}
\to
\widetilde\sigma.
$$

So the reward-agnostic version nearly recovers the known-scale rate.

---

# 20. Markovian Sampling

The i.i.d. asynchronous model is useful for intuition, but real online \(Q\)-Learning follows a single trajectory:

$$
s_0,a_0,s_1,a_1,s_2,a_2,\dots
$$

The data are time-correlated.

Let the behavior policy \(\mu\) induce an ergodic Markov chain with stationary distribution \(\pi\). Define

$$
\lambda(s,a)=\pi(s)\mu(a\mid s).
$$

The Markovian difficulty is that the indicators

$$
\mathbf 1\{(s_t,a_t)=(s,a)\}
$$

are no longer independent over time.

To control this, one uses mixing.

Let

$$
d_{\mathrm{mix}}(t)
=
\sup_z
D_{\mathrm{TV}}
\left(
\mathbb P(Z_t\in\cdot\mid Z_0=z),\rho
\right),
$$

and define the mixing time

$$
\bar\tau
=
\inf\{t:d_{\mathrm{mix}}(t)\le 1/4\}.
$$

A standard consequence is

$$
d_{\mathrm{mix}}(\ell\bar\tau)\le 2^{-\ell}.
$$

The algorithm can update on a thinned subsequence, for example every

$$
\tau
\asymp
\bar\tau\log(T/\delta)
$$

steps.

This makes the retained samples approximately independent. The final Markovian rate resembles the asynchronous i.i.d. rate, but the clean statistical term is inflated by mixing-time effects.

The intuition is:

$$
\text{slow mixing reduces the number of effectively independent samples.}
$$

However, the robust reward-estimation principle remains unchanged.

---

# 21. Lower Bound: Why \(\sqrt\varepsilon\) Is Unavoidable

A natural question is whether the corruption term can be improved.

Can one replace

$$
\sqrt\varepsilon
$$

by

$$
\varepsilon?
$$

In general, no.

The lower-bound intuition is inherited from robust mean estimation.

Consider the simplest possible MDP: one state and one action. Then the Bellman equation is

$$
Q^\star = R+\gamma Q^\star.
$$

Therefore,

$$
Q^\star=\frac{R}{1-\gamma}.
$$

Estimating \(Q^\star\) is exactly equivalent to estimating the reward mean \(R\), scaled by \(1/(1-\gamma)\).

Now construct two reward distributions with means separated by

$$
\Delta
\asymp
\sigma\sqrt\varepsilon,
$$

but whose \(\varepsilon\)-contaminated versions are statistically indistinguishable.

If no estimator can reliably distinguish the two corrupted observation models, then no estimator can reliably distinguish the two optimal \(Q\)-values.

The \(Q\)-gap is

$$
\frac{\Delta}{1-\gamma}
\asymp
\frac{\sigma\sqrt\varepsilon}{1-\gamma}.
$$

Therefore any estimator must suffer error at least

$$
\Omega
\left(
\frac{\sigma\sqrt\varepsilon}{1-\gamma}
\right).
$$

This matches the upper bound dependence on \(\sqrt\varepsilon\), \(\sigma\), and \(1/(1-\gamma)\).

Thus,

$$
\boxed{
\text{the corruption floor is information-theoretic, not a proof artifact.}
}
$$

---

# 22. The Complete Picture

The mathematical structure can now be summarized in one diagram:

$$
\text{corrupted rewards}
\Rightarrow
\text{biased empirical means}
\Rightarrow
\text{perturbed Bellman operator}
\Rightarrow
\text{wrong fixed point}.
$$

Robust \(Q\)-Learning changes the middle step:

$$
\text{corrupted rewards}
\Rightarrow
\text{robust mean estimator}
\Rightarrow
\text{robust empirical Bellman operator}
\Rightarrow
\text{near-correct fixed point}.
$$

The resulting error has the form

$$
\boxed{
\|Q_T-Q^\star\|_\infty
\lesssim
\text{optimization bias}
+
\text{sampling noise}
+
\text{corruption floor}.
}
$$

In the synchronous case,

$$
\text{sampling noise}
\sim
\widetilde O\left(\frac1{\sqrt T}\right),
\qquad
\text{corruption floor}
\sim
O(\sqrt\varepsilon).
$$

In the asynchronous case,

$$
\text{sampling noise}
\sim
\widetilde O\left(
\frac1{\lambda_{\min}^{3/2}\sqrt T}
\right),
\qquad
\text{corruption floor}
\sim
O\left(
\frac{\sqrt\varepsilon}{\lambda_{\min}}
\right).
$$

In the Markovian case, the clean sampling term also pays a mixing-time price.

---

# 23. Final Takeaways

The main mathematical lessons are:

### 1. Bellman contraction alone is not robustness

The Bellman operator may remain contractive, but under corrupted rewards vanilla \(Q\)-Learning contracts toward a corrupted fixed point.

### 2. Reward estimation is the weak link

The reward mean enters the Bellman operator additively. If the reward mean estimate is corrupted, the entire fixed point shifts.

### 3. Robust statistics repairs the Bellman update

Trimmed mean estimation gives

$$
|\widehat R-R|
\lesssim
\sigma
\left(
\sqrt\varepsilon+
\sqrt{\frac{\log(1/\delta)}{n}}
\right),
$$

which is precisely the kind of guarantee needed inside \(Q\)-Learning.

### 4. Discounting amplifies reward errors

A reward error \(\Delta\) becomes a value error of order

$$
\frac{\Delta}{1-\gamma}.
$$

### 5. Asynchrony introduces coverage dependence

If a state-action pair is rarely visited, robustly estimating its reward is harder. This creates dependence on

$$
\lambda_{\min}
=
\min_{(s,a)}\pi(s)\mu(a\mid s).
$$

### 6. The \(\sqrt\varepsilon\) term is unavoidable

Even in the simplest one-state one-action MDP, robust mean estimation lower bounds imply a \(Q\)-function error floor of order

$$
\frac{\sigma\sqrt\varepsilon}{1-\gamma}.
$$

---

# 24. Closing Thought

Classical \(Q\)-Learning asks:

$$
\text{How can we learn the Bellman fixed point from samples?}
$$

Robust \(Q\)-Learning asks a harder question:

$$
\text{How can we learn the Bellman fixed point from samples we do not fully trust?}
$$

The answer is to combine three ideas:

1. Bellman contraction,
2. robust mean estimation,
3. finite-time stochastic approximation.

The resulting theory is reassuring but honest.

It says that adversarial reward corruption does not destroy learning.  
But it also says that corruption leaves an unavoidable statistical scar.

That scar is

$$
\Theta(\sqrt\varepsilon),
$$

magnified through the Bellman equation by roughly

$$
\frac1{1-\gamma}.
$$

So the right goal is not to pretend corruption disappears.

The right goal is to make the error depend only on how much corruption there is, not on how large the adversarial values are.

That is precisely what robust \(Q\)-Learning achieves.
