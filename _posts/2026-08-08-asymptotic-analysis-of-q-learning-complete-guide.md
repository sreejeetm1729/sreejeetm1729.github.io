---
title: "Asymptotic Analysis of Q-Learning: A Complete Proof-Oriented Guide"
date: 2026-08-08
categories: [rl-blogs]
rl_section: research-papers
tags: [reinforcement-learning, q-learning, stochastic-approximation, asymptotic-analysis, central-limit-theorem, theory]
toc: true
math: true
description: "A complete proof-oriented guide to almost-sure convergence, ODE stability, asymptotic rates, central limit theory, asymptotic covariance, averaging, constant stepsizes, Markovian sampling, and infinite-variance rewards in Q-Learning."
---

## Overview and Roadmap

Finite-time analysis asks how accurately an algorithm estimates the optimal action-value function after a prescribed number of samples. Asymptotic analysis asks a different collection of questions. Does the algorithm converge when it is allowed to run indefinitely? What deterministic dynamics govern that convergence? After convergence has begun, what is the correct scale of the remaining random fluctuations? Do those fluctuations become Gaussian? What changes when the data are Markovian, the stepsize is constant, or the reward has infinite variance?

These questions are related, but they are not interchangeable. Almost-sure convergence alone does not imply a square-root rate. A square-root rate does not by itself imply a central limit theorem. A central limit theorem for independent samples does not automatically survive asynchronous Markovian sampling. Finally, an algorithm may converge almost surely even when its usual Gaussian central limit theorem is false.

This guide develops these distinctions carefully for tabular discounted Q-Learning.

### 1. The common mathematical object

Consider a finite discounted Markov decision process with state space, action space, transition kernel, mean reward function, and discount factor denoted respectively by

$$
\mathcal S,\qquad
\mathcal A,\qquad
P,\qquad
R,\qquad
\gamma\in[0,1).
$$

The Bellman optimality operator is

$$
(\mathcal T Q)(s,a)
=
R(s,a)
+
\gamma
\sum_{s'\in\mathcal S}
P(s'\mid s,a)
\max_{a'\in\mathcal A}Q(s',a').
$$

Its defining property is the sup-norm contraction

$$
\|\mathcal TQ-\mathcal TQ'\|_\infty
\leq
\gamma\|Q-Q'\|_\infty.
$$

Consequently, there is a unique fixed point

$$
Q^*=\mathcal TQ^*.
$$

For a trajectory

$$
(s_0,a_0,r_0,s_1,a_1,r_1,\ldots),
$$

tabular asynchronous Q-Learning performs the update

$$
\begin{aligned}
Q_{t+1}(s_t,a_t)
&=
Q_t(s_t,a_t)
+
\alpha_t(s_t,a_t)
\Bigl(
r_t
+
\gamma\max_{a'}Q_t(s_{t+1},a')
-Q_t(s_t,a_t)
\Bigr),
\\
Q_{t+1}(s,a)
&=Q_t(s,a),
\qquad
(s,a)\neq(s_t,a_t).
\end{aligned}
$$

Every post in the series begins from this recursion, but studies it at a different resolution.

### 2. Four levels of asymptotic analysis

#### 2.1 Almost-sure convergence

The first question is qualitative:

$$
Q_t\longrightarrow Q^*
\qquad\text{almost surely?}
$$

The classical proofs require sufficient exploration, diminishing stepsizes, control of the stochastic noise, and stability of the iterates. The conclusion says that almost every infinite sample path eventually approaches the correct fixed point. It does not specify the distribution of the error at a large but finite time.

#### 2.2 Rate of decay

The next question is quantitative:

$$
\|Q_t-Q^*\|
=
O(t^{-\kappa})
\qquad\text{for which exponent }\kappa?
$$

The exponent can depend on the Bellman contraction, the stepsize constant, and state-action visitation frequencies. In particular, a stepsize proportional to the reciprocal of the global time index need not provide the canonical square-root rate for every discount factor and every sampling distribution.

#### 2.3 Central limit theory

A central limit theorem asks for the distribution of a rescaled error. In its standard form,

$$
\sqrt{t}\,(Q_t-Q^*)
\Rightarrow
\mathcal N(0,\Sigma).
$$

Here, convergence is in distribution and the covariance matrix contains detailed information that a big-O rate discards. It reveals which coordinates are statistically difficult, how transition noise propagates through the optimal dynamics, and how the stepsize or gain matrix affects efficiency.

#### 2.4 Constant-stepsize limits

When the stepsize is fixed, exact convergence is generally not expected. Instead, the Q-Learning iterates form a time-homogeneous Markov chain whose distribution approaches a stationary law:

$$
Q_t
\Rightarrow
Q_\infty^{(\alpha)}.
$$

The stationary law is concentrated near the optimal action-value function when the stepsize is small. Its mean can nevertheless be biased, and averaging reduces variance without automatically removing that bias.

### 3. The stochastic-approximation viewpoint

The central organizing idea is to rewrite Q-Learning as stochastic approximation:

$$
Q_{t+1}
=
Q_t
+
\alpha_t
\Bigl(
h(Q_t)
+
M_{t+1}
+
\varepsilon_{t+1}
\Bigr).
$$

The three terms inside the parentheses have distinct roles.

- The mean field governs the deterministic direction of motion.
- The martingale term represents centered one-step randomness.
- The remainder contains Markovian, asynchronous, or nonlinear approximation errors that must be proved negligible on the relevant scale.

Under stationary asynchronous sampling with state-action occupancy distribution

$$
\nu(s,a)>0,
$$

the idealized mean field takes the form

$$
h(Q)
=
D_\nu(\mathcal TQ-Q),
$$

where

$$
D_\nu
=
\operatorname{diag}
\bigl(\nu(s,a):(s,a)\in\mathcal S\times\mathcal A\bigr).
$$

This formula immediately displays the two forces that control convergence:

- Bellman contraction contributes the factor associated with the discount;
- asynchronous sampling contributes the state-action visitation frequencies.

### 4. Why local linearization matters

To prove convergence, global contraction is usually enough. To obtain a central limit theorem, one needs a local first-order approximation near the fixed point.

Suppose each state has a unique optimal action. Define the optimal greedy policy by

$$
\pi^*(s)
=
\operatorname*{arg\,max}_{a\in\mathcal A}Q^*(s,a).
$$

For a deterministic policy, define the state-action transition operator by

$$
\bigl(P^\pi Q\bigr)(s,a)
=
\sum_{s'\in\mathcal S}
P(s'\mid s,a)
Q(s',\pi(s')).
$$

When the optimal action gaps are positive, the greedy policy does not change under sufficiently small perturbations of the Q-function. Locally,

$$
\mathcal TQ-\mathcal TQ^*
=
\gamma P^{\pi^*}(Q-Q^*).
$$

Therefore, the mean field has the expansion

$$
h(Q)
=
A(Q-Q^*)
+
o(\|Q-Q^*\|),
$$

with

$$
A
=
D_\nu
\bigl(\gamma P^{\pi^*}-I\bigr).
$$

The nonlinear Q-Learning recursion is thus asymptotically governed by a linear stochastic-approximation recursion. The eigenvalues of this matrix determine whether a square-root central limit theorem is possible for the last iterate, while its inverse determines the covariance of efficiently averaged iterates.

If optimal actions are tied, the maximization map is not differentiable at the fixed point. In that case, a Gaussian limit cannot be asserted merely by writing down a formal derivative. One needs directional differentiability, a margin condition, a switching-system argument, or a nonsmooth limit theorem.

### 5. The chapters in this guide

The guide is organized as follows.

1. **Classical convergence of asynchronous Q-Learning.** We derive the stochastic recursion, filtration, martingale noise, Robbins-Monro conditions, and contraction argument underlying the results of Jaakkola, Jordan, and Singh and of Tsitsiklis.

2. **The ODE method and stability.** We explain the Borkar-Meyn scaled-ODE argument, prove global stability of the mean ODE in the sup norm, and separate boundedness from convergence.

3. **Asymptotic convergence rates.** We study the competition between deterministic contraction and accumulated noise and explain the rate transition identified by Szepesvári.

4. **Asymptotic covariance and Zap Q-Learning.** We linearize the recursion, derive the Lyapunov equation for the covariance, and explain why a matrix gain can be asymptotically optimal.

5. **Polyak-Ruppert averaged Q-Learning.** We show why averaging removes delicate gain tuning and derive the covariance formula appearing in the statistical analysis of averaged Q-Learning.

6. **Markovian and asynchronous central limit theorems.** We introduce the Poisson equation, long-run covariance, and the additional remainders caused by a single behavior trajectory.

7. **Constant-stepsize Q-Learning.** We replace pointwise convergence by convergence to a stationary distribution and study bias, variance, averaging, and Richardson-Romberg extrapolation.

8. **Infinite-variance rewards.** We explain which parts of classical convergence survive, why the Gaussian scaling fails, and how stable-law asymptotics may arise.

9. **Decay-to-zero learning rates.** We study horizon-dependent LD2Z and PD2Z schedules, tail averaging, and the recent Gaussian approximation theory.

### 6. Three warnings to remember

First, almost-sure convergence and asymptotic normality use different moment assumptions. Square summability of stepsizes controls finite-variance martingale noise, but a Gaussian central limit theorem requires a separate fluctuation analysis.

Second, the sampling model matters. Synchronous generative samples, independent asynchronous coordinate samples, and a single Markovian trajectory produce different noise covariance matrices and require different decompositions.

Third, the maximization in the Bellman optimality operator is harmless for contraction arguments but central to distributional theory. Global Lipschitz continuity is not the same as local differentiability.

### 7. Core reading list

- J. N. Tsitsiklis, [Asynchronous Stochastic Approximation and Q-Learning](https://link.springer.com/article/10.1023/A%3A1022689125041), 1994.
- T. Jaakkola, M. I. Jordan, and S. P. Singh, [On the Convergence of Stochastic Iterative Dynamic Programming Algorithms](https://direct.mit.edu/neco/article/6/6/1185/5826/On-the-Convergence-of-Stochastic-Iterative-Dynamic), 1994.
- V. S. Borkar and S. P. Meyn, [The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning](https://epubs.siam.org/doi/10.1137/S0363012997331639), 2000.
- C. Szepesvári, [The Asymptotic Convergence-Rate of Q-Learning](https://papers.neurips.cc/paper/1383-the-asymptotic-convergence-rate-of-q-learning), 1997.
- A. M. Devraj and S. P. Meyn, [Fastest Convergence for Q-Learning](https://arxiv.org/abs/1707.03770), 2017.
- B. T. Polyak and A. B. Juditsky, [Acceleration of Stochastic Approximation by Averaging](https://epubs.siam.org/doi/10.1137/0330046), 1992.
- X. Li, W. Yang, J. Liang, Z. Zhang, and M. I. Jordan, [A Statistical Analysis of Polyak-Ruppert Averaged Q-Learning](https://proceedings.mlr.press/v206/li23b.html), 2023.
- G. Fort, [Central Limit Theorems for Stochastic Approximation with Controlled Markov Chain Dynamics](https://www.numdam.org/articles/10.1051/ps/2014013/), 2015.
- Y. Zhang and Q. Xie, [Constant Stepsize Q-Learning: Distributional Convergence, Bias and Extrapolation](https://arxiv.org/abs/2401.13884), 2024.
- S. Bonnerjee, Z. Lou, and W. B. Wu, [Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](https://proceedings.iclr.cc/paper_files/paper/2026/file/b3b52663b1c01ae961895e419a55fb28-Paper-Conference.pdf), ICLR 2026.

---

## Chapter 1: Classical Convergence of Asynchronous Q-Learning

The first theorem one learns about tabular Q-Learning is deceptively simple: if every state-action pair is updated sufficiently often and the stepsizes decay appropriately, then the iterates converge almost surely to the optimal action-value function. The statement hides nearly every important idea in stochastic approximation: filtrations, conditional expectations, martingale noise, asynchronous coordinate updates, contraction, and stability.

This chapter unpacks those ideas carefully.

### 1. Model and Bellman operator

Let the state and action spaces be finite. For each state-action pair, let the conditional mean reward be

$$
R(s,a)
=
\mathbb E[r_t\mid s_t=s,a_t=a].
$$

The transition kernel is

$$
P(s'\mid s,a)
=
\mathbb P(s_{t+1}=s'\mid s_t=s,a_t=a).
$$

The Bellman optimality operator is

$$
(\mathcal TQ)(s,a)
=
R(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q(s',a').
$$

For any two Q-functions,

$$
\begin{aligned}
|\mathcal TQ(s,a)-\mathcal TQ'(s,a)|
&\leq
\gamma
\sum_{s'}P(s'\mid s,a)
\left|
\max_{a'}Q(s',a')-
\max_{a'}Q'(s',a')
\right|
\\
&\leq
\gamma\|Q-Q'\|_\infty.
\end{aligned}
$$

Taking the maximum over all state-action pairs gives

$$
\|\mathcal TQ-\mathcal TQ'\|_\infty
\leq
\gamma\|Q-Q'\|_\infty.
$$

Because the space of finite Q-tables is complete, the Banach fixed-point theorem gives a unique fixed point

$$
Q^*=\mathcal TQ^*.
$$

This deterministic contraction is the anchor of the stochastic proof.

### 2. The asynchronous recursion

Let

$$
x_t=(s_t,a_t)
$$

denote the state-action pair visited at the current time. To avoid relying on inline indexed notation, define the coordinate vector associated with a pair by

$$
e_x(y)
=
\mathbf 1\{x=y\}.
$$

The vector form of asynchronous Q-Learning is

$$
Q_{t+1}
=
Q_t
+
\alpha_t(x_t)e_{x_t}
\Bigl(
r_t
+
\gamma\max_{a'}Q_t(s_{t+1},a')
-Q_t(x_t)
\Bigr).
$$

It is often cleaner to absorb the update indicator into a coordinatewise stepsize:

$$
\widetilde\alpha_t(x)
=
\alpha_t(x)
\mathbf 1\{x_t=x\}.
$$

Then, for every coordinate,

$$
Q_{t+1}(x)
=
Q_t(x)
+
\widetilde\alpha_t(x)
\Bigl(
r_t
+
\gamma\max_{a'}Q_t(s_{t+1},a')
-Q_t(x)
\Bigr),
$$

where the bracketed term is used only when the coordinate is visited.

### 3. Local clocks and stepsizes

Define the number of updates to a fixed coordinate before the current time by

$$
N_t(x)
=
\sum_{k=0}^{t-1}
\mathbf 1\{x_k=x\}.
$$

A standard local-clock stepsize has the form

$$
\alpha_t(x_t)
=
\beta_{N_t(x_t)},
$$

where the deterministic sequence satisfies the Robbins-Monro conditions

$$
\sum_{n=0}^{\infty}\beta_n=\infty,
\qquad
\sum_{n=0}^{\infty}\beta_n^2<\infty,
\qquad
0\leq\beta_n\leq1.
$$

If every state-action pair is visited infinitely often, then every coordinate receives an infinite total amount of learning:

$$
\sum_{t=0}^{\infty}\widetilde\alpha_t(x)=\infty.
$$

At the same time, the total squared learning rate is finite:

$$
\sum_{t=0}^{\infty}\widetilde\alpha_t^2(x)<\infty.
$$

The first series prevents learning from stopping prematurely. The second ensures that finite-variance stochastic noise does not accumulate indefinitely.

### 4. The filtration and the centered noise

Let the information available immediately before observing the current reward and next state be

$$
\mathcal F_t
=
\sigma
\bigl(
Q_0,
s_0,a_0,r_0,
\ldots,
s_t,a_t
\bigr).
$$

The current Q-table and the current state-action pair are measurable with respect to this filtration. Define the random Bellman target

$$
Y_{t+1}
=
r_t
+
\gamma\max_{a'}Q_t(s_{t+1},a').
$$

Under the Markov property,

$$
\mathbb E[Y_{t+1}\mid\mathcal F_t]
=
(\mathcal TQ_t)(x_t).
$$

Define the centered Bellman noise by

$$
M_{t+1}
=
Y_{t+1}
-
(\mathcal TQ_t)(x_t).
$$

Then

$$
\mathbb E[M_{t+1}\mid\mathcal F_t]=0.
$$

This is the martingale-difference property. The update becomes

$$
Q_{t+1}(x_t)
=
Q_t(x_t)
+
\alpha_t(x_t)
\Bigl(
(\mathcal TQ_t)(x_t)
-Q_t(x_t)
+M_{t+1}
\Bigr).
$$

### 5. The error recursion

Define

$$
\Delta_t
=
Q_t-Q^*.
$$

Since the optimal Q-function is a fixed point,

$$
Q^*=\mathcal TQ^*.
$$

Subtracting the fixed-point equation from the update gives

$$
\Delta_{t+1}(x_t)
=
\bigl(1-\alpha_t(x_t)\bigr)\Delta_t(x_t)
+
\alpha_t(x_t)
\Bigl(
(\mathcal TQ_t)(x_t)
-(\mathcal TQ^*)(x_t)
+M_{t+1}
\Bigr).
$$

The conditional mean of the bracketed signal contracts toward zero:

$$
\|\mathcal TQ_t-\mathcal TQ^*\|_\infty
\leq
\gamma\|\Delta_t\|_\infty.
$$

Ignoring noise momentarily, an update to a maximally erroneous coordinate moves that coordinate toward a quantity whose magnitude is at most the discount factor times the current maximum error. This is the stochastic analogue of asynchronous value iteration.

### 6. Conditional variance control

Suppose the reward has bounded conditional second moment:

$$
\mathbb E[r_t^2\mid s_t=s,a_t=a]
\leq
C_r
$$

uniformly over all state-action pairs. Using

$$
\left|
\max_{a'}Q_t(s_{t+1},a')
\right|
\leq
\|Q_t\|_\infty,
$$

one obtains a bound of the form

$$
\mathbb E[M_{t+1}^2\mid\mathcal F_t]
\leq
C
\bigl(1+\|Q_t\|_\infty^2\bigr).
$$

This is why stability cannot be silently omitted. Before using martingale convergence, one must either prove that the iterates remain bounded or invoke a theorem whose assumptions and proof already establish boundedness.

### 7. Why the two stepsize sums have opposite roles

Consider the scalar recursion

$$
z_{t+1}
=
(1-c\alpha_t)z_t
+
\alpha_t\xi_{t+1},
$$

where the noise is centered. The deterministic contraction accumulated by the current time resembles

$$
\prod_{k=0}^{t-1}(1-c\alpha_k)
\leq
\exp
\left(
-c\sum_{k=0}^{t-1}\alpha_k
\right).
$$

Therefore, the condition

$$
\sum_k\alpha_k=\infty
$$

forces the initial condition to disappear. Meanwhile, the conditional variance of the cumulative noise is controlled by a series proportional to

$$
\sum_k\alpha_k^2.
$$

Thus, the condition

$$
\sum_k\alpha_k^2<\infty
$$

prevents persistent finite-variance fluctuations. These are not arbitrary technical conditions: one removes the initialization, while the other suppresses stochastic noise.

### 8. A stochastic contraction theorem

The classical proof can be organized around the following abstract principle. Consider an asynchronous recursion

$$
\Delta_{t+1}(x)
=
\bigl(1-\widetilde\alpha_t(x)\bigr)\Delta_t(x)
+
\widetilde\alpha_t(x)F_t(x).
$$

Suppose that the conditional mean is a strict contraction:

$$
\left|
\mathbb E[F_t\mid\mathcal F_t]
\right|_\infty
\leq
\gamma\|\Delta_t\|_\infty,
$$

and the conditional variance obeys a quadratic-growth bound:

$$
\operatorname{Var}(F_t(x)\mid\mathcal F_t)
\leq
C
\bigl(1+\|\Delta_t\|_\infty^2\bigr).
$$

If every coordinate satisfies the two Robbins-Monro sums, a stochastic contraction theorem yields

$$
\Delta_t\longrightarrow0
\qquad\text{almost surely}.
$$

For Q-Learning, take

$$
F_t(x_t)
=
(\mathcal TQ_t)(x_t)
-(\mathcal TQ^*)(x_t)
+M_{t+1}.
$$

Its conditional mean is precisely the Bellman difference, and Bellman contraction provides the required factor.

### 9. A direct boundedness intuition

Suppose rewards are almost surely bounded by a constant:

$$
|r_t|\leq \overline R.
$$

Let

$$
B
=
\max
\left\{
\|Q_0\|_\infty,
\frac{\overline R}{1-\gamma}
\right\}.
$$

If the current table satisfies

$$
\|Q_t\|_\infty\leq B,
$$

then every sample target satisfies

$$
\left|
r_t+\gamma\max_{a'}Q_t(s_{t+1},a')
\right|
\leq
\overline R+\gamma B
\leq
B.
$$

Since the updated coordinate is a convex combination of its previous value and the sample target, it remains between minus B and B. By induction,

$$
\|Q_t\|_\infty\leq B
\qquad\text{for all }t.
$$

This simple invariant-region argument is unavailable when rewards are unbounded. In that setting, boundedness must be proved using moment and drift arguments, truncation, projection, or a general stability theorem.

### 10. Exploration versus stationarity

Almost-sure convergence fundamentally needs every coordinate to keep receiving updates. A sufficient trajectory-level condition is

$$
N_t(s,a)\longrightarrow\infty
\qquad\text{almost surely for every }(s,a).
$$

If the behavior policy induces an irreducible positive recurrent Markov chain over state-action pairs with stationary distribution

$$
\nu(s,a)>0,
$$

then an ergodic theorem gives

$$
\frac{N_t(s,a)}{t}
\longrightarrow
\nu(s,a)
\qquad\text{almost surely}.
$$

The qualitative convergence theorem uses the fact that the count diverges. Refined asymptotic-rate and covariance results use the limiting frequency itself.

### 11. What the classical theorem does not say

The convergence conclusion

$$
Q_t\longrightarrow Q^*
\qquad\text{almost surely}
$$

does not imply any of the following without additional analysis:

- that the error is of order (t^{-1/2});
- that the rescaled error is asymptotically Gaussian;
- that the limiting covariance is finite;
- that a constant stepsize converges to the exact fixed point;
- that the result remains valid under infinite-variance rewards.

The classical theorem answers the consistency question. The remaining posts study rate and distribution.

### 12. Proof checklist

When checking an almost-sure convergence proof for Q-Learning, verify each of the following explicitly:

1. The filtration is defined before conditional expectations are taken.
2. The sample Bellman target has the correct conditional mean.
3. The noise is a martingale difference with respect to the chosen filtration.
4. The conditional noise moment is controlled.
5. Every state-action coordinate is updated infinitely often.
6. The local stepsizes have infinite sum and square-summable squares.
7. The Bellman operator is a strict contraction in the norm used by the theorem.
8. Boundedness is proved or supplied by an applicable stability theorem.
9. The fixed point of the mean recursion is exactly the optimal Q-function.

Missing any one of these steps leaves a genuine mathematical gap.

### 13. Primary references

- T. Jaakkola, M. I. Jordan, and S. P. Singh, [On the Convergence of Stochastic Iterative Dynamic Programming Algorithms](https://direct.mit.edu/neco/article/6/6/1185/5826/On-the-Convergence-of-Stochastic-Iterative-Dynamic), 1994.
- J. N. Tsitsiklis, [Asynchronous Stochastic Approximation and Q-Learning](https://link.springer.com/article/10.1023/A%3A1022689125041), 1994.
- C. J. C. H. Watkins and P. Dayan, [Q-Learning](https://link.springer.com/article/10.1007/BF00992698), 1992.

---

## Chapter 2: The ODE Method for Q-Learning: Stability Before Convergence

The ordinary-differential-equation method translates a noisy discrete recursion into a deterministic continuous-time dynamical system. Its central message is subtle: identifying a stable limiting ODE is not by itself enough. One must first prove that the stochastic iterates do not escape to infinity.

This distinction between stability and convergence is one of the most important lessons in stochastic approximation.

### 1. General stochastic approximation

Consider a recursion in a finite-dimensional vector space:

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t
\Bigl(
h(\theta_t)
+
M_{t+1}
+
\varepsilon_{t+1}
\Bigr).
$$

The function h is the deterministic mean field, the centered stochastic noise is represented by the martingale difference, and the last term contains an asymptotically negligible perturbation.

Introduce stochastic-approximation time by

$$
\tau_0=0,
\qquad
\tau_t
=
\sum_{k=0}^{t-1}\alpha_k.
$$

Construct a continuous-time interpolation that equals the iterate at each interpolation time. Over a fixed interval of stochastic-approximation time, the cumulative martingale noise becomes small because the squared stepsizes are summable. The interpolated process therefore approximately follows

$$
\dot\theta(\tau)=h(\theta(\tau)).
$$

If the iterates are bounded and the ODE has a globally asymptotically stable equilibrium, then the stochastic recursion converges to that equilibrium.

The phrase “if the iterates are bounded” carries most of the difficulty.

### 2. Mean field for asynchronous Q-Learning

Suppose the state-action sampling process has stationary occupancy probabilities

$$
\nu(s,a)>0.
$$

Define the positive diagonal matrix

$$
D_\nu
=
\operatorname{diag}
\bigl(
\nu(s,a):(s,a)\in\mathcal S\times\mathcal A
\bigr).
$$

Under an idealized global-clock description, the expected update is

$$
h(Q)
=
D_\nu(\mathcal TQ-Q).
$$

The associated ODE is

$$
\dot Q
=
D_\nu(\mathcal TQ-Q).
$$

The fixed points satisfy

$$
D_\nu(\mathcal TQ-Q)=0.
$$

Because every diagonal entry is strictly positive, this is equivalent to

$$
\mathcal TQ=Q.
$$

Hence the unique equilibrium is the optimal Q-function.

### 3. Global stability in the sup norm

Define the ODE error by

$$
E(\tau)
=
Q(\tau)-Q^*.
$$

The error dynamics are

$$
\dot E
=
D_\nu
\bigl(
\mathcal TQ-\mathcal TQ^*-E
\bigr).
$$

Let

$$
V(E)=\|E\|_\infty.
$$

Because the sup norm is nonsmooth when several coordinates tie for the maximum, we use its upper Dini derivative. Select a coordinate attaining the maximum absolute error.

#### 3.1 Positive maximal coordinate

Suppose the selected coordinate satisfies

$$
E(x)=\|E\|_\infty.
$$

Bellman contraction implies

$$
(\mathcal TQ-\mathcal TQ^*)(x)
\leq
\gamma\|E\|_\infty.
$$

Consequently,

$$
\begin{aligned}
\dot E(x)
&=
\nu(x)
\Bigl(
(\mathcal TQ-\mathcal TQ^*)(x)-E(x)
\Bigr)
\\
&\leq
-\nu(x)(1-\gamma)\|E\|_\infty
\\
&\leq
-\nu_{\min}(1-\gamma)\|E\|_\infty.
\end{aligned}
$$

#### 3.2 Negative maximal coordinate

Suppose instead that

$$
E(x)=-\|E\|_\infty.
$$

Bellman contraction also gives the lower bound

$$
(\mathcal TQ-\mathcal TQ^*)(x)
\geq
-\gamma\|E\|_\infty.
$$

Therefore,

$$
\dot E(x)
\geq
\nu(x)(1-\gamma)\|E\|_\infty.
$$

Since the absolute value of a negative coordinate decreases when that coordinate increases,

$$
\frac{d}{d\tau}|E(x)|
\leq
-\nu_{\min}(1-\gamma)\|E\|_\infty.
$$

Combining the two cases yields

$$
D^+V(E)
\leq
-\nu_{\min}(1-\gamma)V(E).
$$

Gronwall's inequality gives

$$
\|Q(\tau)-Q^*\|_\infty
\leq
\exp
\bigl(
-\nu_{\min}(1-\gamma)\tau
\bigr)
\|Q(0)-Q^*\|_\infty.
$$

Thus, the mean ODE is globally exponentially stable.

### 4. Why ODE stability does not automatically bound the recursion

It is tempting to argue as follows: the limiting ODE is stable, so the stochastic iterates must be bounded. This inference is not valid without additional assumptions. The approximation between a stochastic recursion and its ODE is reliable only over bounded regions unless stability has already been established. If an iterate becomes extremely large, the noise variance and nonlinear remainder may also become large, invalidating the local approximation.

The logic must therefore be:

1. Prove that the stochastic iterates are almost surely bounded.
2. Use the ODE method on the resulting compact region.
3. Conclude that the limit set lies in the globally attracting equilibrium.

The Borkar-Meyn theorem provides a powerful way to establish the first step.

### 5. Scaling the drift at infinity

For a general mean field, define the scaled drift

$$
h_c(\theta)
=
\frac{h(c\theta)}{c}.
$$

Suppose that as the scale tends to infinity,

$$
h_c(\theta)
\longrightarrow
h_\infty(\theta)
$$

uniformly on compact sets. The limiting scaled ODE is

$$
\dot\theta=h_\infty(\theta).
$$

If the origin is globally asymptotically stable for this ODE, then the drift points inward at sufficiently large scales. Under suitable noise and stepsize conditions, this prevents the stochastic recursion from escaping to infinity.

This method separates the finite-scale equilibrium from the large-scale stability mechanism.

### 6. The scaled Q-Learning drift

Recall

$$
h(Q)=D_\nu(\mathcal TQ-Q).
$$

For a positive scaling constant,

$$
\frac{\mathcal T(cQ)(s,a)}{c}
=
\frac{R(s,a)}{c}
+
\gamma
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q(s',a').
$$

As the scale tends to infinity, the reward term vanishes. Define the homogeneous Bellman recession operator by

$$
(\mathcal T_\infty Q)(s,a)
=
\gamma
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q(s',a').
$$

Then

$$
h_\infty(Q)
=
D_\nu(\mathcal T_\infty Q-Q).
$$

The scaled ODE is

$$
\dot Q
=
D_\nu(\mathcal T_\infty Q-Q).
$$

The recession operator remains a contraction:

$$
\|\mathcal T_\infty Q-\mathcal T_\infty Q'\|_\infty
\leq
\gamma\|Q-Q'\|_\infty.
$$

Moreover,

$$
\mathcal T_\infty 0=0.
$$

Repeating the Dini-derivative argument gives

$$
D^+\|Q\|_\infty
\leq
-\nu_{\min}(1-\gamma)\|Q\|_\infty.
$$

Therefore, the origin is globally exponentially stable for the scaled ODE. This is the large-scale inward drift required by the Borkar-Meyn stability argument.

### 7. Noise conditions

A standard martingale-noise assumption is

$$
\mathbb E[M_{t+1}\mid\mathcal F_t]=0
$$

and

$$
\mathbb E
\bigl[
\|M_{t+1}\|^2
\mid
\mathcal F_t
\bigr]
\leq
C
\bigl(1+\|Q_t\|^2\bigr).
$$

The linear growth on the right is compatible with the scaled-ODE argument: at a large radius, both drift and noise scale at most linearly, but the stable recession ODE creates a systematic inward force.

Together with

$$
\sum_t\alpha_t=\infty,
\qquad
\sum_t\alpha_t^2<\infty,
$$

the theorem yields almost-sure boundedness of the iterates under its remaining regularity assumptions.

### 8. From boundedness to convergence

Once boundedness is known, the continuous interpolation is an asymptotic pseudo-trajectory of the mean ODE. Informally, for every fixed stochastic-approximation time horizon, the interpolated stochastic path becomes uniformly close to an ODE solution as the starting time tends to infinity.

The limit set of a bounded asymptotic pseudo-trajectory is internally chain transitive for the ODE. Because the Q-Learning mean ODE has a unique globally asymptotically stable equilibrium, the only possible limit set is

$$
\{Q^*\}.
$$

Hence

$$
Q_t\longrightarrow Q^*
\qquad\text{almost surely}.
$$

### 9. Markovian sampling and the Poisson remainder

For a single behavior trajectory, the update noise is not generally an independent martingale difference after merely subtracting the stationary mean. The instantaneous update depends on a Markov state. A more accurate representation is

$$
Q_{t+1}
=
Q_t
+
\alpha_t H(Q_t,Z_{t+1}),
$$

where the augmented Markov state may contain

$$
Z_t=(s_t,a_t,s_{t+1},r_t).
$$

The stationary mean field is

$$
h(Q)
=
\int H(Q,z)\,\pi(dz).
$$

To connect the Markov update with this mean, one typically solves a Poisson equation and decomposes the centered Markov noise into:

- a martingale difference;
- a telescoping boundary term;
- a small remainder caused by the movement of the parameter.

For almost-sure convergence, these extra terms must be negligible on finite stochastic-approximation time windows. For a central limit theorem, their first-order contribution determines the long-run covariance and cannot simply be discarded.

### 10. Local clocks versus a fixed diagonal mean field

The mean field

$$
D_\nu(\mathcal TQ-Q)
$$

is the clean global-clock representation under stationary visitation frequencies. Classical asynchronous stochastic-approximation theorems can be more general: coordinate update rates may vary with time, provided they remain balanced and every coordinate receives a nonvanishing share of updates on the asymptotic time scale.

Thus, one should not automatically insert a fixed occupancy matrix into every asynchronous proof. The correct mean dynamics depend on whether the algorithm uses:

- a global stepsize indexed by physical time;
- a local stepsize indexed by the number of visits to each coordinate;
- normalized importance weights;
- an explicitly stationary i.i.d. coordinate sampler;
- a single Markovian behavior trajectory.

The fixed diagonal matrix is a useful and important model, but the chosen clock must match the actual algorithm.

### 11. What the ODE method gives and what it does not

The ODE method gives a qualitative description of the limiting trajectory and a principled route to stability. By itself, it usually does not give the exact distribution of the fluctuations around the equilibrium. For that, one must linearize the stochastic recursion and retain the leading martingale term.

The two layers are therefore:

$$
\text{global ODE analysis}
\quad\Longrightarrow\quad
\text{consistency and stability},
$$

followed by

$$
\text{local linear stochastic analysis}
\quad\Longrightarrow\quad
\text{rates and limit distributions}.
$$

### 12. Proof checklist

For an ODE-based Q-Learning proof, check the following in order:

1. The stochastic recursion is written with the correct mean field.
2. The time scale matches the global or local stepsize convention.
3. The martingale and Markov remainders are identified.
4. The mean field is locally Lipschitz or otherwise generates well-defined solutions.
5. The scaled drift exists at infinity.
6. The origin is globally asymptotically stable for the scaled ODE.
7. The noise has the moment growth required by the stability theorem.
8. Almost-sure boundedness follows before the limit-set theorem is invoked.
9. The original mean ODE has the desired fixed point as its unique global attractor.

### 13. Primary reference

- V. S. Borkar and S. P. Meyn, [The O.D.E. Method for Convergence of Stochastic Approximation and Reinforcement Learning](https://epubs.siam.org/doi/10.1137/S0363012997331639), 2000.

---

## Chapter 3: The Asymptotic Convergence Rate of Q-Learning

After proving that Q-Learning converges, the next natural question is how fast it converges. The answer is more delicate than the familiar slogan “stochastic approximation has a square-root rate.” The deterministic contraction may be too weak relative to a reciprocal-time stepsize, in which case the initial condition decays more slowly than the stochastic fluctuations. Unequal state-action visitation frequencies can weaken this contraction further.

The classical paper of Szepesvári makes this phenomenon explicit for tabular Q-Learning. This chapter develops the underlying mechanism from a scalar model and then connects it to the full algorithm.

### 1. A scalar recursion containing the entire phenomenon

Consider

$$
z_{t+1}
=
\left(1-\frac{c}{t+1}\right)z_t
+
\frac{1}{t+1}\xi_{t+1},
$$

where the noise is centered and has finite variance:

$$
\mathbb E[\xi_{t+1}\mid\mathcal F_t]=0,
\qquad
\mathbb E[\xi_{t+1}^2\mid\mathcal F_t]\leq\sigma^2.
$$

The coefficient c is the effective contraction strength. The exact solution is

$$
z_t
=
\Phi_{t,0}z_0
+
\sum_{k=0}^{t-1}
\frac{1}{k+1}
\Phi_{t,k+1}\xi_{k+1},
$$

where

$$
\Phi_{t,k}
=
\prod_{j=k}^{t-1}
\left(1-\frac{c}{j+1}\right).
$$

For large indices, this product behaves like

$$
\Phi_{t,k}
\asymp
\left(\frac{k}{t}\right)^c.
$$

Therefore, the initialization term behaves as

$$
\Phi_{t,0}z_0
=
O(t^{-c}).
$$

The standard deviation of the noise term is approximately

$$
\begin{aligned}
\left[
\sum_{k=1}^{t}
\frac{1}{k^2}
\left(\frac{k}{t}\right)^{2c}
\right]^{1/2}
&=
t^{-c}
\left[
\sum_{k=1}^{t}k^{2c-2}
\right]^{1/2}.
\end{aligned}
$$

The behavior of the sum changes at the threshold

$$
c=\frac12.
$$

### 2. Three rate regimes

#### 2.1 Weak contraction

If

$$
0<c<\frac12,
$$

then the series

$$
\sum_{k=1}^{\infty}k^{2c-2}
$$

converges. Both the early-noise contribution and the initialization remain of order

$$
t^{-c}.
$$

This is slower than the square-root rate.

#### 2.2 Critical contraction

If

$$
c=\frac12,
$$

then

$$
\sum_{k=1}^{t}k^{-1}
\asymp
\log t.
$$

The typical stochastic scale becomes

$$
\sqrt{\frac{\log t}{t}}.
$$

Almost-sure pathwise statements can involve iterated-logarithm corrections rather than the mean-square logarithm shown by this elementary calculation.

#### 2.3 Strong contraction

If

$$
c>\frac12,
$$

then

$$
\sum_{k=1}^{t}k^{2c-2}
\asymp
t^{2c-1}.
$$

The noise term has the canonical scale

$$
t^{-1/2}.
$$

The initialization decays faster and no longer determines the leading error.

This threshold is also the reason that a last-iterate central limit theorem for reciprocal stepsizes requires an eigenvalue condition involving one half.

### 3. Deterministic relaxed value iteration

Before adding sampling noise, consider

$$
Q_{t+1}
=
(1-\alpha_t)Q_t
+
\alpha_t\mathcal TQ_t.
$$

Subtracting the fixed point and applying Bellman contraction gives

$$
\begin{aligned}
\|Q_{t+1}-Q^*\|_\infty
&\leq
(1-\alpha_t)\|Q_t-Q^*\|_\infty
+
\alpha_t\|\mathcal TQ_t-\mathcal TQ^*\|_\infty
\\
&\leq
\bigl(1-\alpha_t(1-\gamma)\bigr)
\|Q_t-Q^*\|_\infty.
\end{aligned}
$$

For

$$
\alpha_t=\frac{a}{t+1},
$$

the deterministic decay is approximately

$$
\|Q_t-Q^*\|_\infty
=
O\left(t^{-a(1-\gamma)}\right).
$$

Thus, even in the synchronous deterministic recursion, the exponent becomes small when the discount factor is close to one.

### 4. How asynchronous visitation weakens contraction

Suppose state-action pairs are sampled from a fixed distribution. Let the smallest and largest sampling probabilities be

$$
p_{\min}
=
\min_{s,a}p(s,a),
\qquad
p_{\max}
=
\max_{s,a}p(s,a),
$$

and define the imbalance ratio

$$
\mathcal R
=
\frac{p_{\min}}{p_{\max}}.
$$

The least frequently updated coordinate accumulates contraction more slowly than the most frequently updated coordinate determines the global clock. Consequently, the effective worst-coordinate contraction can scale like

$$
\mathcal R(1-\gamma).
$$

The quantity appearing in the classical rate transition is therefore not merely the Bellman gap from one, but the product of that gap and a coverage-balance factor.

### 5. The classical asymptotic-rate statement

Under the fixed-distribution sampling and stepsize setting studied by Szepesvári, the almost-sure asymptotic behavior exhibits two regimes. When

$$
\mathcal R(1-\gamma)<\frac12,
$$

the rate is governed by the weak deterministic contraction and takes the form

$$
\|Q_t-Q^*\|_\infty
=
O\left(t^{-\mathcal R(1-\gamma)}\right)
$$

in the norm used in the paper, up to the precise interpretation of the almost-sure statement. In the stronger-contraction regime, the random fluctuations produce the familiar square-root behavior, with an iterated-logarithm correction in the pathwise bound:

$$
\|Q_t-Q^*\|
=
O\left(
\sqrt{\frac{\log\log t}{t}}
\right).
$$

The important conceptual point is the phase transition. A convergent reciprocal-stepsize algorithm can be slower than the square-root scale because the deterministic memory of early errors has not vanished fast enough.

### 6. Why a larger stepsize constant can help

Consider

$$
\alpha_t=\frac{a}{t+1}.
$$

The scalar effective contraction becomes

$$
c=a\lambda,
$$

where the scalar contraction magnitude is determined by the relevant stable eigenvalue. Increasing the gain can move the recursion from the weak-contraction regime into the square-root regime:

$$
a\lambda>\frac12.
$$

However, increasing the gain also amplifies the one-step noise. Once a central limit theorem holds, the limiting covariance depends on that gain. Thus, the gain constant must negotiate two objectives:

- it must be large enough to erase initialization at the square-root scale;
- it should be chosen to avoid unnecessarily large asymptotic variance.

This tension motivates matrix-gain stochastic approximation and Zap Q-Learning.

### 7. Local clocks change the interpretation

Suppose each coordinate uses a local stepsize based on its own visit count:

$$
\alpha_t(s_t,a_t)
=
\frac{a}{N_t(s_t,a_t)+1}.
$$

Then the rare coordinate takes a larger step whenever it is finally observed. On its local clock, the coordinate experiences approximately the same reciprocal schedule as a common coordinate. This can remove part of the visitation imbalance from the deterministic exponent.

Nevertheless, physical-time fluctuations still depend on how many samples each coordinate receives. The distinction is:

- local-clock convergence measures progress per coordinate update;
- global-clock convergence measures progress per environment interaction.

A rate theorem must state explicitly which clock is used.

### 8. Last iterate versus averaged iterate

The slow-rate phenomenon is especially severe for the last iterate. Averaging forms

$$
\overline Q_t
=
\frac1t\sum_{k=1}^t Q_k.
$$

Under appropriate stability and local smoothness conditions, Polyak-Ruppert averaging can recover a square-root limit even when the raw stepsizes decay more slowly than reciprocal time. It also removes delicate dependence of asymptotic efficiency on the scalar gain constant.

This does not contradict the last-iterate rate result. The two estimators are different random objects.

### 9. Relation to finite-time bounds

A finite-time high-probability theorem often has the schematic form

$$
\|Q_T-Q^*\|_\infty
\leq
\text{transient term}
+
\text{statistical term}
$$

with high probability. The asymptotic rate transition says that the transient term cannot always be dismissed as lower order. For a reciprocal stepsize, it may decay as

$$
T^{-a\lambda},
$$

while the statistical term decays as

$$
T^{-1/2}.
$$

If

$$
a\lambda<\frac12,
$$

the transient remains dominant asymptotically. A finite-time theorem whose initialization term matches this exponent is revealing a genuine feature of the recursion, not merely a loose proof artifact.

### 10. A matrix version of the threshold

Near the optimal Q-function, suppose the mean recursion can be linearized as

$$
e_{t+1}
=
e_t
+
\frac{a}{t+1}
\bigl(Ae_t+\xi_{t+1}\bigr),
$$

where the drift matrix is Hurwitz. The scalar contraction coefficient is replaced by the real parts of the eigenvalues of the gain-scaled matrix. For a standard square-root central limit theorem of the last iterate, one requires

$$
\operatorname{Re}\lambda(aA)<-\frac12
$$

for every eigenvalue. Equivalently,

$$
aA+\frac12I
$$

must be Hurwitz.

If one stable mode lies between zero and minus one half, that mode retains early errors too long for square-root scaling.

### 11. What to remember

The asymptotic rate is controlled by three interacting quantities:

$$
\text{stepsize gain}
\times
\text{Bellman contraction}
\times
\text{coverage balance}.
$$

The square-root rate emerges only when the effective contraction is strong enough to make initialization negligible at that scale. This is why almost-sure convergence, which merely requires infinite accumulated contraction, is strictly weaker than asymptotic normality.

### 12. Primary reference

- C. Szepesvári, [The Asymptotic Convergence-Rate of Q-Learning](https://papers.neurips.cc/paper/1383-the-asymptotic-convergence-rate-of-q-learning), 1997.

---

## Chapter 4: Asymptotic Covariance and Zap Q-Learning

Two convergent Q-Learning algorithms can have dramatically different fluctuations around the optimal Q-function. Almost-sure convergence cannot distinguish them. Asymptotic covariance can.

This chapter develops the local linear theory used by Devraj and Meyn to analyze Q-Learning and motivate Zap Q-Learning. The essential idea is to choose a matrix gain that neutralizes poorly conditioned mean dynamics.

### 1. Begin with linear stochastic approximation

Consider

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{t+1}
\bigl(
A(\theta_t-\theta^*)
+
\xi_{t+1}
\bigr),
$$

where the drift matrix is Hurwitz and the noise is centered. For the reciprocal schedule,

$$
\alpha_t=\frac{a}{t},
$$

the rescaled error is

$$
Z_t
=
\sqrt t\,(\theta_t-\theta^*).
$$

The appearance of one half in the central limit theorem follows directly from the change in the normalization:

$$
\sqrt{t+1}
=
\sqrt t
\left(
1+\frac{1}{2t}+o(t^{-1})
\right).
$$

After substituting the recursion and retaining first-order terms,

$$
Z_{t+1}
=
Z_t
+
\frac1t
\left(aA+\frac12I\right)Z_t
+
\frac{a}{\sqrt t}\xi_{t+1}
+
o(t^{-1}).
$$

Thus, the rescaled process is stable only if

$$
aA+\frac12I
$$

is Hurwitz.

### 2. The asymptotic Lyapunov equation

Let the effective long-run covariance of the noise be

$$
\Gamma
=
\lim_{T\to\infty}
\frac1T
\mathbb E
\left[
\left(\sum_{t=1}^T\xi_t\right)
\left(\sum_{t=1}^T\xi_t\right)^{\mathsf T}
\right].
$$

For independent martingale differences, this reduces to the one-step covariance. For Markovian data, it contains all temporal autocovariances.

Under the central-limit conditions,

$$
\sqrt t\,(\theta_t-\theta^*)
\Rightarrow
\mathcal N(0,\Sigma).
$$

The limiting covariance solves

$$
\left(aA+\frac12I\right)\Sigma
+
\Sigma
\left(aA^{\mathsf T}+\frac12I\right)
+
a^2\Gamma
=0.
$$

Because the shifted drift matrix is Hurwitz, this equation has a unique positive semidefinite solution. It can also be represented as

$$
\Sigma
=
a^2
\int_0^\infty
\exp
\left[
\left(aA+\frac12I\right)u
\right]
\Gamma
\exp
\left[
\left(aA^{\mathsf T}+\frac12I\right)u
\right]
du.
$$

This formula shows how noise is amplified along slowly contracting directions.

### 3. Scalar covariance and gain tuning

Take the scalar mean dynamics

$$
A=-\lambda,
\qquad
\lambda>0.
$$

The central limit theorem requires

$$
a\lambda>\frac12.
$$

If the noise variance is

$$
\Gamma=\sigma^2,
$$

then the Lyapunov equation gives

$$
2\left(-a\lambda+\frac12\right)\Sigma
+
a^2\sigma^2
=0.
$$

Hence

$$
\Sigma(a)
=
\frac{a^2\sigma^2}{2a\lambda-1}.
$$

Differentiating with respect to the gain shows that the minimum occurs at

$$
a^*=\frac1\lambda.
$$

The corresponding covariance is

$$
\Sigma(a^*)
=
\frac{\sigma^2}{\lambda^2}.
$$

Thus, the gain should invert the local contraction strength. A single scalar gain cannot do this simultaneously for several modes with different eigenvalues.

### 4. Local linearization of Q-Learning

Suppose the optimal action at every state is unique and separated by a positive gap. Define

$$
\pi^*(s)
=
\operatorname*{arg\,max}_{a}Q^*(s,a).
$$

For Q-functions sufficiently close to the optimum, the greedy policy remains fixed. Therefore,

$$
\mathcal TQ-\mathcal TQ^*
=
\gamma P^{\pi^*}(Q-Q^*).
$$

For a stationary asynchronous sampler with a diagonal occupancy matrix, the mean field is

$$
h(Q)
=
D_\nu(\mathcal TQ-Q).
$$

Its Jacobian at the fixed point is

$$
A
=
D_\nu
\bigl(
\gamma P^{\pi^*}-I
\bigr).
$$

The Q-Learning error therefore has the local form

$$
Q_{t+1}-Q^*
=
Q_t-Q^*
+
\alpha_{t+1}
\Bigl(
A(Q_t-Q^*)
+
\xi_{t+1}
+
r_{t+1}
\Bigr),
$$

where the nonlinear remainder satisfies

$$
\frac{\|r_{t+1}\|}{\|Q_t-Q^*\|}
\longrightarrow0
$$

under an appropriate local differentiability condition.

### 5. Why large discount factors are difficult

The matrix

$$
I-\gamma P^{\pi^*}
$$

becomes poorly conditioned as the discount approaches one, particularly along directions associated with slowly mixing or nearly invariant modes. In the simplest direction corresponding to the constant vector, a stochastic transition matrix has eigenvalue one, so the Bellman linearization contains the factor

$$
1-\gamma.
$$

With a reciprocal stepsize and a unit scalar gain, the square-root stability condition can fail when this effective contraction is at most one half. The algorithm may still converge almost surely, but the last iterate no longer has a finite covariance under square-root scaling.

Even when the central limit theorem exists, the resolvent

$$
(I-\gamma P^{\pi^*})^{-1}
$$

can amplify noise by powers of the effective horizon.

### 6. General matrix-gain stochastic approximation

Introduce a gain matrix:

$$
\theta_{t+1}
=
\theta_t
+
\alpha_{t+1}
G
\bigl(
h(\theta_t)+\xi_{t+1}
\bigr).
$$

The linearized drift becomes

$$
GA.
$$

If the local drift matrix were known and invertible, the Newton gain would be

$$
G^*=-A^{-1}.
$$

Then

$$
G^*A=-I.
$$

Every local mode contracts at the same normalized rate. Under reciprocal stepsizes, the square-root stability condition is automatically satisfied because

$$
-I+\frac12I=-\frac12I
$$

is Hurwitz.

The optimally normalized covariance becomes

$$
\Sigma^*
=
A^{-1}\Gamma A^{-\mathsf T}.
$$

This is the same covariance obtained by efficient Polyak-Ruppert averaging under standard regularity conditions.

### 7. Why the optimal gain cannot simply be inserted

In model-free Q-Learning, the Jacobian is unknown because it depends on:

- the state-action occupancy distribution;
- the transition kernel;
- the optimal greedy policy;
- the derivative of the Bellman mean field at the unknown fixed point.

Therefore, an implementable algorithm must estimate the Jacobian online while simultaneously estimating the optimal Q-function. This creates a two-time-scale problem.

### 8. The Zap idea

Zap Q-Learning maintains both a Q-function estimate and a matrix estimate of the local mean-field Jacobian. Schematically,

$$
\begin{aligned}
A_{t+1}
&=
A_t
+
\beta_{t+1}
\bigl(
\widehat A_{t+1}-A_t
\bigr),
\\
Q_{t+1}
&=
Q_t
-
\alpha_{t+1}
G_{t+1}
\widehat h_{t+1}(Q_t),
\end{aligned}
$$

with a regularized inverse gain of the form

$$
G_{t+1}
\approx
A_{t+1}^{-1}.
$$

This uses the same Newton sign convention as the earlier gain formula: the update contains an explicit minus sign, so its linearized drift is approximately minus the identity.

The matrix recursion runs on the faster time scale:

$$
\frac{\alpha_t}{\beta_t}
\longrightarrow0.
$$

From the perspective of the slower Q-update, the matrix estimate has nearly equilibrated to the current Jacobian. The resulting algorithm approximates a stochastic Newton-Raphson method.

The name “Zap” reflects the goal of rapidly eliminating slow modes rather than allowing them to decay according to the original Bellman conditioning.

### 9. Where the noise covariance comes from

At the optimum, define the Bellman innovation for a sampled transition by

$$
\zeta_{t+1}
=
e_{x_t}
\Bigl(
r_t
+
\gamma\max_{a'}Q^*(s_{t+1},a')
-Q^*(x_t)
\Bigr).
$$

The conditional mean vanishes after the asynchronous mean-field normalization is handled correctly. With independent sampling, the covariance is essentially

$$
\Gamma
=
\mathbb E
\left[
\zeta_{t+1}\zeta_{t+1}^{\mathsf T}
\right].
$$

With Markovian sampling, temporal dependence changes this to a long-run covariance:

$$
\Gamma
=
\sum_{k=-\infty}^{\infty}
\operatorname{Cov}(\zeta_0,\zeta_k),
$$

provided the series is well defined. The gain matrix can correct slow deterministic modes, but it cannot remove irreducible Bellman noise.

### 10. Action ties and nonsmoothness

If two actions are optimal at a state, the mapping

$$
Q\longmapsto\max_aQ(s,a)
$$

is not differentiable at the optimal Q-function. Different perturbation directions can select different greedy actions and hence different transition matrices. In this case, the displayed linearization matrix is not automatically a valid ordinary Jacobian.

Possible remedies include:

- imposing a unique optimal policy with positive action gaps;
- assuming a quadratic local policy-stability condition;
- using a generalized or directional derivative;
- analyzing a switching or piecewise-linear limit.

Any asymptotic covariance formula that writes (P^{\pi^*}) without addressing ties is relying on an unstated smoothness assumption.

### 11. Last-iterate covariance versus averaged covariance

For a well-tuned matrix gain, the last iterate can attain

$$
A^{-1}\Gamma A^{-\mathsf T}.
$$

Polyak-Ruppert averaging can often attain the same covariance while using a simpler scalar stepsize. The tradeoff is conceptual and computational:

- matrix-gain methods attempt to accelerate the iterates themselves;
- averaging leaves the raw recursion simple and improves the final estimator.

Both methods are responses to the same local linear system.

### 12. What to check in a covariance proof

1. Almost-sure convergence and boundedness are established first.
2. The mean field is differentiable, or an adequate local smoothness replacement is stated.
3. The linearized matrix is Hurwitz.
4. For reciprocal stepsizes, the shifted matrix satisfies the one-half stability condition.
5. The nonlinear remainder is negligible at the square-root scale.
6. The partial sums of the noise satisfy a martingale or Markov-chain central limit theorem.
7. The covariance uses the long-run noise covariance when samples are dependent.
8. The Lyapunov equation has the correct signs and includes the one-half normalization shift.

### 13. Primary references

- A. M. Devraj and S. P. Meyn, [Fastest Convergence for Q-Learning](https://arxiv.org/abs/1707.03770), 2017.
- A. M. Devraj and S. P. Meyn, [Zap Q-Learning](https://proceedings.neurips.cc/paper/2017/hash/4671aeaf49c792689533b00664a5c3ef-Abstract.html), 2017.

---

## Chapter 5: Polyak-Ruppert Averaged Q-Learning

The raw stochastic-approximation iterate remembers the choice of stepsize gain. Polyak-Ruppert averaging largely removes that sensitivity. Instead of reporting the last Q-table, one reports the average of the iterates. Under appropriate conditions, this simple transformation produces a square-root central limit theorem with an asymptotically efficient covariance.

This chapter first explains the mechanism for a linear recursion and then develops the statistical analysis of averaged Q-Learning studied by Li, Yang, Liang, Zhang, and Jordan.

### 1. Why average the iterates?

Consider a stochastic approximation recursion

$$
\theta_t
=
\theta_{t-1}
+
\eta_t
\bigl(
h(\theta_{t-1})+\xi_t
\bigr).
$$

Define the average

$$
\overline\theta_T
=
\frac1T\sum_{t=1}^T\theta_t.
$$

The individual iterates are correlated and can fluctuate substantially. Averaging does not create independent samples. Its benefit comes from an algebraic cancellation: after linearizing the recursion, the average error can be written as the inverse Jacobian applied to the average noise, plus smaller boundary and nonlinear terms.

### 2. The key algebra in the linear case

Suppose

$$
e_t
=
e_{t-1}
+
\eta_t
\bigl(
Ae_{t-1}+\xi_t
\bigr),
$$

where the matrix is invertible and Hurwitz. Rearranging gives

$$
Ae_{t-1}
=
\frac{e_t-e_{t-1}}{\eta_t}
-
\xi_t.
$$

Summing and multiplying by the inverse matrix yields

$$
\frac1T\sum_{t=1}^T e_{t-1}
=
-A^{-1}
\frac1T\sum_{t=1}^T\xi_t
+
A^{-1}
\frac1T\sum_{t=1}^T
\frac{e_t-e_{t-1}}{\eta_t}.
$$

The first term is a transformed empirical average of the noise. The second is a weighted boundary term. Under the standard stepsize and stability conditions,

$$
\frac1{\sqrt T}
\sum_{t=1}^T
\frac{e_t-e_{t-1}}{\eta_t}
\longrightarrow0
$$

in probability. Therefore,

$$
\sqrt T\,\overline e_T
=
-A^{-1}
\frac1{\sqrt T}\sum_{t=1}^T\xi_t
+
o_{\mathbb P}(1).
$$

If the noise partial sums satisfy

$$
\frac1{\sqrt T}\sum_{t=1}^T\xi_t
\Rightarrow
\mathcal N(0,\Gamma),
$$

then Slutsky's theorem gives

$$
\sqrt T\,\overline e_T
\Rightarrow
\mathcal N
\left(
0,
A^{-1}\Gamma A^{-\mathsf T}
\right).
$$

The leading covariance does not contain the stepsize gain.

### 3. Why slower-decaying stepsizes are useful

A common Polyak-Ruppert schedule is

$$
\eta_t
=
c(t+1)^{-\kappa},
\qquad
\frac12<\kappa<1.
$$

The lower bound on the exponent ensures square summability:

$$
\sum_t\eta_t^2<\infty.
$$

The upper bound ensures that the stepsize decays more slowly than reciprocal time. This produces aggressive contraction of the initialization:

$$
\prod_{k=1}^t(1-c_0\eta_k)
\lesssim
\exp
\bigl(-c_1t^{1-\kappa}\bigr).
$$

The raw iterate can have fluctuations larger than the square-root scale, but their average still has a square-root limit. Averaging converts a stable but noisy trajectory into an efficient estimator.

### 4. Synchronous Q-Learning with a generative model

The statistical analysis of Li and coauthors uses synchronous sampling. At every iteration, a generative model supplies an independent reward and next-state sample for every state-action pair. Define the empirical Bellman operator by

$$
(\widehat{\mathcal T}_tQ)(s,a)
=
r_t(s,a)
+
\gamma
\max_{a'}Q(s_t'(s,a),a').
$$

The update is

$$
Q_t
=
(1-\eta_t)Q_{t-1}
+
\eta_t\widehat{\mathcal T}_tQ_{t-1}.
$$

Equivalently,

$$
Q_t-Q_{t-1}
=
\eta_t
\bigl(
\widehat{\mathcal T}_tQ_{t-1}-Q_{t-1}
\bigr).
$$

Conditional unbiasedness gives

$$
\mathbb E
\bigl[
\widehat{\mathcal T}_tQ
\bigr]
=
\mathcal TQ.
$$

Hence the mean field is

$$
h(Q)=\mathcal TQ-Q.
$$

There is no occupancy matrix in this synchronous generative model because every coordinate is updated at every iteration.

### 5. Bellman noise at the optimum

Define

$$
Z_t
=
\widehat{\mathcal T}_tQ^*
-
\mathcal TQ^*.
$$

Since the samples are independent across iterations and the empirical operator is unbiased,

$$
\mathbb E[Z_t]=0.
$$

The covariance of the Bellman noise is

$$
\Sigma_Z
=
\operatorname{Var}(Z_t).
$$

Coordinatewise,

$$
\begin{aligned}
Z_t(s,a)
&=
r_t(s,a)-R(s,a)
\\
&\quad+
\gamma
\left[
\max_{a'}Q^*(s_t'(s,a),a')
-
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q^*(s',a')
\right].
\end{aligned}
$$

Thus, even deterministic rewards can produce Bellman noise through random transitions.

### 6. Local smoothness of the optimality operator

The maximization map is globally Lipschitz but not globally differentiable. The averaged central limit theorem needs a first-order approximation near the optimal Q-function.

Under a unique optimal policy with a positive action gap, the greedy policy is locally constant. More generally, one can impose a local policy-stability condition ensuring that the policy-switching remainder is quadratic. A representative form is

$$
\left\|
\bigl(P^{\pi_Q}-P^{\pi^*}\bigr)
(Q-Q^*)
\right\|_\infty
\leq
L\|Q-Q^*\|_\infty^2,
$$

where

$$
\pi_Q(s)
\in
\operatorname*{arg\,max}_aQ(s,a).
$$

This condition implies the local expansion

$$
\mathcal TQ-\mathcal TQ^*
=
\gamma P^{\pi^*}(Q-Q^*)
+
R(Q),
$$

with

$$
\|R(Q)\|_\infty
\leq
C\|Q-Q^*\|_\infty^2.
$$

The Jacobian of the mean field is therefore

$$
A
=
\gamma P^{\pi^*}-I.
$$

It is convenient to define the positive stable operator

$$
G
=
I-\gamma P^{\pi^*}.
$$

Then

$$
A=-G.
$$

### 7. Error decomposition

Let

$$
\Delta_t=Q_t-Q^*.
$$

Add and subtract the empirical Bellman operator evaluated at the optimum:

$$
\begin{aligned}
\Delta_t
&=
\Delta_{t-1}
+
\eta_t
\Bigl(
\widehat{\mathcal T}_tQ_{t-1}
-
\widehat{\mathcal T}_tQ^*
\Bigr)
\\
&\quad+
\eta_t
\Bigl(
\widehat{\mathcal T}_tQ^*
-
\mathcal TQ^*
\Bigr)
-
\eta_t\Delta_{t-1}.
\end{aligned}
$$

After isolating the local linear drift, the recursion becomes

$$
\Delta_t
=
(I-\eta_tG)\Delta_{t-1}
+
\eta_tZ_t
+
\eta_tU_t
+
\eta_tR_t.
$$

Here the first random term is the Bellman noise evaluated at the optimum. The second is a multiplicative-noise term caused by evaluating the random transition operator at the current error. The third is the nonlinear policy-switching remainder.

The proof must show that after averaging and multiplying by the square root of the sample size, the last two terms vanish.

### 8. The averaged asymptotic representation

Under the moment, stepsize, stability, and local smoothness assumptions, the averaged error satisfies

$$
\sqrt T
\bigl(
\overline Q_T-Q^*
\bigr)
=
G^{-1}
\frac1{\sqrt T}
\sum_{t=1}^T Z_t
+
o_{\mathbb P}(1).
$$

Because the Bellman noises are independent across iterations in the synchronous generative model,

$$
\frac1{\sqrt T}
\sum_{t=1}^T Z_t
\Rightarrow
\mathcal N(0,\Sigma_Z).
$$

Consequently,

$$
\sqrt T
\bigl(
\overline Q_T-Q^*
\bigr)
\Rightarrow
\mathcal N(0,\Sigma_Q),
$$

where

$$
\Sigma_Q
=
\bigl(I-\gamma P^{\pi^*}\bigr)^{-1}
\Sigma_Z
\bigl(I-\gamma P^{\pi^*}\bigr)^{-\mathsf T}.
$$

This is the central covariance formula.

### 9. Interpreting the covariance

Using the Neumann expansion,

$$
\bigl(I-\gamma P^{\pi^*}\bigr)^{-1}
=
\sum_{k=0}^{\infty}
\gamma^k
\bigl(P^{\pi^*}\bigr)^k.
$$

A one-step Bellman perturbation is propagated through every future step of the optimal controlled dynamics. The asymptotic covariance therefore contains the intrinsic covariance of one empirical Bellman update, the discounted propagation of that noise through the optimal transition system, and correlations between coordinates created by this propagation.

The operator norm obeys

$$
\left\|
\bigl(I-\gamma P^{\pi^*}\bigr)^{-1}
\right\|_\infty
\leq
\frac1{1-\gamma}.
$$

Therefore, a worst-case covariance bound can carry two powers of the effective horizon.

### 10. Functional central limit theorem

The result can be strengthened from a terminal-time central limit theorem to a process-level statement. Define

$$
\Phi_T(r)
=
\frac1{\sqrt T}
\sum_{t=1}^{\lfloor Tr\rfloor}
(Q_t-Q^*),
\qquad
0\leq r\leq1.
$$

Under the assumptions of the statistical analysis,

$$
\Phi_T(\cdot)
\Rightarrow
\Sigma_Q^{1/2}B(\cdot),
$$

where the limit is a standard multivariate Brownian motion transformed by the asymptotic covariance. Evaluating at the terminal time recovers the ordinary central limit theorem for the average.

The functional result is useful for online inference because it describes the joint behavior of estimates formed at multiple fractions of the data.

### 11. Why fourth moments appear

The analysis of nonlinear and multiplicative remainders often requires stronger control than the terminal Gaussian limit itself. A bounded fourth-moment assumption on rewards is used to control products of the current error and random transition deviations and to prove uniform integrability of several remainder terms.

This creates an important distinction:

- a classical central limit theorem for a fixed sum may need only slightly more than two moments;
- proving that an adaptive nonlinear stochastic recursion is asymptotically equivalent to that fixed sum can require higher moments.

### 12. Synchronous versus asynchronous covariance

In synchronous Q-Learning, every coordinate receives one independent sample per iteration, so the mean Jacobian is

$$
\gamma P^{\pi^*}-I.
$$

In asynchronous Q-Learning, the visitation matrix enters:

$$
D_\nu
\bigl(
\gamma P^{\pi^*}-I
\bigr).
$$

Moreover, a single behavior trajectory creates temporal dependence. The one-step covariance must then be replaced by a long-run covariance obtained through a Markov-chain central limit theorem. The algebraic role of averaging remains similar, but the noise decomposition is substantially harder.

### 13. Tail averaging

The full average gives equal weight to early iterates, which may still reflect initialization. A tail average discards a burn-in period:

$$
\overline Q_{T_0:T}
=
\frac1{T-T_0}
\sum_{t=T_0+1}^{T}Q_t.
$$

If the burn-in is negligible relative to the total sample size but long enough to suppress initialization, the tail average has the same first-order asymptotic covariance while often performing better at finite sample sizes.

### 14. Proof checklist

1. The Q-Learning recursion converges to the correct fixed point.
2. The stepsizes satisfy the averaging conditions.
3. The Bellman noise at the optimum is centered and has sufficient moments.
4. The optimality operator admits a valid local first-order approximation.
5. The multiplicative-noise remainder is negligible after averaging.
6. The nonlinear greedy-policy remainder is negligible at the square-root scale.
7. The weighted telescoping boundary term vanishes.
8. The partial-sum noise satisfies the required central or functional central limit theorem.
9. The covariance is transformed by the inverse local Bellman operator on both sides.

### 15. Primary references

- B. T. Polyak and A. B. Juditsky, [Acceleration of Stochastic Approximation by Averaging](https://epubs.siam.org/doi/10.1137/0330046), 1992.
- X. Li, W. Yang, J. Liang, Z. Zhang, and M. I. Jordan, [A Statistical Analysis of Polyak-Ruppert Averaged Q-Learning](https://proceedings.mlr.press/v206/li23b.html), 2023.

---

## Chapter 6: Markovian and Asynchronous Central Limit Theory for Q-Learning

Synchronous Q-Learning with a generative model is statistically clean: every state-action pair receives an independent sample at every iteration. Real trajectory-based Q-Learning is different. Only the visited coordinate is updated, and successive updates are temporally dependent because they come from a Markov chain.

The central limit theorem must account for both differences. The main tool is the Poisson equation, which converts centered Markovian fluctuations into a martingale plus lower-order correction terms.

### 1. The augmented Markov state

Let a behavior policy generate a trajectory. A convenient Markov state is

$$
Z_t=(s_t,a_t,s_{t+1},r_t).
$$

Depending on the reward-generation model, it may be preferable to include only the current state-action pair in the Markov chain and treat the next state and reward as conditionally sampled emissions. The precise augmentation is less important than ensuring that the update can be written as a measurable function of the current parameter and one Markov observation.

Write the recursion as

$$
Q_{t+1}
=
Q_t
+
\alpha_{t+1}H(Q_t,Z_{t+1}).
$$

For asynchronous Q-Learning,

$$
H(Q,Z_{t+1})
=
e_{x_t}
\Bigl(
r_t
+
\gamma\max_{a'}Q(s_{t+1},a')
-Q(x_t)
\Bigr),
$$

where

$$
x_t=(s_t,a_t).
$$

### 2. Stationary mean field

Assume the behavior chain is ergodic with invariant law

$$
\pi.
$$

The stationary mean field is

$$
h(Q)
=
\int H(Q,z)\,\pi(dz).
$$

When the invariant state-action occupancy distribution is denoted by

$$
\nu,
$$

this becomes

$$
h(Q)
=
D_\nu(\mathcal TQ-Q).
$$

The target fixed point is still the optimal Q-function because every diagonal entry of the occupancy matrix is positive.

The instantaneous centered observation is

$$
H(Q_t,Z_{t+1})-h(Q_t).
$$

Unlike an independent sample centered by its expectation, this is generally not a martingale difference with respect to the trajectory filtration.

### 3. Why naive centering fails

For a Markov chain,

$$
\mathbb E
\bigl[
H(Q_t,Z_{t+1})
\mid
\mathcal F_t
\bigr]
$$

depends on the current Markov state. It need not equal the stationary average

$$
h(Q_t).
$$

Therefore,

$$
H(Q_t,Z_{t+1})-h(Q_t)
$$

contains predictable temporal structure in addition to innovation noise. Treating this term as a martingale difference would give the wrong covariance and can invalidate the proof.

### 4. The Poisson equation

Let the Markov transition kernel be denoted by

$$
P_Z.
$$

For a fixed parameter, define the centered function

$$
g_Q(z)
=
H(Q,z)-h(Q).
$$

A Poisson solution is a function satisfying

$$
\widehat g_Q(z)
-
(P_Z\widehat g_Q)(z)
=
g_Q(z).
$$

Under geometric ergodicity and suitable moment conditions, one formal representation is

$$
\widehat g_Q(z)
=
\sum_{k=0}^{\infty}
P_Z^kg_Q(z).
$$

The series sums the future effect of the current Markov state on centered observations. It also explains why mixing assumptions appear: they guarantee that the accumulated future effect is finite in a suitable norm.

### 5. Martingale decomposition for a fixed parameter

The Poisson equation can be used to express the cumulative centered sum as

$$
\sum_{t=0}^{T-1}g_Q(Z_t)
=
\sum_{t=0}^{T-1}M_{t+1}^{Q}
+
\widehat g_Q(Z_0)
-
\widehat g_Q(Z_T),
$$

where the increments satisfy

$$
\mathbb E
\bigl[
M_{t+1}^{Q}
\mid
\mathcal F_t
\bigr]
=0.
$$

One suitable innovation is constructed by subtracting the one-step conditional prediction:

$$
M_{t+1}^{Q}
=
\widehat g_Q(Z_{t+1})
-
(P_Z\widehat g_Q)(Z_t).
$$

The remaining terms form a telescoping boundary. Dividing by the square root of the sample size, this boundary vanishes under a finite second-moment condition.

Thus, a Markov-chain central limit theorem can be proved through a martingale central limit theorem.

### 6. The moving-parameter complication

In Q-Learning, the parameter is not fixed. The relevant Poisson solution changes from one iteration to the next. The decomposition therefore produces an additional term of the form

$$
\widehat g_{Q_t}(Z_{t+1})
-
\widehat g_{Q_{t-1}}(Z_{t+1}).
$$

If the Poisson solution is Lipschitz in the parameter,

$$
\left\|
\widehat g_Q(z)-\widehat g_{Q'}(z)
\right\|
\leq
L(z)\|Q-Q'\|,
$$

then the movement term is controlled by

$$
\|Q_t-Q_{t-1}\|
=
O(\alpha_t)
$$

times an integrable random Lipschitz factor. Its cumulative contribution can then be shown negligible under appropriate stepsize conditions.

This is one reason Markovian stochastic-approximation CLTs require more than an ordinary Markov-chain CLT for a fixed test function.

### 7. Long-run covariance

At the optimum, define

$$
g_*(z)
=
H(Q^*,z)-h(Q^*).
$$

Because

$$
h(Q^*)=0,
$$

this is simply the stationary Q-Learning update evaluated at the optimum. The long-run covariance is

$$
\Gamma
=
\lim_{T\to\infty}
\frac1T
\operatorname{Var}
\left(
\sum_{t=1}^Tg_*(Z_t)
\right).
$$

When the autocovariance series is absolutely summable,

$$
\begin{aligned}
\Gamma
&=
\operatorname{Cov}(g_*(Z_0),g_*(Z_0))
\\
&\quad+
\sum_{k=1}^{\infty}
\left[
\operatorname{Cov}(g_*(Z_0),g_*(Z_k))
+
\operatorname{Cov}(g_*(Z_k),g_*(Z_0))
\right].
\end{aligned}
$$

This matrix differs from the one-step covariance unless the observations are independent.

### 8. Covariance through the Poisson solution

Let the martingale innovation at the optimum be

$$
M_{t+1}^*
=
\widehat g_*(Z_{t+1})
-
(P_Z\widehat g_*)(Z_t).
$$

Under stationarity, the same long-run covariance can be represented as

$$
\Gamma
=
\mathbb E
\left[
M_{t+1}^*
(M_{t+1}^*)^{\mathsf T}
\right].
$$

This representation is often more convenient for a martingale central limit theorem.

### 9. Local linearization of the asynchronous mean field

Assume the optimal policy is locally stable. The Bellman operator has the expansion

$$
\mathcal TQ-\mathcal TQ^*
=
\gamma P^{\pi^*}(Q-Q^*)
+
R(Q),
$$

where the remainder is higher order. Therefore,

$$
h(Q)
=
A(Q-Q^*)
+
D_\nu R(Q),
$$

where

$$
A
=
D_\nu
\bigl(
\gamma P^{\pi^*}-I
\bigr).
$$

The occupancy matrix affects both the contraction rates and the transformation of the noise.

### 10. Polyak-Ruppert average under Markovian sampling

Define

$$
\overline Q_T
=
\frac1T\sum_{t=1}^TQ_t.
$$

After the Poisson decomposition and local linearization, the desired asymptotic representation is

$$
\sqrt T
\bigl(
\overline Q_T-Q^*
\bigr)
=
-A^{-1}
\frac1{\sqrt T}
\sum_{t=1}^TM_t^*
+
o_{\mathbb P}(1).
$$

The martingale central limit theorem gives

$$
\frac1{\sqrt T}
\sum_{t=1}^TM_t^*
\Rightarrow
\mathcal N(0,\Gamma).
$$

Consequently,

$$
\sqrt T
\bigl(
\overline Q_T-Q^*
\bigr)
\Rightarrow
\mathcal N
\left(
0,
A^{-1}\Gamma A^{-\mathsf T}
\right).
$$

The formula looks like the independent-sampling covariance, but both the Jacobian and long-run covariance now reflect asynchronous Markovian sampling.

### 11. What asynchronous sampling changes

Asynchronous sampling creates four changes.

#### 11.1 Coordinate masking

Only one coordinate is updated per transition. The update vector contains a random coordinate basis vector.

#### 11.2 Occupancy weighting

The mean drift contains the stationary visitation matrix.

#### 11.3 Temporal dependence

The covariance is a long-run covariance rather than a one-step variance.

#### 11.4 Random update counts

If local-clock stepsizes are used, the step taken at physical time depends on the random number of previous visits to the current coordinate. A central limit theorem must control the difference between random local counts and their deterministic stationary approximations.

### 12. Local-clock linearization

Define the visit count by

$$
N_t(s,a)
=
\sum_{k=0}^{t-1}
\mathbf 1\{(s_k,a_k)=(s,a)\}.
$$

The ergodic theorem gives

$$
\frac{N_t(s,a)}{t}
\longrightarrow
\nu(s,a)
\qquad
\text{almost surely}.
$$

For a local reciprocal-count stepsize,

$$
\alpha_t(s_t,a_t)
=
\frac{a}{N_t(s_t,a_t)+1},
$$

one expects

$$
\alpha_t(s,a)
\approx
\frac{a}{t\nu(s,a)}
$$

when the coordinate is visited. The factor from the visit probability can cancel the reciprocal occupancy factor in the local stepsize at the mean-drift level.

However, fluctuations in the counts still create remainder terms, and the physical-time covariance retains dependence on coverage. This cancellation must be derived from the actual recursion; it should not be assumed from an informal frequency substitution.

### 13. Non-asymptotic versus asymptotic CLTs

An asymptotic central limit theorem states that a probability distance tends to zero, without specifying the rate. A non-asymptotic CLT bounds a distance such as Wasserstein distance at a finite sample size:

$$
d_{\mathrm W}
\left(
\mathcal L
\left(
\sqrt T(\overline Q_T-Q^*)
\right),
\mathcal N(0,\Sigma)
\right)
\leq
\varepsilon_T.
$$

Recent work on asynchronous averaged Q-Learning controls this approximation while tracking nonlinear Bellman remainders, Markov mixing, asynchronous coordinate updates, finite-sample moment bounds, and the number of state-action coordinates.

The proof is substantially stronger than merely invoking an asymptotic theorem.

### 14. A practical proof architecture

A rigorous proof is easiest to organize into the following steps.

1. Prove almost-sure convergence and moment bounds for the iterates.
2. Establish a local smoothness or margin condition for the Bellman operator.
3. Linearize the mean field around the optimal Q-function.
4. Solve the parameter-dependent Poisson equation for the Markov noise.
5. Extract the leading martingale differences.
6. Bound the telescoping Poisson boundary.
7. Bound the error caused by the moving parameter inside the Poisson solution.
8. Bound the asynchronous count and stepsize approximations.
9. Show that nonlinear and multiplicative remainders are negligible after averaging.
10. Apply a martingale or Markov-chain central limit theorem.
11. Transform the long-run covariance through the inverse Jacobian.

### 15. Important warning about mixing time

Mixing time is not generally inserted as a simple multiplicative factor into the exact asymptotic covariance. The exact object is

$$
\Gamma.
$$

Mixing assumptions are used to prove that this covariance exists and to upper-bound it. Two chains with the same coarse mixing-time bound can have different signed autocovariance structures and therefore different exact asymptotic covariances.

Finite-time concentration often pays an explicit mixing penalty. Asymptotic theory retains a more instance-specific autocorrelation matrix.

### 16. Primary references

- G. Fort, [Central Limit Theorems for Stochastic Approximation with Controlled Markov Chain Dynamics](https://www.numdam.org/articles/10.1051/ps/2014013/), 2015.
- X. Liu, [Central Limit Theorems for Asynchronous Averaged Q-Learning](https://arxiv.org/abs/2509.18964), 2025, updated 2026.

---

## Chapter 7: Constant-Stepsize Q-Learning: Stationary Laws, Bias, and Extrapolation

With a diminishing stepsize, the noise is gradually suppressed and Q-Learning can converge to the exact optimal Q-function. With a constant stepsize, new noise continues to enter with nonvanishing magnitude. The iterate usually does not settle at one point. Instead, its distribution approaches a stationary law concentrated near the optimum.

This changes the basic object of asymptotic analysis. The questions become:

1. Does the joint process possess a unique stationary distribution?
2. How fast does the distribution approach stationarity?
3. How far is the stationary mean from the optimal Q-function?
4. Does averaging produce a central limit theorem?
5. Can the stationary bias be removed?

Zhang and Xie answer these questions for asynchronous Q-Learning with Markovian data under their stated assumptions.

### 1. The constant-stepsize recursion

Let the stepsize be fixed:

$$
\alpha_t\equiv\alpha>0.
$$

The visited state-action coordinate is updated according to

$$
\begin{aligned}
Q_{t+1}(s_t,a_t)
&=
Q_t(s_t,a_t)
\\
&\quad+
\alpha
\Bigl(
r_t
+
\gamma\max_{a'}Q_t(s_{t+1},a')
-
Q_t(s_t,a_t)
\Bigr).
\end{aligned}
$$

All other coordinates remain unchanged. The increment does not vanish as time grows. Therefore, even after the iterate reaches a neighborhood of the optimum, each new reward and next-state sample perturbs it again.

### 2. Why exact convergence is generally impossible

At the optimal Q-function, the sample temporal-difference error is

$$
\delta_t^*
=
r_t
+
\gamma\max_{a'}Q^*(s_{t+1},a')
-
Q^*(s_t,a_t).
$$

Its conditional expectation is zero:

$$
\mathbb E
\bigl[
\delta_t^*
\mid
s_t,a_t
\bigr]
=0.
$$

But the random variable itself is not normally zero. If the algorithm were exactly at the optimum, the next update would be

$$
Q_{t+1}
=
Q^*
+
\alpha e_{x_t}\delta_t^*.
$$

Thus, the optimum is a fixed point of the mean dynamics, not an absorbing state of the random recursion.

### 3. The joint Markov-chain viewpoint

The behavior state alone is Markov under a stationary behavior policy. The Q-table depends on its full history. However, the pair consisting of the behavior state and the current Q-table forms a time-homogeneous Markov chain:

$$
X_t=(Z_t,Q_t).
$$

Its transition law is determined by the MDP, behavior policy, reward model, maximization operation, and fixed stepsize.

The desired limiting object is an invariant probability measure

$$
\overline\mu_\alpha
$$

satisfying

$$
\overline\mu_\alpha K_\alpha
=
\overline\mu_\alpha,
$$

where the joint transition kernel is denoted by

$$
K_\alpha.
$$

If the invariant measure is unique and the distribution of the chain converges to it, one can define a stationary random Q-table

$$
Q_\infty^{(\alpha)}
\sim
\overline\mu_\alpha^Q.
$$

The superscript emphasizes that the stationary law depends on the stepsize.

### 4. Coupling and contraction

Consider two Q-Learning recursions driven by the same trajectory and reward samples but initialized at different Q-tables. Let their difference be

$$
\Delta_t
=
Q_t^{(1)}-Q_t^{(2)}.
$$

For the updated coordinate,

$$
\begin{aligned}
|\Delta_{t+1}(s_t,a_t)|
&\leq
(1-\alpha)
|\Delta_t(s_t,a_t)|
\\
&\quad+
\alpha\gamma
\|\Delta_t\|_\infty.
\end{aligned}
$$

Hence

$$
\|\Delta_{t+1}\|_\infty
\leq
\|\Delta_t\|_\infty.
$$

A single asynchronous update need not contract the global sup norm because the largest-error coordinate may not be visited. Over a sufficiently long block, however, ergodicity ensures that all relevant coordinates are updated. Bellman contraction then produces strict average contraction.

This blockwise coupling is used to establish existence, uniqueness, and geometric convergence toward a stationary law in a Wasserstein metric.

### 5. Distributional convergence

Under the ergodicity, boundedness, and sufficiently small stepsize conditions of the constant-stepsize analysis, the joint chain has a unique invariant distribution and

$$
\mathcal L(Z_t,Q_t)
\longrightarrow
\overline\mu_\alpha
$$

in Wasserstein distance. The convergence is geometric after a mixing-dependent transient. Schematically,

$$
\mathcal W_2
\left(
\mathcal L(Z_t,Q_t),
\overline\mu_\alpha
\right)
\leq
C_\alpha\rho_\alpha^t,
\qquad
0<\rho_\alpha<1.
$$

The contraction becomes weaker as the stepsize becomes smaller, because the algorithm moves only a small amount per update. This creates a central tradeoff:

- a larger stepsize forgets initialization faster;
- a smaller stepsize produces a tighter stationary neighborhood.

### 6. Error decomposition at stationarity

The mean-square error can be conceptually decomposed as

$$
\begin{aligned}
\mathbb E\|Q_t-Q^*\|^2
&\approx
\left\|
\mathbb E Q_t
-
\mathbb E Q_\infty^{(\alpha)}
\right\|^2
\\
&\quad+
\left\|
\mathbb E Q_\infty^{(\alpha)}
-
Q^*
\right\|^2
\\
&\quad+
\operatorname{tr}
\operatorname{Var}(Q_t).
\end{aligned}
$$

The three terms are:

1. optimization or burn-in error;
2. stationary bias;
3. stationary variance.

The first vanishes geometrically with time. The latter two remain for a fixed stepsize.

### 7. The averaged-iterate central limit theorem

Define the stationary mean

$$
m_\alpha
=
\mathbb E
\left[
Q_\infty^{(\alpha)}
\right].
$$

The centered partial sum is

$$
S_n
=
\sum_{t=0}^{n-1}
\bigl(
Q_t-m_\alpha
\bigr).
$$

Under the conditions of the paper,

$$
\frac1{\sqrt n}S_n
\Rightarrow
\mathcal N(0,\Sigma_\alpha).
$$

Equivalently, for the average

$$
\overline Q_n
=
\frac1n\sum_{t=0}^{n-1}Q_t,
$$

one has

$$
\sqrt n
\bigl(
\overline Q_n-m_\alpha
\bigr)
\Rightarrow
\mathcal N(0,\Sigma_\alpha).
$$

The center is the stationary mean, not the optimal Q-function. Therefore, averaging reduces fluctuations around the stationary mean but does not by itself remove the constant-stepsize bias.

### 8. Functional central limit theorem

Define the partial-sum process

$$
Y_n(u)
=
\frac1{\sqrt n}
\sum_{t=0}^{\lfloor nu\rfloor-1}
\bigl(
Q_t-m_\alpha
\bigr),
\qquad
0\leq u\leq1.
$$

The process-level result has the form

$$
Y_n(\cdot)
\Rightarrow
\Sigma_\alpha^{1/2}B(\cdot)
$$

in the Skorokhod space, where the limit is Brownian motion transformed by the stationary long-run covariance.

### 9. Why the stationary mean is biased

The stationary recursion satisfies a zero-drift identity:

$$
\mathbb E
\left[
H(Q_\infty^{(\alpha)},Z_\infty)
\right]
=0.
$$

It is generally invalid to replace the random argument by its mean:

$$
\mathbb E
\left[
H(Q_\infty^{(\alpha)},Z_\infty)
\right]
\neq
h
\left(
\mathbb E Q_\infty^{(\alpha)}
\right).
$$

There are two reasons:

- the Bellman optimality update is nonlinear because of the maximum;
- the stationary Q-table and the Markov state are dependent.

As a result, the stationary mean need not solve the Bellman fixed-point equation.

### 10. First-order bias expansion

Under the local regularity assumptions used by Zhang and Xie, the stationary mean admits the expansion

$$
\mathbb E
\left[
Q_\infty^{(\alpha)}
\right]
=
Q^*
+
\alpha B
+
\widetilde O(\alpha^2),
$$

where the vector

$$
B
$$

does not depend on the stepsize. It depends on the MDP, behavior policy, reward law, and Markovian dependence structure.

The linear dependence is more precise than the coarser bound

$$
O(\sqrt\alpha)
$$

that can follow from a general mean-square estimate. A square-root upper bound does not imply that the true bias is of square-root order.

For i.i.d. data, the paper shows that the first-order coefficient vanishes in the setting it studies:

$$
B=0.
$$

Higher-order nonlinear bias can still remain.

### 11. Tail averaging

After a burn-in index, define

$$
\overline Q_{k_0:k}^{(\alpha)}
=
\frac1{k-k_0}
\sum_{t=k_0}^{k-1}
Q_t^{(\alpha)}.
$$

Tail averaging controls three quantities:

$$
\text{initialization error},
\qquad
\text{variance},
\qquad
\text{stationary bias}.
$$

The initialization error decreases geometrically in the burn-in. The variance of the average decreases with the averaging length. The stationary bias remains approximately

$$
\alpha B.
$$

This demonstrates why “average longer” cannot remove all error when the stepsize is fixed.

### 12. Richardson-Romberg extrapolation

Run two Q-Learning recursions on the same data stream, one with stepsize

$$
\alpha
$$

and one with stepsize

$$
2\alpha.
$$

Their stationary means satisfy

$$
\begin{aligned}
m_\alpha
&=
Q^*+\alpha B+\widetilde O(\alpha^2),
\\
m_{2\alpha}
&=
Q^*+2\alpha B+\widetilde O(\alpha^2).
\end{aligned}
$$

Form the extrapolated estimator

$$
\widetilde Q^{(\alpha)}
=
2\overline Q^{(\alpha)}
-
\overline Q^{(2\alpha)}.
$$

Its expectation is

$$
\begin{aligned}
\mathbb E
\left[
\widetilde Q^{(\alpha)}
\right]
&=
2
\bigl(
Q^*+\alpha B+\widetilde O(\alpha^2)
\bigr)
\\
&\quad-
\bigl(
Q^*+2\alpha B+\widetilde O(\alpha^2)
\bigr)
\\
&=
Q^*+\widetilde O(\alpha^2).
\end{aligned}
$$

The first-order bias cancels.

Using the same data stream is valuable because it correlates the two estimators and can reduce the variance of their difference. The exact covariance must nevertheless be analyzed jointly.

### 13. Small-stepsize diffusion intuition

Near the optimum and under a locally fixed greedy policy, the recursion resembles

$$
e_{t+1}
=
e_t
+
\alpha
\bigl(
Ae_t+\xi_{t+1}
\bigr).
$$

At stationarity, the typical fluctuation scale is

$$
\|e_t\|
=
O_{\mathbb P}(\sqrt\alpha).
$$

Define the normalized error

$$
U_t^{(\alpha)}
=
\frac{Q_t-Q^*}{\sqrt\alpha}.
$$

As the stepsize tends to zero and time is accelerated by the reciprocal stepsize, one expects an Ornstein-Uhlenbeck approximation:

$$
dU(\tau)
=
AU(\tau)\,d\tau
+
\Gamma^{1/2}\,dB(\tau).
$$

Its stationary covariance solves

$$
A\Sigma
+
\Sigma A^{\mathsf T}
+
\Gamma
=0.
$$

This diffusion describes the leading random spread. The order-alpha mean bias is smaller than the order-square-root-alpha random fluctuation of a single stationary iterate, but it matters after long averaging reduces the variance.

### 14. Choosing the stepsize and run length

A schematic error for a tail average is

$$
\text{MSE}
\approx
C_1\alpha^2
+
\frac{C_2}{n\alpha}
+
C_3\exp(-c\alpha k_0).
$$

The terms represent squared bias, averaged variance, and burn-in error. This formula is only schematic, but it displays the tuning problem:

- small stepsizes reduce bias;
- large stepsizes shorten burn-in;
- sufficient averaging is needed to suppress stationary variance.

Richardson-Romberg extrapolation replaces the leading squared-bias contribution by a higher-order term, permitting a larger stepsize for the same bias target.

### 15. Proof checklist

1. The behavior process is ergodic.
2. The joint state and Q-table process is correctly identified as Markov.
3. A coupling or drift argument proves existence and uniqueness of the stationary law.
4. Distributional convergence is established in a metric strong enough for the required moments.
5. The averaged CLT is centered at the stationary mean.
6. The long-run covariance includes temporal dependence.
7. The local greedy-policy smoothness assumption is stated before deriving a bias expansion.
8. Averaging is not claimed to remove stationary bias.
9. Richardson-Romberg weights correctly cancel the linear stepsize term.

### 16. Primary reference

- Y. Zhang and Q. Xie, [Constant Stepsize Q-Learning: Distributional Convergence, Bias and Extrapolation](https://arxiv.org/abs/2401.13884), 2024.

---

## Chapter 8: Q-Learning with Infinite-Variance Rewards: Beyond the Gaussian Limit

Classical Q-Learning theory usually assumes bounded rewards or at least a finite conditional second moment. Suppose instead that the reward is perturbed by centered noise whose mean exists but whose variance is infinite. Does Q-Learning still converge? If it converges, what replaces the Gaussian central limit theorem?

The short answer is:

- almost-sure convergence may survive;
- the square-root Gaussian limit generally does not;
- a stable-law limit can arise under regular variation;
- the correct normalization is slower than the square-root normalization;
- the gain and Bellman contraction still create a threshold between transient-dominated and noise-dominated behavior.

This chapter separates established heavy-tailed stochastic-approximation principles from the additional work required for a complete Q-Learning theorem.

### 1. “Zero mean” must mean an actual expectation

Let the reward be

$$
r_t
=
R(s_t,a_t)+\xi_{t+1}.
$$

To say that the perturbation has zero mean, one needs

$$
\mathbb E
\left[
|\xi_{t+1}|
\mid
s_t,a_t
\right]
<\infty
$$

and

$$
\mathbb E
\left[
\xi_{t+1}
\mid
s_t,a_t
\right]
=0.
$$

A symmetric Cauchy random variable has a location parameter equal to zero, but its expectation does not exist. In that case, the usual mean-reward Bellman operator is not defined.

The most natural infinite-variance setting therefore assumes a finite moment of order strictly greater than one but less than two. For some tail index,

$$
p\in(1,2),
$$

assume

$$
\mathbb E|\xi_{t+1}|^q<\infty
\qquad
\text{for every }q<p,
$$

while

$$
\mathbb E|\xi_{t+1}|^2=\infty.
$$

### 2. Regular variation and stable attraction

Infinite variance alone does not determine a limiting distribution. A stable limit typically requires that the noise lie in the domain of attraction of a stable law. A representative tail condition is

$$
\mathbb P(\xi>x)
\sim
c_+x^{-p}L(x),
$$

and

$$
\mathbb P(\xi<-x)
\sim
c_-x^{-p}L(x),
$$

where the function is slowly varying:

$$
\frac{L(cx)}{L(x)}
\longrightarrow1
\qquad
\text{for every }c>0.
$$

Under suitable centering, there exists a normalizing sequence of order

$$
b_n
\asymp
n^{1/p}L_0(n)
$$

such that

$$
\frac1{b_n}
\sum_{t=1}^n\xi_t
\Rightarrow
S_p,
$$

where the limit is a non-Gaussian stable random variable.

Ignoring slowly varying corrections for intuition,

$$
n^{1-1/p}
\left(
\frac1n\sum_{t=1}^n\xi_t
\right)
\Rightarrow
S_p.
$$

Because

$$
1-\frac1p<\frac12
$$

for every tail index between one and two, the heavy-tailed sample mean converges more slowly than the finite-variance sample mean.

### 3. The one-state one-action Q-Learning model

The cleanest way to see the phenomenon is a Markov decision process with one state and one action. Let

$$
r_t=\mu+\xi_{t+1}.
$$

The optimal action-value is

$$
Q^*
=
\frac{\mu}{1-\gamma}.
$$

With stepsize

$$
\alpha_{t+1}
=
\frac{a}{t+1},
$$

the Q-Learning update is

$$
Q_{t+1}
=
Q_t
+
\frac{a}{t+1}
\bigl(
r_t-(1-\gamma)Q_t
\bigr).
$$

Define the error

$$
e_t=Q_t-Q^*.
$$

Then

$$
e_{t+1}
=
\left(
1-\frac{a(1-\gamma)}{t+1}
\right)e_t
+
\frac{a}{t+1}\xi_{t+1}.
$$

This scalar recursion already contains the competition between Bellman contraction and stable noise.

### 4. Exact weighted representation

Define the effective contraction

$$
c=a(1-\gamma).
$$

Iterating the recursion gives

$$
e_t
=
\Phi_{t,0}e_0
+
a
\sum_{k=1}^{t}
\frac{\Phi_{t,k}}{k}\xi_k,
$$

where

$$
\Phi_{t,k}
=
\prod_{j=k+1}^{t}
\left(
1-\frac{c}{j}
\right).
$$

For large indices,

$$
\Phi_{t,k}
\asymp
\left(\frac{k}{t}\right)^c.
$$

Therefore,

$$
e_t
\approx
t^{-c}e_0
+
a t^{-c}
\sum_{k=1}^{t}
k^{c-1}\xi_k.
$$

For stable noise, the scale of a weighted sum is determined by the sum of the absolute weights raised to the tail index:

$$
\left(
\sum_{k=1}^{t}
k^{p(c-1)}
\right)^{1/p}.
$$

The behavior changes at

$$
p(c-1)=-1,
$$

or equivalently,

$$
c=1-\frac1p.
$$

This is the heavy-tailed counterpart of the finite-variance threshold at one half.

### 5. Three heavy-tailed rate regimes

#### 5.1 Weak contraction

If

$$
c<1-\frac1p,
$$

then

$$
\sum_{k=1}^{\infty}
k^{p(c-1)}
<\infty.
$$

Early observations and the initialization are not forgotten fast enough. The error scale is governed by

$$
t^{-c}.
$$

The canonical stable normalization does not apply to the last iterate.

#### 5.2 Critical contraction

If

$$
c=1-\frac1p,
$$

the weight sum grows logarithmically. The expected normalization includes a logarithmic correction in addition to

$$
t^{1-1/p}.
$$

The exact correction depends on the chosen stable normalization and slowly varying tail factor.

#### 5.3 Strong contraction

If

$$
c>1-\frac1p,
$$

then the weighted noise scale is

$$
t^{c-1+1/p}.
$$

Multiplying by the outside factor gives

$$
t^{-c}t^{c-1+1/p}
=
t^{-1+1/p}.
$$

Hence the natural last-iterate scaling is

$$
t^{1-1/p}e_t.
$$

Under the full stable-attraction conditions, a non-Gaussian stable limit is expected.

### 6. Gain calibration

The strong-contraction condition is

$$
a(1-\gamma)
>
1-\frac1p.
$$

When the discount factor is close to one, a unit gain can fail this condition. The algorithm may still converge, but the last iterate is dominated by slowly decaying early effects.

The Newton-calibrated scalar gain is

$$
a^*
=
\frac1{1-\gamma}.
$$

Then

$$
a^*(1-\gamma)=1,
$$

which is larger than the stable threshold for every tail index greater than one. The resulting recursion is closely related to the sample mean:

$$
e_{t+1}
=
\left(
1-\frac1{t+1}
\right)e_t
+
\frac{1}{(1-\gamma)(t+1)}\xi_{t+1}.
$$

Thus,

$$
e_t
\approx
\frac1{1-\gamma}
\frac1t
\sum_{k=1}^t\xi_k.
$$

The stable limit is then the stable limit of the reward sample mean amplified by the effective horizon.

### 7. Almost-sure convergence without a second moment

The classical square-summability condition is tailored to finite-variance martingale noise:

$$
\sum_t\alpha_t^2<\infty.
$$

Fix a moment order strictly below the stable tail index:

$$
1<q<p.
$$

Suppose the martingale differences have a finite conditional moment of this order. A natural replacement is

$$
\sum_t\alpha_t^q<\infty.
$$

Together with

$$
\sum_t\alpha_t=\infty,
$$

appropriate martingale convergence inequalities can control the cumulative heavy-tailed noise.

For reciprocal stepsizes,

$$
\alpha_t=\frac{a}{t},
$$

one has

$$
\sum_t\alpha_t^q<\infty
$$

for every finite moment order greater than one. Thus, finite variance is not intrinsically necessary for almost-sure convergence.

This observation is only part of a Q-Learning proof. One must also establish stability of the nonlinear iterates under unbounded rewards.

### 8. Why boundedness becomes difficult

With bounded rewards, the interval

$$
\left[
-\frac{\overline R}{1-\gamma},
\frac{\overline R}{1-\gamma}
\right]
$$

creates an invariant bound for every Q-coordinate. Infinite-variance rewards are unbounded, so no deterministic invariant interval exists.

A heavy-tailed convergence proof needs an alternative mechanism, such as:

- a Lyapunov drift argument in a moment of order below the tail index;
- projection onto expanding compact sets;
- truncation or clipping of rewards;
- a robust Bellman-target estimator;
- a Borkar-Meyn stability argument with heavy-tailed martingale control.

One cannot simply retain the bounded-iterate assumption from the classical proof without justification.

### 9. Multidimensional local linearization

Near a locally unique optimal policy, asynchronous Q-Learning has the approximate error recursion

$$
e_{t+1}
=
e_t
+
\alpha_{t+1}
\bigl(
Ae_t+\xi_{t+1}
\bigr),
$$

where

$$
A
=
D_\nu
\bigl(
\gamma P^{\pi^*}-I
\bigr).
$$

For finite-variance noise, asymptotic behavior is summarized by a covariance matrix. A stable distribution with tail index below two has no finite covariance. Its multidimensional law is instead described by a spectral measure that records the directions of extreme jumps.

The linear system transforms the stable noise through its resolvent. Formally, a stable Ornstein-Uhlenbeck limit has the structure

$$
dU(\tau)
=
AU(\tau)\,d\tau
+
dL_p(\tau),
$$

where the driving process is a multivariate stable Lévy process. Its stationary law is stable, but it is not characterized by a Lyapunov covariance equation.

### 10. Polyak-Ruppert averaging under infinite variance

Consider a stable stochastic-approximation recursion with stepsizes

$$
\eta_t
\asymp
t^{-\kappa}.
$$

Heavy-tailed stochastic-gradient results show that a useful averaging regime requires

$$
\frac1p<\kappa<1.
$$

When the exponent satisfies this lower bound, one can choose a finite moment order

$$
1<q<p
$$

close enough to the tail index that

$$
\kappa q>1.
$$

Consequently,

$$
\sum_t\eta_t^q<\infty.
$$

Under local linearity, stability, and stable-attraction assumptions, the average can satisfy

$$
T^{1-1/p}
\bigl(
\overline\theta_T-\theta^*
\bigr)
\Rightarrow
-A^{-1}S_p.
$$

The limit is stable rather than Gaussian. Averaging removes delicate last-iterate gain effects, but it cannot manufacture a finite variance that the underlying noise does not possess.

For Q-Learning, deriving this representation requires proving that Bellman nonlinearities and policy-switching remainders vanish under the slower stable normalization.

### 11. Markovian heavy tails

For a single trajectory, heavy-tailed observations can be temporally clustered. Geometric mixing alone may not be sufficient to identify a stable limit. Stable limit theory for dependent sequences commonly requires:

- joint regular variation;
- an anti-clustering condition;
- control of small jumps;
- identification of a tail process or cluster index.

The limiting stable law depends not only on the marginal reward tail but also on how extremes cluster along the Markov trajectory.

In finite-variance theory, dependence modifies the long-run covariance. In infinite-variance theory, dependence modifies the spectral or Lévy measure of the stable limit.

### 12. The maximization operator under extreme noise

Reward noise enters additively, but the Q-table enters future targets through

$$
\max_aQ_t(s,a).
$$

An extreme reward can perturb one coordinate enough to change the greedy action at a later time. A stable-limit proof must show that after convergence, policy switching is negligible at the chosen normalization, or it must analyze the nonsmooth switching limit directly.

A positive action gap supplies local policy stability:

$$
\Delta_{\min}
=
\min_s
\left[
Q^*(s,\pi^*(s))
-
\max_{a\neq\pi^*(s)}Q^*(s,a)
\right]
>0.
$$

But almost-sure convergence alone does not immediately give the probability estimates needed to discard rare large policy switches at the stable-law scale.

### 13. Classical Q-Learning versus robust Q-Learning

There are two different research questions.

#### 13.1 Preserve the classical update

Keep the raw reward in the temporal-difference target and characterize the stable-law fluctuations. This studies what ordinary Q-Learning naturally does under infinite variance.

#### 13.2 Robustify the update

Replace raw rewards or Bellman targets by clipped, median-of-means, Catoni-type, or other robust estimates. This can improve finite-sample concentration and may restore Gaussian-type behavior after suitable normalization.

The robust algorithm generally has a different asymptotic law. Clipping can introduce bias; adaptive clipping creates a triangular-array problem; batch robust means change the noise dependence across iterations.

Therefore, stable asymptotics for classical Q-Learning and robust finite-time analysis are complementary rather than interchangeable.

### 14. A theorem that one would want to prove

A complete Q-Learning stable-limit theorem would have the following architecture.

#### Assumptions

1. The behavior chain is geometrically ergodic and covers every state-action pair.
2. The reward innovations have conditional mean zero and are jointly regularly varying with tail index between one and two.
3. The Markov sequence satisfies anti-clustering and small-jump conditions.
4. The Q-Learning iterates converge almost surely and possess uniform moments of some order below the tail index.
5. The optimal policy is locally stable, or the Bellman operator satisfies a suitable directional expansion.
6. The stepsize satisfies the heavy-tailed stochastic-approximation conditions.

#### Desired conclusion for an average

For a suitable slowly varying normalization,

$$
\frac{T}{b_T}
\bigl(
\overline Q_T-Q^*
\bigr)
\Rightarrow
-A^{-1}S,
$$

where the stable random vector is determined by the tail process of the Bellman innovations.

#### Required proof steps

1. Establish stability under unbounded rewards.
2. Prove the law of large numbers and local policy stabilization.
3. Obtain a stable functional limit theorem for the Bellman innovations.
4. Linearize the Q-Learning mean field.
5. Control multiplicative transition noise.
6. Show that the nonlinear remainder is negligible under stable normalization.
7. Apply a stable continuous-mapping theorem through the linear resolvent.

This statement should be treated as a research target, not as an already established classical Q-Learning theorem.

### 15. What changes relative to Gaussian theory?

| Feature                  |                      Finite variance |                         Infinite variance with stable index |
| ------------------------ | -----------------------------------: | ----------------------------------------------------------: |
| Natural mean scale       |           square root of sample size |    sample size to the power one minus reciprocal tail index |
| Limit law                |                             Gaussian |                                                      stable |
| Second-order descriptor  |                    covariance matrix |                                    spectral or Lévy measure |
| Stepsize noise condition |                   square summability |   summability in a finite moment order below the tail index |
| Last-iterate threshold   | effective contraction above one half | effective contraction above one minus reciprocal tail index |
| Markov dependence        |                  long-run covariance |                                clustered-extreme stable law |

### 16. Main conclusions

The loss of finite variance does not automatically destroy Q-Learning consistency. It destroys the usual second-order theory. If the mean exists and the stepsizes suppress the heavy-tailed martingale increments strongly enough, almost-sure convergence can remain possible.

The fluctuations are fundamentally different. Their scale is slower, their limit can be stable, and covariance is no longer the right descriptor. The Bellman contraction and stepsize gain still determine whether the last iterate reaches the canonical noise-dominated regime.

### 17. Primary references

- H. Wang, M. Gurbuzbalaban, L. Zhu, U. Simsekli, and M. A. Erdogdu, [Convergence Rates of Stochastic Gradient Descent under Infinite Noise Variance](https://proceedings.neurips.cc/paper/2021/hash/9cdf26568d166bc6793ef8da5afa0846-Abstract.html), 2021.
- V. Zhuang and Y. Sui, [No-Regret Reinforcement Learning with Heavy-Tailed Rewards](https://proceedings.mlr.press/v130/zhuang21a.html), 2021.
- B. Basrak, D. Krizmanic, and J. Segers, [A Functional Limit Theorem for Dependent Sequences with Infinite Variance Stable Limits](https://projecteuclid.org/journals/annals-of-probability/volume-40/issue-5/A-functional-limit-theorem-for-dependent-sequences-with-infinite-variance/10.1214/11-AOP669.full), 2012.

---

## Chapter 9: Decay-to-Zero Learning Rates and Tail-Averaged Q-Learning

The most familiar theoretical learning rates are either constant or polynomially decreasing. A constant rate rapidly removes initialization but leaves a stationary bias and variance. A polynomially decreasing rate converges to the exact fixed point, but its early movement can be slow.

Decay-to-zero schedules try to combine the two behaviors. They remain large during most of a predetermined training horizon and then decay to zero near the end. Bonnerjee, Lou, and Wu develop a sharp asymptotic theory for this idea in synchronous Q-Learning.

### 1. A horizon-dependent triangular array

Fix a total number of iterations

$$
n.
$$

The power-law decay-to-zero schedule is

$$
\eta_{t,n}
=
\eta
\left(
1-\frac{t}{n}
\right)^\nu,
\qquad
1\leq t\leq n.
$$

The special case

$$
\nu=1
$$

is called linear decay to zero, abbreviated LD2Z. The general class is abbreviated PD2Z.

The notation has two indices because changing the total horizon changes the entire schedule. For a fixed fraction of training,

$$
t=\lfloor cn\rfloor,
\qquad
0<c<1,
$$

the stepsize remains of constant order:

$$
\eta_{t,n}
\approx
\eta(1-c)^\nu.
$$

Only near the terminal time does it become small.

This is a triangular-array problem rather than a single infinite recursion. For each horizon, one runs a different sequence of learning rates and obtains a different array of iterates:

$$
\bigl\{
Q_{t,n}:1\leq t\leq n
\bigr\}.
$$

### 2. Synchronous empirical Bellman updates

The analysis uses a synchronous generative model:

$$
Q_{t,n}
=
\bigl(
1-\eta_{t,n}
\bigr)Q_{t-1,n}
+
\eta_{t,n}
\widehat{\mathcal T}_tQ_{t-1,n}.
$$

At every iteration, the empirical Bellman operator supplies an independent reward and next-state sample for every state-action pair.

The Bellman innovation at the optimum is

$$
Z_t
=
\widehat{\mathcal T}_tQ^*
-
\mathcal TQ^*.
$$

The paper assumes a finite moment of order

$$
p\geq2
$$

for this noise and a local quadratic policy-stability condition of the form

$$
\left\|
\bigl(
P^{\pi_Q}-P^{\pi^*}
\bigr)
\bigl(
Q-Q^*
\bigr)
\right\|_\infty
\leq
L
\|Q-Q^*\|_\infty^2.
$$

The second assumption controls the nonsmooth maximization remainder near the optimal policy.

### 3. Why initialization disappears quickly

For a contractive deterministic recursion, the initialization is multiplied by a product resembling

$$
\prod_{j=1}^{t}
\bigl(
1-c_0\eta_{j,n}
\bigr).
$$

Using the exponential bound for products,

$$
\prod_{j=1}^{t}
\bigl(
1-c_0\eta_{j,n}
\bigr)
\leq
\exp
\left(
-c_0
\sum_{j=1}^{t}
\eta_{j,n}
\right).
$$

For the full horizon,

$$
\sum_{j=1}^{n}
\eta_{j,n}
\asymp
n.
$$

Consequently, the terminal initialization effect decreases exponentially in the training horizon:

$$
\exp(-cn)\|Q_0-Q^*\|.
$$

By comparison, for the polynomial schedule

$$
\eta_t
\asymp
t^{-\alpha},
\qquad
\frac12<\alpha<1,
$$

the accumulated stepsize is of order

$$
t^{1-\alpha},
$$

and the corresponding initialization factor is only

$$
\exp
\bigl(
-ct^{1-\alpha}
\bigr).
$$

Both are faster than any polynomial, but decay-to-zero schedules erase the initialization much more aggressively over the prescribed horizon.

### 4. The terminal statistical scale

The stepsize becomes small only in a terminal window. The relevant window length is

$$
m_n
\asymp
n^{\nu/(\nu+1)}.
$$

Within this window, the stochastic error reaches the scale

$$
n^{-\nu/[2(\nu+1)]}.
$$

The paper's terminal moment bound has the schematic form

$$
\|Q_{n,n}-Q^*\|_2
\lesssim
\exp(-cn)\|Q_0-Q^*\|
+
n^{-\nu/[2(\nu+1)]}.
$$

For LD2Z,

$$
\nu=1,
$$

the displayed statistical term is

$$
n^{-1/4}.
$$

Larger powers increase the exponent toward one half, although constants and admissible gain conditions must also be considered. The paper emphasizes that the power is not chosen by its exponent alone.

### 5. Two temporal regimes

The non-asymptotic bound distinguishes two parts of training.

#### 5.1 Transient regime

Before the final window, the stepsize is still of constant order. The algorithm rapidly forgets initialization and behaves qualitatively like a constant-stepsize recursion whose noise level is gradually decreasing.

#### 5.2 Convergence regime

During the last window, the learning rate approaches zero. New stochastic fluctuations are suppressed, and the error settles at the terminal statistical scale.

The transition occurs around

$$
t
\approx
n-Cn^{\nu/(\nu+1)}.
$$

This explains why the last portion of the trajectory has a different statistical character from the first portion.

### 6. Why the ordinary full average is problematic

The usual Polyak-Ruppert estimator is

$$
\widetilde Q_n
=
\frac1n
\sum_{t=1}^{n}Q_{t,n}.
$$

For much of the horizon, the stepsize is bounded away from zero. Those early iterates resemble a constant-stepsize chain and need not individually concentrate at the optimal Q-function as the horizon grows.

Therefore, the full average mixes together:

- early constant-rate-like iterates;
- intermediate transition iterates;
- terminal small-rate iterates.

The Gaussian behavior of the terminal portion does not automatically cancel the persistent fluctuations of the early portion. This is why the usual full average is not the estimator analyzed by the paper.

### 7. Tail Polyak-Ruppert averaging

Choose a constant greater than zero and define the tail length

$$
m_n
=
\left\lfloor
c
n^{\nu/(\nu+1)}
\right\rfloor.
$$

The tail average is

$$
\overline Q_n
=
\frac1{m_n}
\sum_{t=n-m_n+1}^{n}
Q_{t,n}.
$$

This estimator uses precisely the terminal window in which the learning rate is vanishing and the iterates have reached the convergence regime.

Because

$$
\sqrt{m_n}
\asymp
n^{\nu/[2(\nu+1)]},
$$

the central limit theorem has the normalization

$$
n^{\nu/[2(\nu+1)]}
\bigl(
\overline Q_n-Q^*
\bigr)
\Rightarrow
\mathcal N(0,\Sigma).
$$

The covariance does not depend on the horizon, although the paper notes that a simple closed form is generally intractable for this nonstationary weighting scheme.

### 8. Gain and moment conditions

A representative gain condition in the theorem is

$$
0<\eta
<
\frac{
2(1-\gamma)
}{
(1-\gamma)^2
+
2(p-1)\gamma^2
}.
$$

The decay power is required to satisfy

$$
\nu\geq\frac1p.
$$

These restrictions arise from moment control of the nonlinear stochastic recursion. They should not be dropped when quoting the central limit theorem.

The theorem also assumes that the initialization and optimal Q-function lie in a suitable compact set and uses the local policy-stability condition introduced earlier.

### 9. Strong Gaussian approximation

A terminal central limit theorem describes only one normalized estimator. A stronger result couples partial sums of Q-Learning errors with a nonstationary Gaussian process whose covariance matches the time-varying linearized recursion.

The Gaussian comparison process has a form such as

$$
Y_{t,n}
=
\bigl(
I-\eta_{t,n}G
\bigr)
Y_{t-1,n}
+
\eta_{t,n}\mathcal E_t,
$$

where

$$
G
=
I-\gamma P^{\pi^*}
$$

and the innovations are Gaussian vectors with the covariance of the Bellman noise.

The resulting strong invariance principle controls tail partial sums uniformly over a range of terminal times. This is useful for bootstrap-based inference because direct estimation of the complicated terminal covariance can be avoided.

### 10. Comparison with the three standard schedules

| Schedule         | Initialization                          | Long-run or terminal behavior                                 | Requires known horizon |
| ---------------- | --------------------------------------- | ------------------------------------------------------------- | ---------------------- |
| Constant         | rapid geometric forgetting              | stationary bias and variance                                  | no                     |
| Polynomial decay | stretched-exponential forgetting        | exact convergence and standard averaging theory               | no                     |
| Decay to zero    | exponential forgetting over the horizon | exact terminal convergence with tail-specific Gaussian theory | yes                    |

The horizon requirement is a genuine tradeoff. If training is stopped early or extended beyond the planned horizon, the schedule no longer has its intended shape.

### 11. What this result does not yet cover

The ICLR 2026 theory studies synchronous Q-Learning with independent generative-model samples. Extending it to a single asynchronous Markov trajectory would require new control of:

- random coordinate visitation;
- Markovian long-run dependence;
- local-clock effects;
- a parameter-dependent Poisson equation;
- nonstationary Gaussian approximation under asynchronous masking.

Thus, the decay-to-zero result should not be quoted as an asynchronous trajectory theorem.

### 12. Main takeaway

The schedule is large when rapid motion is useful and small when statistical precision is useful. Its analysis is intrinsically horizon dependent. The final iterates form a special window, and averaging that window rather than the full trajectory produces the appropriate Gaussian estimator.

This gives a fourth perspective on asymptotic Q-Learning:

$$
\text{design the entire learning-rate path for a fixed horizon}
$$

rather than choosing only a constant or an infinite-horizon decay exponent.

### 13. Primary reference

- S. Bonnerjee, Z. Lou, and W. B. Wu, [Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](https://proceedings.iclr.cc/paper_files/paper/2026/file/b3b52663b1c01ae961895e419a55fb28-Paper-Conference.pdf), ICLR 2026.
