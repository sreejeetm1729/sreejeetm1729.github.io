---
title: "Robust Asynchronous Q-Learning under Reward and State Corruption via Batching"
date: 2026-08-08 
categories: [rl-blogs]
rl_section: robust-rl
tags: [q-learning, robust reinforcement learning, state corruption, reward corruption, batching]
math: true
description: "How batching and robust Bellman-operator estimation control simultaneous reward and next-state corruption in asynchronous Q-Learning."
---

Reward corruption changes the immediate part of a $Q$-Learning target. State corruption is more subtle: it changes the state at which the learner evaluates its own current $Q$-function. A corrupted next state therefore attacks the bootstrapping mechanism itself. When both forms of feedback can be corrupted, a one-step update may be wrong in two coupled ways, and these errors can propagate through all later iterates.

The paper *Robust Asynchronous Q-Learning under Reward and State Corruption via Batching* by Sreejeet Maity and Aritra Mitra develops BR-Async-Q, an epoch-based algorithm that estimates an entire Bellman update robustly before modifying the $Q$-table. The main insight is that infrequent, low-variance updates can be substantially easier to robustify than high-variance one-sample updates.

## The joint corruption model

Consider a discounted MDP

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
$$

For a clean interaction at $(s_t,a_t)$, the learner receives

$$
s_{t+1}\sim P(\cdot\mid s_t,a_t),
\qquad
r_t(s_t,a_t)\sim\mathcal R(s_t,a_t).
$$

The mean and variance of the clean reward satisfy

$$
|R(s,a)|\leq\overline R,
\qquad
\operatorname{Var}(r(s,a))\leq\overline\sigma^2,
$$

and

$$
\widetilde\sigma=\max\{\overline R,\overline\sigma\}.
$$

The observed reward and next state are

$$
\widetilde r_t
=
(1-Y_{t,1})r_t+Y_{t,1}z_t,
$$

$$
\widetilde s_{t+1}
=
\begin{cases}
s_{t+1}, & Y_{t,2}=0,\\
u_t, & Y_{t,2}=1,
\end{cases}
$$

where

$$
Y_{t,1}\sim\operatorname{Bernoulli}(\varepsilon_R),
\qquad
Y_{t,2}\sim\operatorname{Bernoulli}(\varepsilon_Y).
$$

The two indicator processes may be mutually correlated, although each has the prescribed marginal corruption probability. The adversarial reward $z_t$ may be unbounded, and the adversarial state $u_t$ may be chosen using the history. Thus the attack may coordinate reward and transition corruption at the same time.

## Asynchronous coverage

A behavior policy $\mu$ generates one state-action pair per time step. Under stationarity, define

$$
\lambda(s,a)=\pi(s)\mu(a\mid s),
\qquad
\lambda_{\min}=\min_{s,a}\lambda(s,a)>0.
$$

Only the visited state-action pair supplies data. The main analysis uses an i.i.d. stationary sampling model for clarity, while noting that blocking can extend the approach to Markovian trajectories. The asynchronous challenge remains: in a block of $H$ interactions, the number of samples for $(s,a)$ is random and is only about $H\lambda(s,a)$.

## Why ordinary robust $Q$-updates remain loose

Suppose a learner robustifies each incoming reward but still updates at every time step. Each update direction is based on one next-state sample, so its variance is large. Under reward corruption alone, earlier asynchronous analyses produced corruption terms inflated by the inverse visitation probability. This suggests that rarely visited coordinates amplify the attack.

The paper observes that this amplification is not fundamental under the Huber timing model. It is partly a consequence of making high-variance updates too frequently. If the learner first accumulates a batch, it can estimate both components of the Bellman target with much lower variance and then make one synchronized update.

## Epochs and frozen iterates

BR-Async-Q divides the $T$ interactions into $K$ epochs of length $H$, with

$$
T=KH.
$$

During epoch $k$, the $Q$-table is frozen at $Q_k$. For each state-action pair, the algorithm constructs two data sets:

$$
\mathcal D_k(s,a)
=
\{\widetilde r_t:(s_t,a_t)=(s,a),\ t\in\mathcal I_k\},
$$

and

$$
\mathcal Y_k(s,a)
=
\left\{
\max_{a'}Q_k(\widetilde s_{t+1},a'):
(s_t,a_t)=(s,a),\ t\in\mathcal I_k
\right\}.
$$

The first estimates the immediate reward mean. The second estimates the one-step look-ahead quantity

$$
\mu_k(s,a)
=
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q_k(s',a')
\right].
$$

Freezing $Q_k$ is essential. Conditional on the past, the clean look-ahead samples within the epoch are now identically distributed around one fixed target $\mu_k(s,a)$. If $Q$ were updated inside the epoch, the target distribution would drift with every observation and the batch estimator would no longer estimate a single well-defined object.

## Two robust estimators and two corruption rates

The reward data can be heavy-tailed and unbounded even when clean. The algorithm therefore applies a Huber-robust trimmed estimator to $\mathcal D_k(s,a)$. Its error has the form

$$
|\widehat R_k(s,a)-R(s,a)|
\lesssim
\overline\sigma
\left(
\sqrt{\frac{\log(1/\delta)}{N_k(s,a)}}
+
\sqrt{\varepsilon_R}
\right),
$$

where $N_k(s,a)$ is the number of visits to $(s,a)$ in epoch $k$.

The look-ahead samples are different. Once the $Q$-iterates are bounded, every clean look-ahead target lies in a known interval $[-B,B]$. A clipped mean with this known range then achieves the sharper corruption dependence

$$
|\widehat\mu_k(s,a)-\mu_k(s,a)|
\lesssim
B
\left(
\sqrt{\frac{\log(1/\delta)}{N_k(s,a)}}
+
\varepsilon_Y
\right).
$$

This explains the asymmetric powers of the two corruption probabilities in the final theorem. Finite variance alone permits only $\sqrt{\varepsilon_R}$ reward-mean estimation. A known deterministic bound on the look-ahead values permits linear $\varepsilon_Y$ dependence.

## Ensuring every coordinate is sampled

Let

$$
N_k(s,a)
=
\sum_{t\in\mathcal I_k}
\mathbf 1\{(s_t,a_t)=(s,a)\}.
$$

For a sufficiently long epoch,

$$
H
\gtrsim
\frac{1}{\lambda_{\min}}
\log\left(
\frac{T|\mathcal S||\mathcal A|}{\delta}
\right),
$$

Bernstein's inequality gives the uniform event

$$
N_k(s,a)
\geq
\frac12\lambda_{\min}H
$$

for every epoch and every state-action pair. This event converts the random sample sizes into deterministic lower bounds that can be inserted into the robust mean guarantees.

If a pair receives no observation in an epoch, the corresponding empirical quantities are set to zero by convention. The main theorem operates in the high-probability regime where the epoch length is large enough that this degenerate case does not occur.

## A robust empirical Bellman operator

The reward estimate is clipped to a radius $G_k$ chosen from the robust reward bound. The empirical Bellman operator is

$$
(\widehat{\mathcal T}_kQ_k)(s,a)
=
\operatorname{clip}_{[-G_k,G_k]}
\left(\widehat R_k(s,a)\right)
+
\gamma\widehat\mu_k(s,a).
$$

The $Q$-table is updated once at the end of the epoch:

$$
Q_{k+1}(s,a)
=
(1-\alpha)Q_k(s,a)
+
\alpha(\widehat{\mathcal T}_kQ_k)(s,a).
$$

Clipping the reward estimate also stabilizes the look-ahead estimates indirectly. If rewards entering the recursion are bounded, an induction shows that the $Q$-iterates remain uniformly bounded. The look-ahead variables $\max_{a'}Q_k(\widetilde s,a')$ therefore lie in the fixed interval required by the bounded robust estimator.

## The main finite-time theorem

Choose

$$
K
=
\left\lceil
\frac{2\log T}{1-\gamma}
\right\rceil,
\qquad
\alpha
=
\frac{\log T}{(1-\gamma)K},
$$

and let $H=T/K$ be large enough to guarantee sufficient visits. Then, with probability at least $1-\delta$,

$$
\|Q_K-Q^\star\|_\infty
\leq
\frac{\|Q_0-Q^\star\|_\infty}{T}
+
\widetilde O\left(
\frac{\widetilde\sigma}
{(1-\gamma)^{5/2}}
\sqrt{\frac{1}{\lambda_{\min}T}}
\right)
+
O\left(
\max\left{
\frac{\widetilde\sigma\varepsilon_Y}{(1-\gamma)^2},
\frac{\overline\sigma\sqrt{\varepsilon_R}}{1-\gamma}
\right}
\right).
$$

The logarithmic factors depend on $T$, $|\mathcal S||\mathcal A|$, and $1/\delta$.

The statistical term decays as $T^{-1/2}$ and depends on coverage through $1/\sqrt{\lambda_{\min}}$. This is sharper than a $1/\lambda_{\min}$ statistical dependence and is a direct benefit of batching. The remaining two terms are the biases caused by corrupted next states and rewards.

## Interpreting the two corruption terms

The reward term is

$$
O\left(
\frac{\overline\sigma\sqrt{\varepsilon_R}}{1-\gamma}
\right).
$$

This matches the fundamental lower bound already known for reward-only corruption. It is minimax optimal in its dependence on the corruption fraction, the reward-noise scale, and the effective planning horizon.

The state-corruption term is

$$
O\left(
\frac{\widetilde\sigma\varepsilon_Y}{(1-\gamma)^2}
\right).
$$

One factor $1/(1-\gamma)$ bounds the size of a $Q$-value. A corrupted next state can therefore change a one-step look-ahead target by order $1/(1-\gamma)$. The Bellman recursion accumulates this bias through another geometric factor $1/(1-\gamma)$, producing the square.

Most importantly, neither corruption term contains $\lambda_{\min}$. Rare visitation slows the vanishing statistical term because fewer data are collected, but it does not amplify the asymptotic corruption floor. Batching closes the earlier gap between an upper bound with coverage inflation and a lower bound without it.

## Proof intuition

Let

$$
e_k=Q_k-Q^\star.
$$

The epoch update and the Bellman fixed-point identity yield

$$
\|e_{k+1}\|_\infty
\leq
\left(1-\alpha(1-\gamma)\right)
\|e_k\|_\infty
+
\alpha
\|\widehat{\mathcal T}_kQ_k-\mathcal TQ_k\|_\infty.
$$

The proof therefore reduces to a uniform bound on the empirical Bellman-operator error. On the sufficient-visitation event, the reward estimator contributes

$$
\widetilde O\left(
\overline\sigma
\sqrt{\frac{1}{\lambda_{\min}H}}
\right)
+
O(\overline\sigma\sqrt{\varepsilon_R}),
$$

while the look-ahead estimator contributes

$$
\widetilde O\left(
\frac{\widetilde\sigma}{1-\gamma}
\sqrt{\frac{1}{\lambda_{\min}H}}
\right)
+
O\left(
\frac{\widetilde\sigma\varepsilon_Y}{1-\gamma}
\right).
$$

Unrolling the contractive recursion contributes one more factor $1/(1-\gamma)$. Since $K$ is only logarithmic in $T$, the epoch length $H=T/K$ is essentially $T$ up to logarithms, giving the final $T^{-1/2}$ rate.

## Why batching is the decisive idea

Batching accomplishes three things at once. It reduces variance by averaging many samples before an update. It freezes the bootstrap function, turning the clean look-ahead observations into samples of one fixed distribution. It also reduces the number of recursive updates from $T$ to $O(\log T)$, so corruption does not enter the dynamics at every time step.

The algorithm is still asynchronous in how data arrive: only one pair is observed at a time. It is synchronous in how the table is updated: after the epoch, every coordinate receives one robust operator update. This deliberate separation between asynchronous collection and synchronized learning is what permits the sharper coverage and corruption dependencies.

## Limitations and open directions

The method stores the reward and look-ahead samples collected during an epoch, so its auxiliary memory is $O(H)$. Replacing the batch estimators by online robust summaries could reduce this cost. The main theorem uses a stationary i.i.d. state-sampling simplification; extending the full argument directly to one Markov trajectory requires a careful blocking or coupling analysis. The algorithm also assumes knowledge of the corruption levels and suitable reward-scale bounds.

Nevertheless, the result establishes an important benchmark. Simultaneous reward and next-state corruption does not make discounted tabular $Q$-Learning hopeless. By estimating the Bellman operator robustly at the right temporal scale, one can retain the ordinary statistical rate and isolate the unavoidable price of each corruption channel.
