---
title: "Discounted vs. Average Reward Reinforcement Learning"
date: 2026-06-09
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [reinforcement-learning, mdp, discounted-rl, average-reward, bellman-equation, q-learning]
math: true
---

In reinforcement learning, one of the first modeling choices is how we measure long-run performance. The most common formulation is the infinite-horizon discounted objective, where future rewards are geometrically downweighted by a discount factor $\gamma \in (0,1)$. Another equally important, but technically more delicate, formulation is the average-reward objective, where the agent is evaluated by its long-run reward per time step.

Both settings describe infinite-horizon decision-making. But mathematically, they are not the same problem. Discounted RL is governed by contraction mappings. Average-reward RL is governed by Poisson equations, bias functions, recurrence, mixing, and span seminorms.

This post gives a detailed mathematical comparison between the two settings.

---

## 1. The finite MDP model

Consider a finite Markov decision process

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r),
$$

where $\mathcal{S}$ is a finite state space, $\mathcal{A}$ is a finite action space,

$$
P(s'\mid s,a)
$$

is the transition probability of moving to state $s'$ after taking action $a$ in state $s$, and

$$
r(s,a)
$$

is the expected one-step reward.

A stationary deterministic policy $\pi:\mathcal{S}\to\mathcal{A}$ induces a Markov chain with transition matrix

$$
P_\pi(s'\mid s)
=
P(s'\mid s,\pi(s)),
$$

and reward vector

$$
r_\pi(s)
=
r(s,\pi(s)).
$$

The controlled process evolves as

$$
s_0,a_0,r_0,s_1,a_1,r_1,\ldots,
$$

where

$$
a_t=\pi(s_t),
\qquad
s_{t+1}\sim P(\cdot\mid s_t,a_t).
$$

The central question is:

> How should we evaluate the long-run quality of a policy?

The discounted setting and the average-reward setting answer this question differently.

---

## 2. The discounted infinite-horizon objective

In discounted RL, the value of a policy $\pi$ from state $s$ is

$$
V_\gamma^\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{\infty}
\gamma^t r(s_t,a_t)
\mid s_0=s
\right],
$$

where $\gamma\in(0,1)$.

The discount factor has two roles.

First, it makes the infinite sum finite whenever rewards are bounded. If

$$
|r(s,a)|\le R_{\max},
$$

then

$$
|V_\gamma^\pi(s)|
\le
\sum_{t=0}^{\infty}\gamma^t R_{\max}
=
\frac{R_{\max}}{1-\gamma}.
$$

Second, it creates a time preference: immediate rewards matter more than rewards far in the future.

The Bellman equation for a fixed policy is

$$
V_\gamma^\pi
=
r_\pi+\gamma P_\pi V_\gamma^\pi.
$$

Equivalently,

$$
(I-\gamma P_\pi)V_\gamma^\pi
=
r_\pi.
$$

Thus,

$$
V_\gamma^\pi
=
(I-\gamma P_\pi)^{-1}r_\pi.
$$

Because $\gamma<1$ and $P_\pi$ is stochastic, the inverse exists. Indeed,

$$
(I-\gamma P_\pi)^{-1}
=
\sum_{t=0}^{\infty}\gamma^t P_\pi^t.
$$

Therefore,

$$
V_\gamma^\pi
=
\sum_{t=0}^{\infty}
\gamma^t P_\pi^t r_\pi.
$$

This expression makes the meaning of $V_\gamma^\pi$ transparent: it is the discounted sum of expected future rewards.

---

## 3. Discounted optimality equation

The optimal discounted value function is

$$
V_\gamma^\star(s)
=
\sup_\pi V_\gamma^\pi(s).
$$

It satisfies the Bellman optimality equation

$$
V_\gamma^\star(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)V_\gamma^\star(s')
\right].
$$

Define the discounted Bellman optimality operator $\mathcal{T}_\gamma$ by

$$
(\mathcal{T}_\gamma V)(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)V(s')
\right].
$$

Then

$$
V_\gamma^\star
=
\mathcal{T}_\gamma V_\gamma^\star.
$$

The key fact is that $\mathcal{T}_\gamma$ is a contraction in the sup norm.

For any two value functions 
$$
V,W\in\mathbb{R}^{\lvert\mathcal{S}\rvert},
$$

$$
\|\mathcal{T}_\gamma V-\mathcal{T}_\gamma W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

Let us verify this. For each state $s$,

$$
\begin{aligned}
|(\mathcal{T}_\gamma V)(s)-(\mathcal{T}_\gamma W)(s)|
&\le
\gamma
\max_{a\in\mathcal{A}}
\left|
\sum_{s'}P(s'\mid s,a)(V(s')-W(s'))
\right|\\
&\le
\gamma
\max_{a\in\mathcal{A}}
\sum_{s'}P(s'\mid s,a)|V(s')-W(s')|\\
&\le
\gamma \|V-W\|_\infty.
\end{aligned}
$$

Taking the maximum over $s$ gives the contraction inequality.

This single inequality is the backbone of discounted RL theory. It implies uniqueness of $V_\gamma^\star$, convergence of value iteration, and stability of many stochastic approximation algorithms.

For value iteration,

$$
V_{k+1}
=
\mathcal{T}_\gamma V_k,
$$

we obtain

$$
\|V_k-V_\gamma^\star\|_\infty
\le
\gamma^k
\|V_0-V_\gamma^\star\|_\infty.
$$

The convergence is geometric, with rate $\gamma$.

---

## 4. Effective horizon in discounted RL

The discount factor determines an effective planning horizon. Since

$$
\sum_{t=0}^{\infty}\gamma^t
=
\frac{1}{1-\gamma},
$$

the quantity

$$
H_\gamma
\asymp
\frac{1}{1-\gamma}
$$

acts as the effective horizon.

When $\gamma$ is small, the agent mainly cares about short-term rewards. When $\gamma$ is close to one, the agent cares about rewards far into the future.

This is why many discounted RL bounds deteriorate polynomially in

$$
\frac{1}{1-\gamma}.
$$

For example, even the value magnitude satisfies

$$
\|V_\gamma^\star\|_\infty
\le
\frac{R_{\max}}{1-\gamma}.
$$

So as $\gamma\to 1$, the value function grows, the contraction weakens, and the problem becomes harder.

---

## 5. The average-reward objective

The average-reward criterion removes discounting. For a policy $\pi$, define

$$
\rho^\pi(s)
=
\lim_{T\to\infty}
\frac{1}{T}
\mathbb{E}_\pi
\left[
\sum_{t=0}^{T-1}
r(s_t,a_t)
\mid s_0=s
\right],
$$

whenever the limit exists.

If the Markov chain induced by $\pi$ is ergodic with stationary distribution $\mu_\pi$, then the average reward is independent of the initial state and equals

$$
\rho^\pi
=
\sum_{s\in\mathcal{S}}
\mu_\pi(s)r_\pi(s).
$$

Thus $\rho^\pi$ is the steady-state reward rate.

This is the natural objective for continuing tasks where there is no terminal time and no obvious reason to discount the future. Examples include queueing systems, wireless scheduling, inventory control, traffic control, robotic maintenance, and communication networks.

The average-reward objective asks:

> What reward per time step can the agent sustain in the long run?

---

## 6. Why we need a bias function

In the discounted setting, the value function is finite because rewards are discounted. In the average-reward setting, the undiscounted cumulative reward

$$
\mathbb{E}_\pi
\left[
\sum_{t=0}^{T-1}r(s_t,a_t)
\mid s_0=s
\right]
$$

typically grows linearly in $T$.

Indeed, if the long-run average reward is $\rho^\pi$, then for large $T$,

$$
\mathbb{E}_\pi
\left[
\sum_{t=0}^{T-1}r(s_t,a_t)
\mid s_0=s
\right]
\approx
T\rho^\pi
+
\text{transient correction}.
$$

The linear term $T\rho^\pi$ diverges as $T\to\infty$. Therefore, the total undiscounted return is not the right value object.

Instead, average-reward RL introduces the bias function, also called the relative value function.

For a fixed policy $\pi$, the average reward $\rho^\pi$ and bias function $h^\pi$ satisfy the Poisson equation

$$
\rho^\pi \mathbf{1}+h^\pi
=
r_\pi+P_\pi h^\pi.
$$

Componentwise,

$$
\rho^\pi+h^\pi(s)
=
r_\pi(s)
+
\sum_{s'}P_\pi(s'\mid s)h^\pi(s').
$$

Equivalently,

$$
h^\pi(s)
=
r_\pi(s)-\rho^\pi
+
\sum_{s'}P_\pi(s'\mid s)h^\pi(s').
$$

This equation says that the relative value of state $s$ is determined by the immediate reward advantage

$$
r_\pi(s)-\rho^\pi
$$

plus the expected future relative value.

A useful formal representation is

$$
h^\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{t=0}^{\infty}
\left(
r(s_t,a_t)-\rho^\pi
\right)
\mid s_0=s
\right],
$$

whenever this series is well-defined.

Thus $h^\pi(s)$ measures the transient surplus obtained by starting from $s$, after subtracting the steady-state reward rate at every time step.

---

## 7. The bias function is not unique

A crucial difference from discounted RL is that $h^\pi$ is not unique.

Suppose $h^\pi$ solves

$$
\rho^\pi \mathbf{1}+h^\pi
=
r_\pi+P_\pi h^\pi.
$$

Then for any constant $c\in\mathbb{R}$,

$$
\rho^\pi \mathbf{1}+(h^\pi+c\mathbf{1})
=
r_\pi+P_\pi(h^\pi+c\mathbf{1}).
$$

This is because

$$
P_\pi\mathbf{1}=\mathbf{1}.
$$

Therefore, adding a constant to $h^\pi$ gives another solution. Only differences such as

$$
h^\pi(s)-h^\pi(s')
$$

are meaningful.

To make the bias unique, one imposes a normalization. Common choices are

$$
h^\pi(s_0)=0
$$

for a reference state $s_0$, or

$$
\sum_s \mu_\pi(s)h^\pi(s)=0.
$$

This is fundamentally different from the discounted setting, where the Bellman equation has a unique solution without normalization.

---

## 8. Average-reward optimality equation

The optimal average reward is

$$
\rho^\star
=
\sup_\pi \rho^\pi.
$$

Under standard communicating or unichain assumptions, there exists a scalar $\rho^\star$ and a bias function $h^\star$ satisfying the average-reward optimality equation

$$
\rho^\star+h^\star(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)h^\star(s')
\right].
$$

A policy $\pi^\star$ is average-reward optimal if

$$
\pi^\star(s)
\in
\arg\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\sum_{s'}P(s'\mid s,a)h^\star(s')
\right].
$$

Compare this with the discounted optimality equation:

$$
V_\gamma^\star(s)
=
\max_a
\left[
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V_\gamma^\star(s')
\right].
$$

The discounted equation solves for one object:

$$
V_\gamma^\star.
$$

The average-reward equation solves for two objects:

$$
(\rho^\star,h^\star).
$$

The scalar $\rho^\star$ captures steady-state performance, while $h^\star$ captures transient state preference.

---

## 9. Why average-reward Bellman operators are harder

Define the average-reward Bellman operator

$$
(\mathcal{T}h)(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\sum_{s'}P(s'\mid s,a)h(s')
\right].
$$

Unlike the discounted Bellman operator, $\mathcal{T}$ is not generally a contraction in the ordinary sup norm.

The reason is simple. For any constant $c$,

$$
\mathcal{T}(h+c\mathbf{1})
=
\mathcal{T}h+c\mathbf{1}.
$$

Thus the operator is invariant to constant shifts. This invariance is not a nuisance; it is built into the average-reward problem.

Because of this, the correct geometry is not the usual sup norm. The correct object is often the span seminorm:

$$
\|h\|_{\mathrm{sp}}
=
\max_s h(s)-\min_s h(s).
$$

The span seminorm ignores additive constants:

$$
\|h+c\mathbf{1}\|_{\mathrm{sp}}
=
\|h\|_{\mathrm{sp}}.
$$

This matches the fact that $h$ is only defined up to a constant.

In some well-connected MDPs, the average-reward Bellman operator behaves like a contraction in the span seminorm, but this requires structural assumptions. In contrast, the discounted operator is automatically a contraction for every finite MDP as long as $\gamma<1$.

---

## 10. Relative value iteration

The average-reward analogue of value iteration is relative value iteration.

Choose a reference state $s_0$. Given $h_k$, define

$$
\widetilde{h}_{k+1}(s)
=
\max_{a\in\mathcal{A}}
\left[
r(s,a)
+
\sum_{s'}P(s'\mid s,a)h_k(s')
\right].
$$

Then normalize:

$$
h_{k+1}(s)
=
\widetilde{h}_{k+1}(s)
-
\widetilde{h}_{k+1}(s_0).
$$

The normalization removes the arbitrary additive constant. The average-reward estimate is often obtained from the removed shift:

$$
\rho_{k+1}
=
\widetilde{h}_{k+1}(s_0).
$$

This should be compared with discounted value iteration:

$$
V_{k+1}
=
\mathcal{T}_\gamma V_k.
$$

In discounted RL, no normalization is needed because the discount factor anchors the value scale. In average-reward RL, normalization is essential.

---

## 11. Q-functions in discounted RL

The discounted optimal Q-function satisfies

$$
Q_\gamma^\star(s,a)
=
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q_\gamma^\star(s',a').
$$

The classical Q-learning update is

$$
Q_{t+1}(s_t,a_t)
=
(1-\alpha_t)Q_t(s_t,a_t)
+
\alpha_t
\left[
r_t+
\gamma
\max_{a'}Q_t(s_{t+1},a')
\right].
$$

Equivalently,

$$
Q_{t+1}(s_t,a_t)
=
Q_t(s_t,a_t)
+
\alpha_t
\left[
r_t+
\gamma
\max_{a'}Q_t(s_{t+1},a')
-
Q_t(s_t,a_t)
\right].
$$

The target

$$
r_t+
\gamma
\max_{a'}Q_t(s_{t+1},a')
$$

is stable because of discounting.

---

## 12. Q-functions in average-reward RL

In average-reward RL, the optimal relative Q-function satisfies

$$
\rho^\star+q^\star(s,a)
=
r(s,a)
+
\sum_{s'}P(s'\mid s,a)
\max_{a'}q^\star(s',a').
$$

This is the Q-version of the average-reward optimality equation.

A stylized average-reward Q-learning update has the form

$$
Q_{t+1}(s_t,a_t)
=
Q_t(s_t,a_t)
+
\alpha_t
\left[
r_t-\rho_t
+
\max_{a'}Q_t(s_{t+1},a')
-
Q_t(s_t,a_t)
\right],
$$

together with an update for $\rho_t$.

The difference from discounted Q-learning is the subtraction of $\rho_t$. This term estimates the long-run reward rate. Without subtracting the average reward, the undiscounted target would drift because the cumulative reward grows linearly over time.

Thus average-reward learning is harder because the algorithm must learn both:

$$
\rho^\star
\quad\text{and}\quad
q^\star.
$$

Discounted Q-learning only needs to learn $Q_\gamma^\star$.

---

## 13. The discounted-to-average connection

A deep relationship between the two settings appears as $\gamma\to 1$.

For a fixed policy $\pi$,

$$
V_\gamma^\pi
=
\sum_{t=0}^{\infty}
\gamma^t P_\pi^t r_\pi.
$$

If the Markov chain induced by $\pi$ is ergodic, then

$$
P_\pi^t r_\pi
\to
\rho^\pi \mathbf{1}
$$

as $t\to\infty$.

Therefore, the discounted value contains a dominant steady-state term:

$$
\sum_{t=0}^{\infty}\gamma^t \rho^\pi
=
\frac{\rho^\pi}{1-\gamma}.
$$

This suggests the asymptotic expansion

$$
V_\gamma^\pi(s)
\approx
\frac{\rho^\pi}{1-\gamma}
+
h^\pi(s).
$$

More precisely, under suitable ergodicity assumptions,

$$
(1-\gamma)V_\gamma^\pi(s)
\to
\rho^\pi
$$

as $\gamma\to 1$.

Also, for a reference state $s_0$,

$$
V_\gamma^\pi(s)-V_\gamma^\pi(s_0)
\to
h^\pi(s)-h^\pi(s_0).
$$

This is the bridge between discounted and average-reward RL:

$$
\boxed{
V_\gamma^\pi(s)
\approx
\frac{\rho^\pi}{1-\gamma}
+
h^\pi(s)
}
$$

The scaled discounted value converges to the average reward, while the centered discounted value converges to the bias function.

---

## 14. Deriving the limiting Poisson equation

The discounted Bellman equation is

$$
V_\gamma^\pi
=
r_\pi+\gamma P_\pi V_\gamma^\pi.
$$

Suppose that as $\gamma\to 1$,

$$
V_\gamma^\pi
\approx
\frac{\rho^\pi}{1-\gamma}\mathbf{1}
+
h^\pi.
$$

Substitute this approximation into the discounted Bellman equation:

$$
\frac{\rho^\pi}{1-\gamma}\mathbf{1}
+
h^\pi
\approx
r_\pi
+
\gamma P_\pi
\left(
\frac{\rho^\pi}{1-\gamma}\mathbf{1}
+
h^\pi
\right).
$$

Since $P_\pi\mathbf{1}=\mathbf{1}$,

$$
\gamma P_\pi
\left(
\frac{\rho^\pi}{1-\gamma}\mathbf{1}
\right)
=
\frac{\gamma\rho^\pi}{1-\gamma}\mathbf{1}.
$$

Therefore,

$$
\frac{\rho^\pi}{1-\gamma}\mathbf{1}
+
h^\pi
\approx
r_\pi
+
\frac{\gamma\rho^\pi}{1-\gamma}\mathbf{1}
+
\gamma P_\pi h^\pi.
$$

Move the large terms together:

$$
\left(
\frac{\rho^\pi}{1-\gamma}
-
\frac{\gamma\rho^\pi}{1-\gamma}
\right)\mathbf{1}
+
h^\pi
\approx
r_\pi+\gamma P_\pi h^\pi.
$$

Since

$$
\frac{\rho^\pi}{1-\gamma}
-
\frac{\gamma\rho^\pi}{1-\gamma}
=
\rho^\pi,
$$

we get

$$
\rho^\pi\mathbf{1}+h^\pi
\approx
r_\pi+\gamma P_\pi h^\pi.
$$

Letting $\gamma\to 1$ gives

$$
\rho^\pi\mathbf{1}+h^\pi
=
r_\pi+P_\pi h^\pi.
$$

This is exactly the average-reward Poisson equation.

So the average-reward Bellman equation is not obtained by blindly setting $\gamma=1$ in the discounted equation. It is obtained after subtracting the divergent steady-state term.

---

## 15. Policy comparison in the two settings

In discounted RL, policies are compared statewise:

$$
\pi \text{ is better than } \pi'
\quad
\text{if}
\quad
V_\gamma^\pi(s)\ge V_\gamma^{\pi'}(s)
$$

for relevant starting states $s$.

In average-reward RL, under ergodicity, policies are compared by scalar reward rates:

$$
\rho^\pi
\quad\text{versus}\quad
\rho^{\pi'}.
$$

This means discounted optimality can depend on the initial state, while average-reward optimality often focuses on steady-state performance independent of the initial state.

As $\gamma\to 1$, discounted optimality begins to prioritize average reward. Indeed,

$$
V_\gamma^\pi(s)
\approx
\frac{\rho^\pi}{1-\gamma}+h^\pi(s).
$$

If $\rho^\pi>\rho^{\pi'}$, then for $\gamma$ sufficiently close to one,

$$
V_\gamma^\pi(s)
>
V_\gamma^{\pi'}(s),
$$

because the difference

$$
\frac{\rho^\pi-\rho^{\pi'}}{1-\gamma}
$$

dominates the bounded bias terms.

However, if two policies have the same average reward, then the bias terms can break ties. Thus the limiting discounted problem often performs a lexicographic selection:

1. first maximize average reward;
2. then among average-reward optimal policies, prefer better transient bias.

This is why the limit $\gamma\to 1$ is subtle.

---

## 16. A two-state example

Consider a fixed policy inducing a two-state Markov chain with

$$
\mathcal{S}=\{0,1\}
$$

and transition matrix

$$
P_\pi
=
\begin{pmatrix}
1-p & p\\
q & 1-q
\end{pmatrix},
$$

where $p,q\in(0,1)$. Let the rewards be

$$
r_\pi(0)=0,
\qquad
r_\pi(1)=1.
$$

The stationary distribution satisfies

$$
\mu_\pi^\top P_\pi=\mu_\pi^\top,
\qquad
\mu_\pi(0)+\mu_\pi(1)=1.
$$

Solving gives

$$
\mu_\pi(0)=\frac{q}{p+q},
\qquad
\mu_\pi(1)=\frac{p}{p+q}.
$$

Hence the average reward is

$$
\rho^\pi
=
\sum_s \mu_\pi(s)r_\pi(s)
=
\frac{p}{p+q}.
$$

Now consider the discounted value. It satisfies

$$
V_\gamma^\pi
=
r_\pi+\gamma P_\pi V_\gamma^\pi.
$$

As $\gamma\to 1$,

$$
(1-\gamma)V_\gamma^\pi(0)
\to
\frac{p}{p+q},
$$

and

$$
(1-\gamma)V_\gamma^\pi(1)
\to
\frac{p}{p+q}.
$$

The scaled values converge to the same number because the long-run reward rate is independent of the starting state.

But the centered values do not necessarily coincide:

$$
V_\gamma^\pi(1)-V_\gamma^\pi(0)
\to
h^\pi(1)-h^\pi(0).
$$

Starting from state $1$ is transiently better because state $1$ gives immediate reward. But in the long run, the chain forgets its starting state, and both states have the same average reward.

This example captures the entire story:

$$
\text{discounted value}
=
\text{large steady-state term}
+
\text{finite transient correction}.
$$

---

## 17. Where the difficulty parameters appear

In discounted RL, the main difficulty parameter is

$$
\frac{1}{1-\gamma}.
$$

This quantity controls the value scale, contraction strength, effective horizon, and error propagation.

In average-reward RL, there is no discount factor. The difficulty instead appears through structural properties of the Markov chain. Typical quantities include:

- the mixing time,
- the diameter of the MDP,
- hitting times between states,
- recurrence times,
- spectral gap,
- the span of the optimal bias function.

A particularly important quantity is

$$
\operatorname{span}(h^\star)
=
\max_s h^\star(s)-\min_s h^\star(s).
$$

If the MDP has slow transitions, rare good states, or bottlenecks, then the bias span can be large. Large bias span means that initial states can differ significantly in their transient advantage before steady-state behavior dominates.

Thus discounted RL hides long-horizon difficulty inside $1/(1-\gamma)$, while average-reward RL exposes the geometry of the Markov chain more directly.

---

## 18. Why discounted RL is mathematically cleaner

Discounted RL has several technical advantages:

1. The value function is finite for bounded rewards.
2. The Bellman operator is a contraction.
3. The fixed point is unique.
4. Value iteration has a short convergence proof.
5. Q-learning targets are naturally stable.
6. Finite-time analysis can exploit contraction directly.

Average-reward RL loses many of these conveniences:

1. The undiscounted cumulative reward diverges.
2. The value function must be replaced by a relative bias function.
3. The bias is unique only up to an additive constant.
4. The Bellman operator is not generally a sup-norm contraction.
5. The algorithm must learn both the reward rate and the bias.
6. Structural assumptions such as ergodicity, unichain structure, or communication become central.

So average-reward RL is not just discounted RL with $\gamma=1$. It is a different limiting regime with different mathematical objects.

---

## 19. When average reward is the better model

Despite being more delicate, average reward is often the more faithful objective for continuing systems.

For example, suppose a wireless scheduler repeatedly allocates channels to users. There may be no terminal time. The goal is not to maximize a geometrically discounted sum of throughput, but to maximize long-run throughput per time step.

Similarly, in queueing systems, server allocation, traffic control, and autonomous monitoring, the natural question is:

$$
\text{What steady-state performance can the controller sustain?}
$$

In such problems, discounting is often introduced for mathematical convenience rather than modeling accuracy.

Average reward directly optimizes

$$
\lim_{T\to\infty}
\frac{1}{T}
\mathbb{E}
\left[
\sum_{t=0}^{T-1}r_t
\right].
$$

The price is that the analysis must handle the actual recurrence and mixing structure of the Markov chain.

---

## 20. Summary table

| Aspect                    | Discounted RL                               | Average-Reward RL                                           |
| ------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| Objective                 | $\mathbb{E}\sum_{t=0}^{\infty}\gamma^t r_t$ | $\lim_{T\to\infty}\frac{1}{T}\mathbb{E}\sum_{t=0}^{T-1}r_t$ |
| Main value object         | $V_\gamma^\pi(s)$                           | $(\rho^\pi,h^\pi(s))$                                       |
| Policy Bellman equation   | $V=r+\gamma PV$                             | $\rho\mathbf{1}+h=r+Ph$                                     |
| Optimality equation       | Discounted Bellman equation                 | Average-reward optimality equation                          |
| Fixed-point uniqueness    | Unique value function                       | Bias unique only up to constants                            |
| Main norm                 | Sup norm                                    | Span seminorm                                               |
| Contraction               | $\gamma$-contraction                        | Not generally a sup-norm contraction                        |
| Main difficulty parameter | $1/(1-\gamma)$                              | Mixing time, diameter, bias span                            |
| Natural algorithms        | Value iteration, Q-learning                 | Relative value iteration, differential Q-learning           |
| Interpretation            | Time-preferenced total reward               | Long-run reward rate                                        |

---

## 21. Main takeaway

Discounted and average-reward RL both study infinite-horizon control, but they encode different notions of long-term performance.

Discounted RL asks:

> How much reward can I collect when future rewards are geometrically downweighted?

Average-reward RL asks:

> What reward rate can I sustain forever?

The discounted value function contains two pieces:

$$
V_\gamma^\pi(s)
\approx
\frac{\rho^\pi}{1-\gamma}
+
h^\pi(s).
$$

The first term is the divergent steady-state contribution. The second term is the finite transient bias.

Therefore,

$$
(1-\gamma)V_\gamma^\pi(s)
\to
\rho^\pi,
$$

while

$$
V_\gamma^\pi(s)-V_\gamma^\pi(s_0)
\to
h^\pi(s)-h^\pi(s_0).
$$

This is the cleanest mathematical bridge between the two settings.

Discounted RL is technically elegant because contraction does much of the work. Average-reward RL is technically richer because one must directly handle recurrence, normalization, span seminorms, and the geometry of the Markov chain.

For research, this distinction matters. A proof that works in discounted RL often relies deeply on $\gamma$-contraction and cannot be transferred directly to average reward. In average-reward problems, the real challenge is not discounting the future but understanding the long-run structure of the controlled Markov chain.
