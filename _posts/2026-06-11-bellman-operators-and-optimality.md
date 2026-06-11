---
title: Bellman Operators and Bellman Optimality
date: 2026-06-11
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [reinforcement-learning, bellman-equation, bellman-operator, dynamic-programming, value-iteration, policy-iteration]
math: true
---

The Bellman equation is often introduced as a recursion. That is correct, but slightly misleading. The deeper view is that Bellman equations are **fixed-point equations** for certain operators acting on value functions.

This operator viewpoint is one of the cleanest ways to understand dynamic programming, policy evaluation, value iteration, policy iteration, Q-learning, temporal-difference learning, and many finite-time analyses in reinforcement learning.

There are two Bellman operators that appear again and again:

1. the **policy-specific Bellman operator**, which evaluates a fixed policy;
2. the **Bellman optimality operator**, which searches over actions and characterizes the optimal value function.

The goal of this post is to explain both operators carefully, prove their main properties, and give the geometric intuition behind why they are so powerful.

We will focus on finite discounted Markov decision processes.

---

## 1. Discounted Markov decision processes

Consider a finite discounted Markov decision process

$$
\mathcal{M}
=
(\mathcal{S},\mathcal{A},P,r,\gamma),
$$

where:

- $$\mathcal{S}$$ is a finite state space;
- $$\mathcal{A}$$ is a finite action space;
- $$P(s' \mid s,a)$$ is the transition probability from state $$s$$ to state $$s'$$ after taking action $$a$$;
- $$r(s,a)$$ is the expected one-step reward;
- $$\gamma \in [0,1)$$ is the discount factor.

A stationary randomized policy $$\pi$$ assigns a distribution over actions to every state:

$$
\pi(a\mid s)\ge 0,
\qquad
\sum_{a\in\mathcal{A}}\pi(a\mid s)=1.
$$

If the agent starts at state $$s_0=s$$ and follows policy $$\pi$$, the discounted value function is

$$
V^\pi(s)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{\infty}
\gamma^t r(s_t,a_t)
\mid s_0=s
\right].
$$

The optimal value function is

$$
V^\star(s)
=
\sup_{\pi} V^\pi(s).
$$

Since $$\mathcal{S}$$ and $$\mathcal{A}$$ are finite and $$\gamma<1$$, the infinite discounted sum is well-defined whenever rewards are bounded.

---

## 2. The policy-specific Bellman operator

Fix a policy $$\pi$$. Define the policy-induced reward vector

$$
r^\pi(s)
=
\sum_{a\in\mathcal{A}}
\pi(a\mid s)r(s,a),
$$

and the policy-induced transition matrix

$$
P^\pi(s,s')
=
\sum_{a\in\mathcal{A}}
\pi(a\mid s)P(s'\mid s,a).
$$

The policy-specific Bellman operator is the map

$$
\mathcal{T}^{\pi}:\mathbb{R}^{|\mathcal{S}|}\to \mathbb{R}^{|\mathcal{S}|}
$$

defined by

$$
(\mathcal{T}^{\pi} V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'\in\mathcal{S}}
P^\pi(s,s')V(s').
$$

In vector form,

$$
\mathcal{T}^{\pi} V
=
r^\pi+\gamma P^\pi V.
$$

Thus $$\mathcal{T}^{\pi}$$ is an affine operator. It first averages the future value using the transition matrix $$P^\pi$$, then discounts it by $$\gamma$$, and finally shifts it by the reward vector $$r^\pi$$.

The value function of policy $$\pi$$ satisfies the Bellman equation

$$
V^\pi
=
\mathcal{T}^{\pi} V^\pi.
$$

Equivalently,

$$
V^\pi
=
r^\pi+\gamma P^\pi V^\pi.
$$

Rearranging gives

$$
(I-\gamma P^\pi)V^\pi
=
r^\pi.
$$

Therefore,

$$
V^\pi
=
(I-\gamma P^\pi)^{-1}r^\pi.
$$

The inverse exists because $$\gamma<1$$ and $$P^\pi$$ is stochastic.

---

## 3. The value function as an infinite series

The equation

$$
V^\pi
=
(I-\gamma P^\pi)^{-1}r^\pi
$$

has a useful expansion. Since $$\gamma<1$$,

$$
(I-\gamma P^\pi)^{-1}
=
\sum_{t=0}^{\infty}\gamma^t(P^\pi)^t.
$$

Therefore,

$$
V^\pi
=
\sum_{t=0}^{\infty}
\gamma^t(P^\pi)^t r^\pi.
$$

This expression has a direct interpretation.

The first term is the immediate reward:

$$
r^\pi.
$$

The second term is the one-step-ahead expected reward:

$$
\gamma P^\pi r^\pi.
$$

The third term is the two-step-ahead expected reward:

$$
\gamma^2(P^\pi)^2r^\pi.
$$

Thus the Bellman equation is a compact fixed-point representation of the infinite discounted return.

The operator form is not merely notation. It exposes the geometry of dynamic programming.

---

## 4. Sup norm and value-function geometry

For a value function $$V\in\mathbb{R}^{\lvert\mathcal{S}\rvert}$$, define the sup norm

$$
\|V\|_\infty
=
\max_{s\in\mathcal{S}} |V(s)|.
$$

This norm measures the largest pointwise error across all states.

The space of value functions is the vector space

$$
\mathbb{R}^{|\mathcal{S}|}.
$$

Thus every value function is a point in a finite-dimensional Euclidean space. For example, if there are two states, then a value function is a point in the plane:

$$
V
=
(V(s_1),V(s_2)).
$$

A Bellman operator maps one point in this space to another point.

The policy Bellman operator

$$
\mathcal{T}^{\pi} V
=
r^\pi+\gamma P^\pi V
$$

is an affine map. It takes a value vector, averages it through $$P^\pi$$, shrinks the future part by $$\gamma$$, and translates it by $$r^\pi$$.

Its fixed point is the policy value function $$V^\pi$$.

---

## 5. Contraction of the policy Bellman operator

The most important property of $$\mathcal{T}^{\pi}$$ is contraction.

For any two value functions $$V,W\in\mathbb{R}^{\lvert\mathcal{S}\rvert}$$,

$$
\|\mathcal{T}^{\pi} V-\mathcal{T}^{\pi} W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

### Proof

For any state $$s$$,

$$
(\mathcal{T}^{\pi} V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')V(s'),
$$

and

$$
(\mathcal{T}^{\pi} W)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')W(s').
$$

Subtracting,

$$
(\mathcal{T}^{\pi} V)(s)-(\mathcal{T}^{\pi} W)(s)
=
\gamma
\sum_{s'}P^\pi(s,s')
\bigl(V(s')-W(s')\bigr).
$$

Taking absolute values,

$$
\left|
(\mathcal{T}^{\pi} V)(s)-(\mathcal{T}^{\pi} W)(s)
\right|
\le
\gamma
\sum_{s'}P^\pi(s,s')
|V(s')-W(s')|.
$$

Since

$$
|V(s')-W(s')|
\le
\|V-W\|_\infty
$$

for every $$s'$$,

$$
\left|
(\mathcal{T}^{\pi} V)(s)-(\mathcal{T}^{\pi} W)(s)
\right|
\le
\gamma
\|V-W\|_\infty
\sum_{s'}P^\pi(s,s').
$$

Because $$P^\pi$$ is stochastic,

$$
\sum_{s'}P^\pi(s,s')=1.
$$

Thus

$$
\left|
(\mathcal{T}^{\pi} V)(s)-(\mathcal{T}^{\pi} W)(s)
\right|
\le
\gamma
\|V-W\|_\infty.
$$

Taking the maximum over $$s$$ gives

$$
\|\mathcal{T}^{\pi} V-\mathcal{T}^{\pi} W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

This proves that $$\mathcal{T}^{\pi}$$ is a $$\gamma$$-contraction under the sup norm.

---

## 6. Geometric meaning of contraction

The contraction proof says that Bellman updates shrink value-function differences.

Let $$V^\pi$$ be the fixed point of $$\mathcal{T}^{\pi}$$. Then

$$
V^\pi
=
\mathcal{T}^{\pi} V^\pi.
$$

For any other value estimate $$V$$,

$$
\mathcal{T}^{\pi} V - V^\pi
=
\mathcal{T}^{\pi} V - \mathcal{T}^{\pi} V^\pi.
$$

Using the affine form,

$$
\mathcal{T}^{\pi} V - \mathcal{T}^{\pi} V^\pi
=
\gamma P^\pi(V-V^\pi).
$$

Therefore the error evolves as

$$
e^+
=
\gamma P^\pi e,
$$

where

$$
e=V-V^\pi.
$$

The matrix $$P^\pi$$ averages the error across next states. The discount factor $$\gamma$$ shrinks it.

So a Bellman update has a simple geometric meaning:

> it pulls the current value function closer to the fixed point $$V^\pi$$.

This is why exact policy evaluation converges.

---

## 7. Unique fixed point of the policy operator

Since $$\mathcal{T}^{\pi}$$ is a contraction on the complete metric space $$\mathbb{R}^{\lvert\mathcal{S}\rvert}$$ under $$\|\cdot\|_\infty$$, the Banach fixed-point theorem implies that $$\mathcal{T}^{\pi}$$ has a unique fixed point.

That unique fixed point is $$V^\pi$$.

We can also prove uniqueness directly.

Suppose $$U$$ and $$V$$ are both fixed points:

$$
U=\mathcal{T}^{\pi} U,
\qquad
V=\mathcal{T}^{\pi} V.
$$

Then

$$
\|U-V\|_\infty
=
\|\mathcal{T}^{\pi} U-\mathcal{T}^{\pi} V\|_\infty.
$$

By contraction,

$$
\|\mathcal{T}^{\pi} U-\mathcal{T}^{\pi} V\|_\infty
\le
\gamma\|U-V\|_\infty.
$$

Hence

$$
\|U-V\|_\infty
\le
\gamma\|U-V\|_\infty.
$$

Since $$\gamma<1$$,

$$
(1-\gamma)\|U-V\|_\infty\le 0.
$$

Therefore,

$$
\|U-V\|_\infty=0.
$$

Thus

$$
U=V.
$$

So the Bellman equation for a fixed policy has exactly one solution.

---

## 8. Policy evaluation by repeated Bellman updates

Given any initial value vector $$V_0$$, define

$$
V_{k+1}
=
\mathcal{T}^{\pi} V_k.
$$

Since $$V^\pi=\mathcal{T}^{\pi} V^\pi$$,

$$
\|V_{k+1}-V^\pi\|_\infty
=
\|\mathcal{T}^{\pi} V_k-\mathcal{T}^{\pi} V^\pi\|_\infty.
$$

Using contraction,

$$
\|V_{k+1}-V^\pi\|_\infty
\le
\gamma
\|V_k-V^\pi\|_\infty.
$$

Applying this recursively,

$$
\|V_k-V^\pi\|_\infty
\le
\gamma^k
\|V_0-V^\pi\|_\infty.
$$

Thus exact policy evaluation converges geometrically.

This is the core deterministic stability result behind dynamic programming.

---

## 9. Bellman residual bound for policy evaluation

The Bellman residual of a value function $$V$$ under policy $$\pi$$ is

$$
\|\mathcal{T}^{\pi} V-V\|_\infty.
$$

It measures how far $$V$$ is from satisfying the Bellman equation.

We claim that

$$
\|V-V^\pi\|_\infty
\le
\frac{1}{1-\gamma}
\|\mathcal{T}^{\pi} V-V\|_\infty.
$$

### Proof

Since $$V^\pi=\mathcal{T}^{\pi} V^\pi$$,

$$
V-V^\pi
=
V-\mathcal{T}^{\pi} V+\mathcal{T}^{\pi} V-\mathcal{T}^{\pi} V^\pi.
$$

Taking norms,

$$
\|V-V^\pi\|_\infty
\le
\|V-\mathcal{T}^{\pi} V\|_\infty
+
\|\mathcal{T}^{\pi} V-\mathcal{T}^{\pi} V^\pi\|_\infty.
$$

By contraction,

$$
\|\mathcal{T}^{\pi} V-\mathcal{T}^{\pi} V^\pi\|_\infty
\le
\gamma
\|V-V^\pi\|_\infty.
$$

Therefore,

$$
\|V-V^\pi\|_\infty
\le
\|V-\mathcal{T}^{\pi} V\|_\infty
+
\gamma
\|V-V^\pi\|_\infty.
$$

Rearranging,

$$
(1-\gamma)\|V-V^\pi\|_\infty
\le
\|V-\mathcal{T}^{\pi} V\|_\infty.
$$

Hence,

$$
\|V-V^\pi\|_\infty
\le
\frac{1}{1-\gamma}
\|\mathcal{T}^{\pi} V-V\|_\infty.
$$

So if a value function approximately satisfies the Bellman equation, then it is close to the true policy value function.

---

## 10. Monotonicity of the policy Bellman operator

The policy Bellman operator is monotone.

If

$$
V(s)\le W(s)
\qquad
\text{for all }s\in\mathcal{S},
$$

then

$$
(\mathcal{T}^{\pi} V)(s)\le (\mathcal{T}^{\pi} W)(s)
\qquad
\text{for all }s\in\mathcal{S}.
$$

### Proof

Assume $$V\le W$$ componentwise. Then for every state $$s$$,

$$
(\mathcal{T}^{\pi} V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')V(s'),
$$

and

$$
(\mathcal{T}^{\pi} W)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')W(s').
$$

Since $$V(s')\le W(s')$$ for every $$s'$$ and $$P^\pi(s,s')\ge 0$$,

$$
\sum_{s'}P^\pi(s,s')V(s')
\le
\sum_{s'}P^\pi(s,s')W(s').
$$

Therefore,

$$
(\mathcal{T}^{\pi} V)(s)
\le
(\mathcal{T}^{\pi} W)(s).
$$

So

$$
\mathcal{T}^{\pi} V\le \mathcal{T}^{\pi} W.
$$

---

## 11. Constant-shift property

Let $$\mathbf{1}$$ denote the all-ones vector. For any constant $$c\in\mathbb{R}$$,

$$
\mathcal{T}^{\pi}(V+c\mathbf{1})
=
\mathcal{T}^{\pi} V+\gamma c\mathbf{1}.
$$

### Proof

For every state $$s$$,

$$
(\mathcal{T}^{\pi}(V+c\mathbf{1}))(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')
\bigl(V(s')+c\bigr).
$$

Expanding,

$$
(\mathcal{T}^{\pi}(V+c\mathbf{1}))(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')V(s')
+
\gamma c
\sum_{s'}P^\pi(s,s').
$$

Since $$P^\pi$$ is stochastic,

$$
\sum_{s'}P^\pi(s,s')=1.
$$

Thus

$$
(\mathcal{T}^{\pi}(V+c\mathbf{1}))(s)
=
(\mathcal{T}^{\pi} V)(s)+\gamma c.
$$

Therefore,

$$
\mathcal{T}^{\pi}(V+c\mathbf{1})
=
\mathcal{T}^{\pi} V+\gamma c\mathbf{1}.
$$

This identity says that adding a constant to all future values only affects the current Bellman backup by the discounted amount $$\gamma c$$.

---

## 12. Bellman optimality operator

The policy-specific Bellman operator evaluates a fixed policy. The Bellman optimality operator chooses the best action at every state.

Define

$$
\mathcal{T}^{\star}:\mathbb{R}^{|\mathcal{S}|}\to\mathbb{R}^{|\mathcal{S}|}
$$

by

$$
(\mathcal{T}^{\star} V)(s)
=
\max_{a\in\mathcal{A}}
\left\{
r(s,a)
+
\gamma
\sum_{s'\in\mathcal{S}}
P(s'\mid s,a)V(s')
\right\}.
$$

The Bellman optimality equation is

$$
V^\star
=
\mathcal{T}^{\star} V^\star.
$$

In words:

> The optimal value of a state is the best immediate reward plus the discounted optimal value of the next state.

This is the mathematical foundation of optimal control in discounted MDPs.

---

## 13. Optimality operator as a maximum over policy operators

For a deterministic policy $$\pi$$,

$$
(\mathcal{T}^{\pi} V)(s)
=
r(s,\pi(s))
+
\gamma
\sum_{s'}P(s'\mid s,\pi(s))V(s').
$$

The optimality operator can be written as

$$
(\mathcal{T}^{\star} V)(s)
=
\max_{\pi}
(\mathcal{T}^{\pi} V)(s),
$$

where the maximum may be taken over deterministic stationary policies.

This means that $$\mathcal{T}^{\star}$$ is the pointwise maximum of many affine operators.

The operator $$\mathcal{T}^{\pi}$$ is affine:

$$
\mathcal{T}^{\pi} V
=
r^\pi+\gamma P^\pi V.
$$

The operator $$\mathcal{T}^{\star}$$ is generally nonlinear because of the maximum:

$$
(\mathcal{T}^{\star} V)(s)
=
\max_a
\left\{
r(s,a)+\gamma P_aV
\right\}.
$$

Here,

$$
P_aV
=
\sum_{s'}P(s'\mid s,a)V(s').
$$

Thus $$\mathcal{T}^{\star}$$ is piecewise affine. Different regions of value-function space correspond to different greedy actions.

---

## 14. Contraction of the Bellman optimality operator

Despite being nonlinear, $$\mathcal{T}^{\star}$$ is still a contraction.

For all $$V,W\in\mathbb{R}^{\lvert\mathcal{S}\rvert}$$,

$$
\|\mathcal{T}^{\star} V-\mathcal{T}^{\star} W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

### Proof

Fix a state $$s$$. Define

$$
F_a(V)
=
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s').
$$

Then

$$
(\mathcal{T}^{\star} V)(s)
=
\max_a F_a(V),
$$

and

$$
(\mathcal{T}^{\star} W)(s)
=
\max_a F_a(W).
$$

We use the elementary inequality

$$
\left|
\max_a x_a-\max_a y_a
\right|
\le
\max_a |x_a-y_a|.
$$

Therefore,

$$
\left|
(\mathcal{T}^{\star} V)(s)-(\mathcal{T}^{\star} W)(s)
\right|
\le
\max_a
|F_a(V)-F_a(W)|.
$$

Now,

$$
F_a(V)-F_a(W)
=
\gamma
\sum_{s'}P(s'\mid s,a)
\bigl(V(s')-W(s')\bigr).
$$

Taking absolute values,

$$
|F_a(V)-F_a(W)|
\le
\gamma
\sum_{s'}P(s'\mid s,a)
|V(s')-W(s')|.
$$

Since

$$
|V(s')-W(s')|
\le
\|V-W\|_\infty,
$$

we obtain

$$
|F_a(V)-F_a(W)|
\le
\gamma
\|V-W\|_\infty
\sum_{s'}P(s'\mid s,a).
$$

Because $$P(\cdot\mid s,a)$$ is a probability distribution,

$$
\sum_{s'}P(s'\mid s,a)=1.
$$

Thus,

$$
|F_a(V)-F_a(W)|
\le
\gamma
\|V-W\|_\infty.
$$

Taking the maximum over actions,

$$
\left|
(\mathcal{T}^{\star} V)(s)-(\mathcal{T}^{\star} W)(s)
\right|
\le
\gamma
\|V-W\|_\infty.
$$

Finally, taking the maximum over states gives

$$
\|\mathcal{T}^{\star} V-\mathcal{T}^{\star} W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

So $$\mathcal{T}^{\star}$$ is a $$\gamma$$-contraction under the sup norm.

---

## 15. Unique fixed point of the optimality operator

Since $$\mathcal{T}^{\star}$$ is a contraction, it has a unique fixed point.

That fixed point is $$V^\star$$.

Suppose $$U$$ and $$V$$ both satisfy

$$
U=\mathcal{T}^{\star} U,
\qquad
V=\mathcal{T}^{\star} V.
$$

Then

$$
\|U-V\|_\infty
=
\|\mathcal{T}^{\star} U-\mathcal{T}^{\star} V\|_\infty.
$$

By contraction,

$$
\|\mathcal{T}^{\star} U-\mathcal{T}^{\star} V\|_\infty
\le
\gamma
\|U-V\|_\infty.
$$

Since $$\gamma<1$$, this implies

$$
\|U-V\|_\infty=0.
$$

Therefore,

$$
U=V.
$$

Thus the Bellman optimality equation has exactly one solution.

---

## 16. Value iteration

Value iteration repeatedly applies the Bellman optimality operator:

$$
V_{k+1}
=
\mathcal{T}^{\star} V_k.
$$

Since $$V^\star=\mathcal{T}^{\star} V^\star$$,

$$
\|V_{k+1}-V^\star\|_\infty
=
\|\mathcal{T}^{\star} V_k-\mathcal{T}^{\star} V^\star\|_\infty.
$$

Using contraction,

$$
\|V_{k+1}-V^\star\|_\infty
\le
\gamma
\|V_k-V^\star\|_\infty.
$$

Repeating,

$$
\|V_k-V^\star\|_\infty
\le
\gamma^k
\|V_0-V^\star\|_\infty.
$$

Therefore value iteration converges geometrically to $$V^\star$$.

This is the basic convergence proof of value iteration.

---

## 17. Monotonicity of the optimality operator

The Bellman optimality operator is monotone.

If

$$
V\le W,
$$

then

$$
\mathcal{T}^{\star} V\le \mathcal{T}^{\star} W.
$$

### Proof

For every state-action pair $$(s,a)$$,

$$
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s')
\le
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)W(s').
$$

Taking the maximum over $$a$$ on both sides preserves the inequality:

$$
(\mathcal{T}^{\star} V)(s)
\le
(\mathcal{T}^{\star} W)(s).
$$

Thus,

$$
\mathcal{T}^{\star} V\le \mathcal{T}^{\star} W.
$$

---

## 18. Constant-shift property of the optimality operator

For any constant $$c\in\mathbb{R}$$,

$$
\mathcal{T}^{\star}(V+c\mathbf{1})
=
\mathcal{T}^{\star} V+\gamma c\mathbf{1}.
$$

### Proof

For each state $$s$$,

$$
(\mathcal{T}^{\star}(V+c\mathbf{1}))(s)
=
\max_a
\left\{
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)(V(s')+c)
\right\}.
$$

Expanding the sum,

$$
(\mathcal{T}^{\star}(V+c\mathbf{1}))(s)
=
\max_a
\left\{
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s')
+
\gamma c
\sum_{s'}P(s'\mid s,a)
\right\}.
$$

Since

$$
\sum_{s'}P(s'\mid s,a)=1,
$$

we get

$$
(\mathcal{T}^{\star}(V+c\mathbf{1}))(s)
=
\max_a
\left\{
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s')
+
\gamma c
\right\}.
$$

The term $$\gamma c$$ does not depend on $$a$$, so

$$
(\mathcal{T}^{\star}(V+c\mathbf{1}))(s)
=
(\mathcal{T}^{\star} V)(s)+\gamma c.
$$

Hence,

$$
\mathcal{T}^{\star}(V+c\mathbf{1})
=
\mathcal{T}^{\star} V+\gamma c\mathbf{1}.
$$

---

## 19. Greedy policies

Given a value function $$V$$, a greedy policy with respect to $$V$$ is any policy $$\pi_V$$ satisfying

$$
\pi_V(s)
\in
\arg\max_{a\in\mathcal{A}}
\left\{
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s')
\right\}.
$$

For such a policy,

$$
T^{\pi_V}V
=
\mathcal{T}^{\star} V.
$$

This identity is central.

It says that the optimality operator can be viewed as a policy-specific operator evaluated at a greedy policy.

However, one has to be careful. The identity

$$
T^{\pi_V}V
=
\mathcal{T}^{\star} V
$$

does not mean that $$\pi_V$$ is optimal. It only means that $$\pi_V$$ is greedy with respect to the current estimate $$V$$.

If $$V$$ is inaccurate, then the greedy policy may still be suboptimal.

But if $$V=V^\star$$, then any greedy policy is optimal.

---

## 20. Greedy policy from $$V^\star$$ is optimal

Let $$\pi^\star$$ be greedy with respect to $$V^\star$$. Then

$$
T^{\pi^\star}V^\star
=
\mathcal{T}^{\star} V^\star.
$$

Since $$V^\star$$ satisfies the Bellman optimality equation,

$$
\mathcal{T}^{\star} V^\star
=
V^\star.
$$

Therefore,

$$
T^{\pi^\star}V^\star
=
V^\star.
$$

So $$V^\star$$ is a fixed point of $$T^{\pi^\star}$$.

But the fixed point of $$T^{\pi^\star}$$ is unique and equals $$V^{\pi^\star}$$. Hence,

$$
V^{\pi^\star}
=
V^\star.
$$

Therefore $$\pi^\star$$ is an optimal policy.

This proves that solving the Bellman optimality equation gives an optimal policy by greedification.

---

## 21. Policy improvement theorem

The policy improvement theorem explains why greedy improvement works.

Let $$\pi$$ be a policy. Suppose another policy $$\pi'$$ satisfies

$$
T^{\pi'}V^\pi
\ge
\mathcal{T}^{\pi} V^\pi.
$$

Since

$$
\mathcal{T}^{\pi} V^\pi=V^\pi,
$$

this means

$$
T^{\pi'}V^\pi
\ge
V^\pi.
$$

We claim that

$$
V^{\pi'}\ge V^\pi.
$$

### Proof

Starting from

$$
T^{\pi'}V^\pi
\ge
V^\pi,
$$

apply $$T^{\pi'}$$ to both sides. Since $$T^{\pi'}$$ is monotone,

$$
(T^{\pi'})^2V^\pi
\ge
T^{\pi'}V^\pi.
$$

Applying the same argument repeatedly,

$$
(T^{\pi'})^kV^\pi
\ge
(T^{\pi'})^{k-1}V^\pi
\ge
\cdots
\ge
T^{\pi'}V^\pi
\ge
V^\pi.
$$

Since repeated application of $$T^{\pi'}$$ converges to its unique fixed point,

$$
(T^{\pi'})^kV^\pi
\to
V^{\pi'}.
$$

Taking limits gives

$$
V^{\pi'}\ge V^\pi.
$$

Thus $$\pi'$$ is at least as good as $$\pi$$.

If $$\pi'$$ is greedy with respect to $$V^\pi$$, then

$$
T^{\pi'}V^\pi
=
\mathcal{T}^{\star} V^\pi.
$$

Since

$$
\mathcal{T}^{\star} V^\pi
\ge
\mathcal{T}^{\pi} V^\pi
=
V^\pi,
$$

we conclude that the greedy policy improves the original policy.

---

## 22. Policy iteration

Policy iteration alternates between two steps.

First, policy evaluation:

$$
V^{\pi_k}
=
T^{\pi_k}V^{\pi_k}.
$$

Second, policy improvement:

$$
\pi_{k+1}
\in
\arg\max_{\pi}
\mathcal{T}^{\pi} V^{\pi_k}.
$$

In words:

1. compute the value of the current policy;
2. choose a new policy greedy with respect to that value.

The operator interpretation is elegant.

Each policy $$\pi$$ defines an affine contraction $$\mathcal{T}^{\pi}$$. Policy evaluation finds the fixed point of that contraction. Policy improvement then switches to another affine contraction whose one-step backup is no worse.

Thus policy iteration moves through a finite collection of affine contractions until it reaches one whose fixed point also satisfies the Bellman optimality equation.

---

## 23. Bellman residual for optimality

The optimal Bellman residual of a value function $$V$$ is

$$
\|\mathcal{T}^{\star} V-V\|_\infty.
$$

It measures how close $$V$$ is to satisfying the Bellman optimality equation.

We claim that

$$
\|V-V^\star\|_\infty
\le
\frac{1}{1-\gamma}
\|\mathcal{T}^{\star} V-V\|_\infty.
$$

### Proof

Since $$V^\star=\mathcal{T}^{\star} V^\star$$,

$$
V-V^\star
=
V-\mathcal{T}^{\star} V
+
\mathcal{T}^{\star} V-\mathcal{T}^{\star} V^\star.
$$

Taking norms,

$$
\|V-V^\star\|_\infty
\le
\|V-\mathcal{T}^{\star} V\|_\infty
+
\|\mathcal{T}^{\star} V-\mathcal{T}^{\star} V^\star\|_\infty.
$$

Using contraction,

$$
\|\mathcal{T}^{\star} V-\mathcal{T}^{\star} V^\star\|_\infty
\le
\gamma
\|V-V^\star\|_\infty.
$$

Therefore,

$$
\|V-V^\star\|_\infty
\le
\|V-\mathcal{T}^{\star} V\|_\infty
+
\gamma
\|V-V^\star\|_\infty.
$$

Rearranging,

$$
(1-\gamma)\|V-V^\star\|_\infty
\le
\|V-\mathcal{T}^{\star} V\|_\infty.
$$

Hence,

$$
\|V-V^\star\|_\infty
\le
\frac{1}{1-\gamma}
\|\mathcal{T}^{\star} V-V\|_\infty.
$$

This bound is extremely useful. It says that small Bellman residual implies small distance to the optimal value function.

---

## 24. Approximate greedy policies

Suppose $$V$$ is an approximation to $$V^\star$$ and $$\pi_V$$ is greedy with respect to $$V$$.

Assume

$$
\|V-V^\star\|_\infty\le \varepsilon.
$$

Then the greedy policy is near-optimal:

$$
\|V^\star-V^{\pi_V}\|_\infty
\le
\frac{2\gamma}{1-\gamma}\varepsilon.
$$

### Proof

Since $$\pi_V$$ is greedy with respect to $$V$$,

$$
T^{\pi_V}V
=
\mathcal{T}^{\star} V.
$$

Because $$\|V-V^\star\|_\infty\le \varepsilon$$,

$$
V^\star-\varepsilon\mathbf{1}
\le
V
\le
V^\star+\varepsilon\mathbf{1}.
$$

Using monotonicity and the shift property of $$\mathcal{T}^{\star}$$,

$$
\mathcal{T}^{\star} V
\ge
\mathcal{T}^{\star}(V^\star-\varepsilon\mathbf{1})
=
\mathcal{T}^{\star} V^\star-\gamma\varepsilon\mathbf{1}
=
V^\star-\gamma\varepsilon\mathbf{1}.
$$

Also, using the shift property of $$T^{\pi_V}$$,

$$
T^{\pi_V}V
\le
T^{\pi_V}(V^\star+\varepsilon\mathbf{1})
=
T^{\pi_V}V^\star+\gamma\varepsilon\mathbf{1}.
$$

Since $$T^{\pi_V}V=\mathcal{T}^{\star} V$$, we combine the inequalities:

$$
V^\star-\gamma\varepsilon\mathbf{1}
\le
T^{\pi_V}V^\star+\gamma\varepsilon\mathbf{1}.
$$

Therefore,

$$
V^\star
\le
T^{\pi_V}V^\star+2\gamma\varepsilon\mathbf{1}.
$$

Now apply $$T^{\pi_V}$$ repeatedly. By monotonicity,

$$
V^\star
\le
(T^{\pi_V})^kV^\star
+
2\gamma\varepsilon
\sum_{j=0}^{k-1}\gamma^j
\mathbf{1}.
$$

As $$k\to\infty$$,

$$
(T^{\pi_V})^kV^\star
\to
V^{\pi_V}.
$$

Also,

$$
\sum_{j=0}^{\infty}\gamma^j
=
\frac{1}{1-\gamma}.
$$

Hence,

$$
V^\star
\le
V^{\pi_V}
+
\frac{2\gamma}{1-\gamma}\varepsilon\mathbf{1}.
$$

Since $$V^\star\ge V^{\pi_V}$$ componentwise,

$$
0
\le
V^\star-V^{\pi_V}
\le
\frac{2\gamma}{1-\gamma}\varepsilon\mathbf{1}.
$$

Therefore,

$$
\|V^\star-V^{\pi_V}\|_\infty
\le
\frac{2\gamma}{1-\gamma}\varepsilon.
$$

This result explains why approximate value functions can still produce good policies.

---

## 25. State-action Bellman operators

Many RL algorithms work with state-action value functions rather than state values.

For a policy $$\pi$$, define

$$
Q^\pi(s,a)
=
\mathbb{E}_{\pi}
\left[
\sum_{t=0}^{\infty}
\gamma^t r(s_t,a_t)
\mid s_0=s,\ a_0=a
\right].
$$

The policy-specific Bellman operator for Q-functions is

$$
(\mathcal{T}^{\pi} Q)(s,a)
=
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)
\sum_{a'}\pi(a'\mid s')Q(s',a').
$$

The optimal Q-Bellman operator is

$$
(\mathcal{T}^{\star} Q)(s,a)
=
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)
\max_{a'}Q(s',a').
$$

The corresponding fixed-point equations are

$$
Q^\pi
=
\mathcal{T}^{\pi} Q^\pi,
$$

and

$$
Q^\star
=
\mathcal{T}^{\star} Q^\star.
$$

The optimal value function is recovered by

$$
V^\star(s)
=
\max_a Q^\star(s,a).
$$

For a fixed policy,

$$
V^\pi(s)
=
\sum_a\pi(a\mid s)Q^\pi(s,a).
$$

The Q-function operators satisfy the same contraction properties:

$$
\|\mathcal{T}^{\pi} Q_1-\mathcal{T}^{\pi} Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty,
$$

and

$$
\|\mathcal{T}^{\star} Q_1-\mathcal{T}^{\star} Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty.
$$

This is the deterministic fixed-point structure behind Q-learning.

---

## 26. Geometry of Bellman optimality

The policy operator

$$
\mathcal{T}^{\pi} V
=
r^\pi+\gamma P^\pi V
$$

is affine. It has one fixed point, and repeated application moves geometrically toward that point.

The optimality operator

$$
(\mathcal{T}^{\star} V)(s)
=
\max_a
\left\{
r(s,a)+\gamma P_aV
\right\}
$$

is nonlinear because of the maximum. But it is still a contraction.

Geometrically, each action defines an affine surface. The optimality operator takes the upper envelope of these surfaces.

Thus $$\mathcal{T}^{\star}$$ is piecewise affine:

- inside one region, action $$a_1$$ is greedy;
- inside another region, action $$a_2$$ is greedy;
- at the boundary, multiple actions may be tied.

Value iteration moves through these regions. The greedy action may change along the way. But regardless of these switches, contraction ensures convergence to $$V^\star$$. The Bellman optimality operator is nonlinear, but it is still stable because the future-value part is discounted.


## 27. Why Bellman contraction matters in RL theory

The contraction property is not just a clean mathematical fact. It is the reason many RL algorithms are stable.

In exact dynamic programming, contraction gives geometric convergence.

In TD learning, the stochastic update is a noisy approximation of a Bellman fixed-point iteration.

In Q-learning, the update is a noisy approximation of the optimal Q-Bellman operator.

In approximate dynamic programming, contraction allows one to convert Bellman residual bounds into value-function error bounds.

In robust RL, the contraction property explains why local errors caused by reward noise, transition noise, or adversarial corruption do not necessarily explode forever. Future errors are propagated through discounted powers:

$$
1,\gamma,\gamma^2,\gamma^3,\ldots.
$$

Their total contribution is governed by

$$
\sum_{t=0}^{\infty}\gamma^t
=
\frac{1}{1-\gamma}.
$$

This is why factors such as

$$
\frac{1}{1-\gamma},
\qquad
\frac{1}{(1-\gamma)^2},
\qquad
\frac{1}{(1-\gamma)^3}
$$

appear throughout RL theory.

The closer $$\gamma$$ is to one, the weaker the contraction becomes. Long-horizon problems are hard precisely because Bellman errors decay slowly.

## 28. Final takeaway

The Bellman equation is best understood as a fixed-point equation.

For a fixed policy,

$$
V^\pi
=
\mathcal{T}^{\pi} V^\pi,
$$

where

$$
\mathcal{T}^{\pi} V
=
r^\pi+\gamma P^\pi V.
$$

This operator is affine, monotone, and a $$\gamma$$-contraction. Repeated application converges geometrically to $$V^\pi$$.

For optimal control,

$$
V^\star
=
\mathcal{T}^{\star} V^\star,
$$

where

$$
(\mathcal{T}^{\star} V)(s)
=
\max_a
\left\{
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)V(s')
\right\}.
$$

This operator is nonlinear and piecewise affine, but it is still monotone and still a $$\gamma$$-contraction.

The policy-specific operator evaluates. The optimality operator improves. Policy iteration alternates between these two ideas. Value iteration directly applies the optimality operator.

Geometrically, Bellman operators move value functions through a high-dimensional space. The discount factor makes these movements contractive. The fixed points of these contractions are the value functions that define policy evaluation and optimal control.

That is the mathematical heart of dynamic programming in reinforcement learning.
