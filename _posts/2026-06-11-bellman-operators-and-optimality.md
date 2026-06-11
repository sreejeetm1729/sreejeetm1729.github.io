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
T^\pi:\mathbb{R}^{|\mathcal{S}|}\to \mathbb{R}^{|\mathcal{S}|}
$$

defined by

$$
(T^\pi V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'\in\mathcal{S}}
P^\pi(s,s')V(s').
$$

In vector form,

$$
T^\pi V
=
r^\pi+\gamma P^\pi V.
$$

Thus $$T^\pi$$ is an affine operator. It first averages the future value using the transition matrix $$P^\pi$$, then discounts it by $$\gamma$$, and finally shifts it by the reward vector $$r^\pi$$.

The value function of policy $$\pi$$ satisfies the Bellman equation

$$
V^\pi
=
T^\pi V^\pi.
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

For a value function $$V\in\mathbb{R}^{|\mathcal{S}|}$$, define the sup norm

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
T^\pi V
=
r^\pi+\gamma P^\pi V
$$

is an affine map. It takes a value vector, averages it through $$P^\pi$$, shrinks the future part by $$\gamma$$, and translates it by $$r^\pi$$.

Its fixed point is the policy value function $$V^\pi$$.

---

## 5. Contraction of the policy Bellman operator

The most important property of $$T^\pi$$ is contraction.

For any two value functions $$V,W\in\mathbb{R}^{|\mathcal{S}|}$$,

$$
\|T^\pi V-T^\pi W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

### Proof

For any state $$s$$,

$$
(T^\pi V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')V(s'),
$$

and

$$
(T^\pi W)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')W(s').
$$

Subtracting,

$$
(T^\pi V)(s)-(T^\pi W)(s)
=
\gamma
\sum_{s'}P^\pi(s,s')
\bigl(V(s')-W(s')\bigr).
$$

Taking absolute values,

$$
\left|
(T^\pi V)(s)-(T^\pi W)(s)
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
(T^\pi V)(s)-(T^\pi W)(s)
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
(T^\pi V)(s)-(T^\pi W)(s)
\right|
\le
\gamma
\|V-W\|_\infty.
$$

Taking the maximum over $$s$$ gives

$$
\|T^\pi V-T^\pi W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

This proves that $$T^\pi$$ is a $$\gamma$$-contraction under the sup norm.

---

## 6. Geometric meaning of contraction

The contraction proof says that Bellman updates shrink value-function differences.

Let $$V^\pi$$ be the fixed point of $$T^\pi$$. Then

$$
V^\pi
=
T^\pi V^\pi.
$$

For any other value estimate $$V$$,

$$
T^\pi V - V^\pi
=
T^\pi V - T^\pi V^\pi.
$$

Using the affine form,

$$
T^\pi V - T^\pi V^\pi
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

Since $$T^\pi$$ is a contraction on the complete metric space $$\mathbb{R}^{|\mathcal{S}|}$$ under $$\|\cdot\|_\infty$$, the Banach fixed-point theorem implies that $$T^\pi$$ has a unique fixed point.

That unique fixed point is $$V^\pi$$.

We can also prove uniqueness directly.

Suppose $$U$$ and $$V$$ are both fixed points:

$$
U=T^\pi U,
\qquad
V=T^\pi V.
$$

Then

$$
\|U-V\|_\infty
=
\|T^\pi U-T^\pi V\|_\infty.
$$

By contraction,

$$
\|T^\pi U-T^\pi V\|_\infty
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
T^\pi V_k.
$$

Since $$V^\pi=T^\pi V^\pi$$,

$$
\|V_{k+1}-V^\pi\|_\infty
=
\|T^\pi V_k-T^\pi V^\pi\|_\infty.
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
\|T^\pi V-V\|_\infty.
$$

It measures how far $$V$$ is from satisfying the Bellman equation.

We claim that

$$
\|V-V^\pi\|_\infty
\le
\frac{1}{1-\gamma}
\|T^\pi V-V\|_\infty.
$$

### Proof

Since $$V^\pi=T^\pi V^\pi$$,

$$
V-V^\pi
=
V-T^\pi V+T^\pi V-T^\pi V^\pi.
$$

Taking norms,

$$
\|V-V^\pi\|_\infty
\le
\|V-T^\pi V\|_\infty
+
\|T^\pi V-T^\pi V^\pi\|_\infty.
$$

By contraction,

$$
\|T^\pi V-T^\pi V^\pi\|_\infty
\le
\gamma
\|V-V^\pi\|_\infty.
$$

Therefore,

$$
\|V-V^\pi\|_\infty
\le
\|V-T^\pi V\|_\infty
+
\gamma
\|V-V^\pi\|_\infty.
$$

Rearranging,

$$
(1-\gamma)\|V-V^\pi\|_\infty
\le
\|V-T^\pi V\|_\infty.
$$

Hence,

$$
\|V-V^\pi\|_\infty
\le
\frac{1}{1-\gamma}
\|T^\pi V-V\|_\infty.
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
(T^\pi V)(s)\le (T^\pi W)(s)
\qquad
\text{for all }s\in\mathcal{S}.
$$

### Proof

Assume $$V\le W$$ componentwise. Then for every state $$s$$,

$$
(T^\pi V)(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')V(s'),
$$

and

$$
(T^\pi W)(s)
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
(T^\pi V)(s)
\le
(T^\pi W)(s).
$$

So

$$
T^\pi V\le T^\pi W.
$$

---

## 11. Constant-shift property

Let $$\mathbf{1}$$ denote the all-ones vector. For any constant $$c\in\mathbb{R}$$,

$$
T^\pi(V+c\mathbf{1})
=
T^\pi V+\gamma c\mathbf{1}.
$$

### Proof

For every state $$s$$,

$$
(T^\pi(V+c\mathbf{1}))(s)
=
r^\pi(s)
+
\gamma
\sum_{s'}P^\pi(s,s')
\bigl(V(s')+c\bigr).
$$

Expanding,

$$
(T^\pi(V+c\mathbf{1}))(s)
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
(T^\pi(V+c\mathbf{1}))(s)
=
(T^\pi V)(s)+\gamma c.
$$

Therefore,

$$
T^\pi(V+c\mathbf{1})
=
T^\pi V+\gamma c\mathbf{1}.
$$

This identity says that adding a constant to all future values only affects the current Bellman backup by the discounted amount $$\gamma c$$.

---

## 12. Bellman optimality operator

The policy-specific Bellman operator evaluates a fixed policy. The Bellman optimality operator chooses the best action at every state.

Define

$$
T^\star:\mathbb{R}^{|\mathcal{S}|}\to\mathbb{R}^{|\mathcal{S}|}
$$

by

$$
(T^\star V)(s)
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
T^\star V^\star.
$$

In words:

> The optimal value of a state is the best immediate reward plus the discounted optimal value of the next state.

This is the mathematical foundation of optimal control in discounted MDPs.

---

## 13. Optimality operator as a maximum over policy operators

For a deterministic policy $$\pi$$,

$$
(T^\pi V)(s)
=
r(s,\pi(s))
+
\gamma
\sum_{s'}P(s'\mid s,\pi(s))V(s').
$$

The optimality operator can be written as

$$
(T^\star V)(s)
=
\max_{\pi}
(T^\pi V)(s),
$$

where the maximum may be taken over deterministic stationary policies.

This means that $$T^\star$$ is the pointwise maximum of many affine operators.

The operator $$T^\pi$$ is affine:

$$
T^\pi V
=
r^\pi+\gamma P^\pi V.
$$

The operator $$T^\star$$ is generally nonlinear because of the maximum:

$$
(T^\star V)(s)
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

Thus $$T^\star$$ is piecewise affine. Different regions of value-function space correspond to different greedy actions.

---

## 14. Contraction of the Bellman optimality operator

Despite being nonlinear, $$T^\star$$ is still a contraction.

For all $$V,W\in\mathbb{R}^{|\mathcal{S}|}$$,

$$
\|T^\star V-T^\star W\|_\infty
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
(T^\star V)(s)
=
\max_a F_a(V),
$$

and

$$
(T^\star W)(s)
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
(T^\star V)(s)-(T^\star W)(s)
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
(T^\star V)(s)-(T^\star W)(s)
\right|
\le
\gamma
\|V-W\|_\infty.
$$

Finally, taking the maximum over states gives

$$
\|T^\star V-T^\star W\|_\infty
\le
\gamma
\|V-W\|_\infty.
$$

So $$T^\star$$ is a $$\gamma$$-contraction under the sup norm.

---

## 15. Unique fixed point of the optimality operator

Since $$T^\star$$ is a contraction, it has a unique fixed point.

That fixed point is $$V^\star$$.

Suppose $$U$$ and $$V$$ both satisfy

$$
U=T^\star U,
\qquad
V=T^\star V.
$$

Then

$$
\|U-V\|_\infty
=
\|T^\star U-T^\star V\|_\infty.
$$

By contraction,

$$
\|T^\star U-T^\star V\|_\infty
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
T^\star V_k.
$$

Since $$V^\star=T^\star V^\star$$,

$$
\|V_{k+1}-V^\star\|_\infty
=
\|T^\star V_k-T^\star V^\star\|_\infty.
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
T^\star V\le T^\star W.
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
(T^\star V)(s)
\le
(T^\star W)(s).
$$

Thus,

$$
T^\star V\le T^\star W.
$$

---

## 18. Constant-shift property of the optimality operator

For any constant $$c\in\mathbb{R}$$,

$$
T^\star(V+c\mathbf{1})
=
T^\star V+\gamma c\mathbf{1}.
$$

### Proof

For each state $$s$$,

$$
(T^\star(V+c\mathbf{1}))(s)
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
(T^\star(V+c\mathbf{1}))(s)
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
(T^\star(V+c\mathbf{1}))(s)
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
(T^\star(V+c\mathbf{1}))(s)
=
(T^\star V)(s)+\gamma c.
$$

Hence,

$$
T^\star(V+c\mathbf{1})
=
T^\star V+\gamma c\mathbf{1}.
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
T^\star V.
$$

This identity is central.

It says that the optimality operator can be viewed as a policy-specific operator evaluated at a greedy policy.

However, one has to be careful. The identity

$$
T^{\pi_V}V
=
T^\star V
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
T^\star V^\star.
$$

Since $$V^\star$$ satisfies the Bellman optimality equation,

$$
T^\star V^\star
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
T^\pi V^\pi.
$$

Since

$$
T^\pi V^\pi=V^\pi,
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
T^\star V^\pi.
$$

Since

$$
T^\star V^\pi
\ge
T^\pi V^\pi
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
T^\pi V^{\pi_k}.
$$

In words:

1. compute the value of the current policy;
2. choose a new policy greedy with respect to that value.

The operator interpretation is elegant.

Each policy $$\pi$$ defines an affine contraction $$T^\pi$$. Policy evaluation finds the fixed point of that contraction. Policy improvement then switches to another affine contraction whose one-step backup is no worse.

Thus policy iteration moves through a finite collection of affine contractions until it reaches one whose fixed point also satisfies the Bellman optimality equation.

---

## 23. Bellman residual for optimality

The optimal Bellman residual of a value function $$V$$ is

$$
\|T^\star V-V\|_\infty.
$$

It measures how close $$V$$ is to satisfying the Bellman optimality equation.

We claim that

$$
\|V-V^\star\|_\infty
\le
\frac{1}{1-\gamma}
\|T^\star V-V\|_\infty.
$$

### Proof

Since $$V^\star=T^\star V^\star$$,

$$
V-V^\star
=
V-T^\star V
+
T^\star V-T^\star V^\star.
$$

Taking norms,

$$
\|V-V^\star\|_\infty
\le
\|V-T^\star V\|_\infty
+
\|T^\star V-T^\star V^\star\|_\infty.
$$

Using contraction,

$$
\|T^\star V-T^\star V^\star\|_\infty
\le
\gamma
\|V-V^\star\|_\infty.
$$

Therefore,

$$
\|V-V^\star\|_\infty
\le
\|V-T^\star V\|_\infty
+
\gamma
\|V-V^\star\|_\infty.
$$

Rearranging,

$$
(1-\gamma)\|V-V^\star\|_\infty
\le
\|V-T^\star V\|_\infty.
$$

Hence,

$$
\|V-V^\star\|_\infty
\le
\frac{1}{1-\gamma}
\|T^\star V-V\|_\infty.
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
T^\star V.
$$

Because $$\|V-V^\star\|_\infty\le \varepsilon$$,

$$
V^\star-\varepsilon\mathbf{1}
\le
V
\le
V^\star+\varepsilon\mathbf{1}.
$$

Using monotonicity and the shift property of $$T^\star$$,

$$
T^\star V
\ge
T^\star(V^\star-\varepsilon\mathbf{1})
=
T^\star V^\star-\gamma\varepsilon\mathbf{1}
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

Since $$T^{\pi_V}V=T^\star V$$, we combine the inequalities:

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
(T^\pi Q)(s,a)
=
r(s,a)
+
\gamma
\sum_{s'}P(s'\mid s,a)
\sum_{a'}\pi(a'\mid s')Q(s',a').
$$

The optimal Q-Bellman operator is

$$
(T^\star Q)(s,a)
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
T^\pi Q^\pi,
$$

and

$$
Q^\star
=
T^\star Q^\star.
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
\|T^\pi Q_1-T^\pi Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty,
$$

and

$$
\|T^\star Q_1-T^\star Q_2\|_\infty
\le
\gamma
\|Q_1-Q_2\|_\infty.
$$

This is the deterministic fixed-point structure behind Q-learning.

---

## 26. Geometry of Bellman optimality

The policy operator

$$
T^\pi V
=
r^\pi+\gamma P^\pi V
$$

is affine. It has one fixed point, and repeated application moves geometrically toward that point.

The optimality operator

$$
(T^\star V)(s)
=
\max_a
\left\{
r(s,a)+\gamma P_aV
\right\}
$$

is nonlinear because of the maximum. But it is still a contraction.

Geometrically, each action defines an affine surface. The optimality operator takes the upper envelope of these surfaces.

Thus $$T^\star$$ is piecewise affine:

- inside one region, action $$a_1$$ is greedy;
- inside another region, action $$a_2$$ is greedy;
- at the boundary, multiple actions may be tied.

Value iteration moves through these regions. The greedy action may change along the way. But regardless of these switches, contraction ensures convergence to $$V^\star$$.

This is the key point:

> The Bellman optimality operator is nonlinear, but it is still stable because the future-value part is discounted.

---

## 27. Interactive Bellman contraction demo

The following small demo visualizes the contraction idea for a two-state Markov reward process. It repeatedly applies

$$
V_{k+1}
=
r+\gamma PV_k.
$$

The red point is the current value estimate. The green point is the Bellman fixed point.

<div class="bellman-demo">
  <div class="bellman-controls">
    <label>
      Discount $$\gamma$$:
      <input id="bellmanGamma" type="range" min="0.05" max="0.98" step="0.01" value="0.75">
      <span id="bellmanGammaVal">0.75</span>
    </label>

    <label>
      Initial $$V_1$$:
      <input id="bellmanV1" type="range" min="-10" max="10" step="0.5" value="8">
      <span id="bellmanV1Val">8</span>
    </label>

    <label>
      Initial $$V_2$$:
      <input id="bellmanV2" type="range" min="-10" max="10" step="0.5" value="-8">
      <span id="bellmanV2Val">-8</span>
    </label>

    <button id="bellmanReset">Reset</button>
    <button id="bellmanStep">One Bellman step</button>
    <button id="bellmanRun">Run</button>
  </div>

  <canvas id="bellmanCanvas" width="620" height="420"></canvas>

  <div class="bellman-readout">
    <p><strong>Current iterate:</strong> <span id="bellmanCurrent"></span></p>
    <p><strong>Fixed point:</strong> <span id="bellmanFixed"></span></p>
    <p><strong>Error:</strong> <span id="bellmanError"></span></p>
  </div>
</div>

<style>
.bellman-demo {
  border: 1px solid var(--main-border-color, #ddd);
  border-radius: 14px;
  padding: 1rem;
  margin: 1.5rem 0;
  background: var(--card-bg, #fff);
}

.bellman-controls {
  display: grid;
  grid-template-columns: 1fr;
  gap: 0.75rem;
  margin-bottom: 1rem;
}

.bellman-controls label {
  display: flex;
  align-items: center;
  gap: 0.65rem;
  flex-wrap: wrap;
}

.bellman-controls input[type="range"] {
  width: 220px;
}

.bellman-controls button {
  max-width: 180px;
  border: 1px solid var(--main-border-color, #ccc);
  border-radius: 8px;
  padding: 0.35rem 0.7rem;
  background: transparent;
  cursor: pointer;
}

#bellmanCanvas {
  width: 100%;
  max-width: 720px;
  height: auto;
  border: 1px solid var(--main-border-color, #ddd);
  border-radius: 12px;
  display: block;
  margin: 0 auto;
}

.bellman-readout {
  margin-top: 1rem;
  font-size: 0.95rem;
}
</style>

<script>
(function () {
  const canvas = document.getElementById("bellmanCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");

  const gammaSlider = document.getElementById("bellmanGamma");
  const v1Slider = document.getElementById("bellmanV1");
  const v2Slider = document.getElementById("bellmanV2");

  const gammaVal = document.getElementById("bellmanGammaVal");
  const v1Val = document.getElementById("bellmanV1Val");
  const v2Val = document.getElementById("bellmanV2Val");

  const currentText = document.getElementById("bellmanCurrent");
  const fixedText = document.getElementById("bellmanFixed");
  const errorText = document.getElementById("bellmanError");

  const resetBtn = document.getElementById("bellmanReset");
  const stepBtn = document.getElementById("bellmanStep");
  const runBtn = document.getElementById("bellmanRun");

  const P = [
    [0.75, 0.25],
    [0.35, 0.65]
  ];

  const r = [2.0, -1.0];

  let gamma = parseFloat(gammaSlider.value);
  let V = [parseFloat(v1Slider.value), parseFloat(v2Slider.value)];
  let running = false;
  let timer = null;

  function solveFixedPoint(g) {
    const a = 1 - g * P[0][0];
    const b = -g * P[0][1];
    const c = -g * P[1][0];
    const d = 1 - g * P[1][1];

    const det = a * d - b * c;

    const x = (d * r[0] - b * r[1]) / det;
    const y = (-c * r[0] + a * r[1]) / det;

    return [x, y];
  }

  function bellmanStep() {
    V = [
      r[0] + gamma * (P[0][0] * V[0] + P[0][1] * V[1]),
      r[1] + gamma * (P[1][0] * V[0] + P[1][1] * V[1])
    ];
    draw();
  }

  function toCanvas(point) {
    const scale = 18;
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    return [
      cx + scale * point[0],
      cy - scale * point[1]
    ];
  }

  function drawAxes() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.lineWidth = 1;
    ctx.strokeStyle = "#999";

    ctx.beginPath();
    ctx.moveTo(0, canvas.height / 2);
    ctx.lineTo(canvas.width, canvas.height / 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(canvas.width / 2, 0);
    ctx.lineTo(canvas.width / 2, canvas.height);
    ctx.stroke();

    ctx.fillStyle = "#666";
    ctx.font = "13px sans-serif";
    ctx.fillText("V(s1)", canvas.width - 55, canvas.height / 2 - 8);
    ctx.fillText("V(s2)", canvas.width / 2 + 8, 18);

    for (let x = -15; x <= 15; x += 5) {
      const px = canvas.width / 2 + 18 * x;
      ctx.beginPath();
      ctx.moveTo(px, canvas.height / 2 - 4);
      ctx.lineTo(px, canvas.height / 2 + 4);
      ctx.stroke();
      if (x !== 0) ctx.fillText(String(x), px - 7, canvas.height / 2 + 18);
    }

    for (let y = -10; y <= 10; y += 5) {
      const py = canvas.height / 2 - 18 * y;
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2 - 4, py);
      ctx.lineTo(canvas.width / 2 + 4, py);
      ctx.stroke();
      if (y !== 0) ctx.fillText(String(y), canvas.width / 2 + 8, py + 4);
    }
  }

  function drawPoint(point, color, label) {
    const [x, y] = toCanvas(point);
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = color;
    ctx.fill();

    ctx.fillStyle = color;
    ctx.font = "14px sans-serif";
    ctx.fillText(label, x + 9, y - 9);
  }

  function drawArrow(from, to) {
    const [x1, y1] = toCanvas(from);
    const [x2, y2] = toCanvas(to);

    ctx.strokeStyle = "#777";
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const angle = Math.atan2(y2 - y1, x2 - x1);
    const len = 9;

    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - len * Math.cos(angle - Math.PI / 6), y2 - len * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - len * Math.cos(angle + Math.PI / 6), y2 - len * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fillStyle = "#777";
    ctx.fill();
  }

  function draw() {
    gamma = parseFloat(gammaSlider.value);

    gammaVal.textContent = gamma.toFixed(2);
    v1Val.textContent = v1Slider.value;
    v2Val.textContent = v2Slider.value;

    const fixed = solveFixedPoint(gamma);

    const next = [
      r[0] + gamma * (P[0][0] * V[0] + P[0][1] * V[1]),
      r[1] + gamma * (P[1][0] * V[0] + P[1][1] * V[1])
    ];

    drawAxes();
    drawArrow(V, next);
    drawPoint(fixed, "#1b8a3a", "fixed point");
    drawPoint(V, "#c0392b", "current");

    const err = Math.max(Math.abs(V[0] - fixed[0]), Math.abs(V[1] - fixed[1]));

    currentText.textContent = "(" + V[0].toFixed(3) + ", " + V[1].toFixed(3) + ")";
    fixedText.textContent = "(" + fixed[0].toFixed(3) + ", " + fixed[1].toFixed(3) + ")";
    errorText.textContent = err.toFixed(4);
  }

  function reset() {
    V = [parseFloat(v1Slider.value), parseFloat(v2Slider.value)];
    draw();
  }

  resetBtn.addEventListener("click", function () {
    running = false;
    if (timer) clearInterval(timer);
    runBtn.textContent = "Run";
    reset();
  });

  stepBtn.addEventListener("click", function () {
    bellmanStep();
  });

  runBtn.addEventListener("click", function () {
    running = !running;

    if (running) {
      runBtn.textContent = "Pause";
      timer = setInterval(bellmanStep, 450);
    } else {
      runBtn.textContent = "Run";
      if (timer) clearInterval(timer);
    }
  });

  gammaSlider.addEventListener("input", draw);
  v1Slider.addEventListener("input", reset);
  v2Slider.addEventListener("input", reset);

  reset();
})();
</script>

---

## 28. Why Bellman contraction matters in RL theory

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

---

## 29. Summary table

| Object                   | Definition                                                                    | Role                        | Structure                              | Fixed point |
| ------------------------ | ----------------------------------------------------------------------------- | --------------------------- | -------------------------------------- | ----------- |
| Policy Bellman operator  | $$T^\pi V=r^\pi+\gamma P^\pi V$$                                              | Evaluates a fixed policy    | Affine contraction                     | $$V^\pi$$   |
| Optimal Bellman operator | $$(T^\star V)(s)=\max_a\{r(s,a)+\gamma P_aV\}$$                               | Optimizes over actions      | Nonlinear piecewise-affine contraction | $$V^\star$$ |
| Policy Q-operator        | $$(T^\pi Q)(s,a)=r(s,a)+\gamma\mathbb{E}_{s'}\mathbb{E}_{a'\sim\pi}Q(s',a')$$ | Evaluates policy in Q-space | Affine contraction                     | $$Q^\pi$$   |
| Optimal Q-operator       | $$(T^\star Q)(s,a)=r(s,a)+\gamma\mathbb{E}_{s'}\max_{a'}Q(s',a')$$            | Basis of Q-learning         | Nonlinear contraction                  | $$Q^\star$$ |

---

## 30. Final takeaway

The Bellman equation is best understood as a fixed-point equation.

For a fixed policy,

$$
V^\pi
=
T^\pi V^\pi,
$$

where

$$
T^\pi V
=
r^\pi+\gamma P^\pi V.
$$

This operator is affine, monotone, and a $$\gamma$$-contraction. Repeated application converges geometrically to $$V^\pi$$.

For optimal control,

$$
V^\star
=
T^\star V^\star,
$$

where

$$
(T^\star V)(s)
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
