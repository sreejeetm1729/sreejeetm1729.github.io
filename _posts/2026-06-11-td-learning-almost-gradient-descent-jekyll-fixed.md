---
title: "TD Learning Is Almost Gradient Descent: A Finite-Time View of Linear TD"
date: 2026-06-11
categories: [rl-blogs]
rl_section: research-papers
tags: [td-learning, linear-function-approximation, finite-time-analysis, markovian-noise, reinforcement-learning-theory]
math: true
description: "A mathematically detailed and intuitive blog-style discussion of Bhandari, Russo, and Singal's finite-time analysis of temporal-difference learning with linear function approximation."
---

This post is about one of my favorite papers in reinforcement learning theory:

> **Jalaj Bhandari, Daniel Russo, and Raghav Singal.**  
> **A Finite Time Analysis of Temporal Difference Learning With Linear Function Approximation.**  
> arXiv:1806.02450, 2018.

This was one of the first reinforcement learning theory papers I read carefully. I still like it because it explains TD learning through a very clean mathematical lens. The paper does not try to hide the main difficulty behind technical machinery. Instead, it isolates the central phenomenon:

> **TD learning is not stochastic gradient descent, but its expected update has enough gradient-like geometry to admit an SGD-style finite-time analysis.**

That sentence is the heart of the paper.

Temporal-difference learning is one of the simplest and most important algorithms in reinforcement learning. It updates a value-function estimate by bootstrapping from the next state. This makes the algorithm computationally light, online, and elegant. But the same bootstrapping also makes the analysis subtle. Unlike supervised learning, where we compare a prediction to a fixed target, TD compares the current prediction to a target that itself depends on the current parameter.

This post explains the paper from the perspective of geometry. I will focus mainly on TD(0) with linear function approximation, then explain how the same proof template extends to i.i.d. sampling, Markovian sampling, and TD($\lambda$).

---

## 1. The policy-evaluation problem

Fix a policy $\mu$ in a discounted Markov decision process. Once the policy is fixed, the system becomes a Markov reward process. There is a finite state space $\mathcal S$, transition matrix $P$, reward function $r$, and discount factor $\gamma \in (0,1)$.

The value function of the policy is

$$
V^\mu(s)
=
\mathbb E_\mu
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,s_{t+1})
\mid s_0=s
\right].
$$

In vector form, the value function satisfies the Bellman equation

$$
V^\mu
=
r+\gamma P V^\mu.
$$

Equivalently, if we define the Bellman operator

$$
\mathcal{T} V
=
r+\gamma P V,
$$

then

$$
V^\mu = \mathcal{T} V^\mu.
$$

The Bellman operator is a $\gamma$-contraction in appropriate norms. In the tabular setting, this already gives a clean way to understand value iteration and TD-style methods.

But the paper studies the more interesting case where the value function is approximated linearly.

---

## 2. Linear function approximation

Suppose each state $s$ has a feature vector

$$
\phi(s)\in \mathbb R^d.
$$

We approximate the value function by

$$
V_\theta(s)
=
\phi(s)^{\texttt{T}} \theta,
\qquad
\theta\in \mathbb R^d.
$$

Let $\Phi\in \mathbb R^{\lvert\mathcal S\rvert\times d}$ be the feature matrix whose row corresponding to state $s$ is $\phi(s)^{\texttt{T}}$. Then the vector of approximate values is

$$
V_\theta = \Phi \theta.
$$

Let $\pi$ denote the stationary distribution of the Markov chain induced by the policy, and define

$$
D = \operatorname{diag}(\pi).
$$

The natural prediction norm is the stationary-distribution weighted norm

$$
\|V\|_D^2
=
V^{\texttt{T}} D V
=
\sum_{s\in \mathcal S} \pi(s) V(s)^2.
$$

This norm measures prediction error on states according to how often the policy visits them.

The best linear approximation to a value function $V$ in this norm is obtained by projecting $V$ onto the span of the features. Let $\Pi_D$ denote the $D$-orthogonal projection onto the linear subspace

$$
\mathcal F
=
\{\Phi\theta:\theta\in\mathbb R^d\}.
$$

Thus,

$$
\Pi_D V
=
\arg\min_{u\in \mathcal F} \|V-u\|_D.
$$

The central object in linear TD theory is not the true Bellman fixed point $V^\mu$, because $V^\mu$ may not lie in the feature span. Instead, TD converges to the solution of the projected Bellman equation

$$
\Phi \theta^*
=
\Pi_D \mathcal{T}(\Phi \theta^*).
$$

This is the value-function approximation that is invariant under applying the Bellman operator and then projecting back into the feature space.

---

## 3. Why the projected Bellman equation matters

The equation

$$
\Phi\theta^*
=
\Pi_D \mathcal{T}(\Phi\theta^*)
$$

has a simple interpretation.

Start with an approximate value function $\Phi\theta$. Apply the Bellman operator:

$$
\mathcal{T}(\Phi\theta)
=
r+\gamma P\Phi\theta.
$$

This new vector usually does not lie in the span of the features. So we project it back:

$$
\Pi_D \mathcal{T}(\Phi\theta).
$$

A TD fixed point is a feature-space value function that remains unchanged under this operation.

Geometrically, TD is trying to find the intersection between the feature space and the Bellman-updated-and-projected feature space.

The projection equation also implies a useful orthogonality condition. Since $$\Phi\theta^*$$ is the $$D$$-projection of $$\mathcal{T}(\Phi\theta^*)$$ onto the feature space, the residual

$$
\mathcal{T}(\Phi\theta^*)-\Phi\theta^*
$$

is $D$-orthogonal to every vector in the feature space. Therefore, for every $\theta$,

$$
(\Phi\theta-\Phi\theta^*)^{\texttt{T}} D
\left(
\mathcal{T}(\Phi\theta^*)-\Phi\theta^*
\right)
=
0.
$$

This orthogonality is the hidden geometric fact that makes the finite-time proof work.

---

## 4. The TD(0) update

Given a transition

$$
(s_t,r_t,s_t'),
$$

TD(0) computes the temporal-difference error

$$
\delta_t(\theta_t)
=
r_t
+
\gamma \phi(s_t')^{\texttt{T}} \theta_t
-
\phi(s_t)^{\texttt{T}} \theta_t.
$$

Then it updates

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t
\phi(s_t)
\delta_t(\theta_t).
$$

Equivalently,

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t g_t(\theta_t),
$$

where

$$
g_t(\theta)
=
\phi(s_t)
\left(
r_t+\gamma \phi(s_t')^{\texttt{T}}\theta-\phi(s_t)^{\texttt{T}}\theta
\right).
$$

The expected TD update under the stationary distribution is

$$
\bar g(\theta)
=
\mathbb E[g_t(\theta)].
$$

A direct calculation gives

$$
\bar g(\theta)
=
\Phi^{\texttt{T}} D
\left(
\mathcal{T}(\Phi\theta)-\Phi\theta
\right).
$$

Equivalently,

$$
\bar g(\theta)
=
b-A\theta,
$$

where

$$
A
=
\Phi^{\texttt{T}} D(I-\gamma P)\Phi,
\qquad
b
=
\Phi^{\texttt{T}} D r.
$$

The TD fixed point satisfies

$$
\bar g(\theta^*)=0,
$$

or

$$
A\theta^*=b.
$$

At first glance, this looks exactly like stochastic approximation for solving a linear system. But there is a catch.

The matrix

$$
A=\Phi^{\texttt{T}} D(I-\gamma P)\Phi
$$

is generally **not symmetric**. Therefore the update

$$
\theta_{t+1}
=
\theta_t+\alpha_t(b-A\theta_t)
$$

is not, in general, gradient descent on a fixed quadratic objective.

For ordinary least squares, the expected stochastic gradient has the form

$$
-\nabla f(\theta)
=
b-H\theta
$$

with $H$ symmetric positive semidefinite. TD has a similar linear form, but the matrix $A$ can be nonsymmetric because the Markov transition matrix $P$ is not symmetric.

So TD is not simply SGD.

The point of the paper is that, despite this, TD has enough of the same geometry.

---

## 5. The key question

For gradient descent on a convex function $f$, the expected update direction points toward the minimizer. In a strongly convex quadratic, one has something like

$$
(\theta^*-\theta)^{\texttt{T}} (-\nabla f(\theta))
\ge
c\|\theta-\theta^*\|^2.
$$

This says that the update direction is positively aligned with the direction from the current point to the optimum.

For TD, the analogous question is:

> Is the expected TD update $\bar g(\theta)$ aligned with the direction $\theta^*-\theta$?

Surprisingly, the answer is yes, but the cleanest alignment is not in parameter norm. It is in value-function prediction norm.

The central inequality is

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
\ge
(1-\gamma)
\|V_{\theta^*}-V_\theta\|_D^2.
$$

This is the paper's most important geometric insight.

It says that even though TD is not gradient descent, the expected TD update still makes progress toward the TD fixed point in the prediction geometry induced by the stationary distribution.

---

## 6. Deriving the key inequality

Let

$$
e
=
V_{\theta^*}-V_\theta
=
\Phi(\theta^*-\theta).
$$

Then

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
=
(\theta^*-\theta)^{\texttt{T}}
\Phi^{\texttt{T}} D
\left(
\mathcal{T} V_\theta - V_\theta
\right).
$$

Since $e=\Phi(\theta^*-\theta)$, this becomes

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
=
e^{\texttt{T}} D
\left(
\mathcal{T} V_\theta - V_\theta
\right).
$$

Now add and subtract $\mathcal{T} V_{\theta^*}$ and $V_{\theta^*}$:

$$
\mathcal{T} V_\theta - V_\theta
=
\left(\mathcal{T}V_\theta-\mathcal{T}V_{\theta^*}\right)
+
\left(\mathcal{T}V_{\theta^*}-V_{\theta^*}\right)
+
\left(V_{\theta^*}-V_\theta\right).
$$

Therefore,

$$
e^{\texttt{T}} D(\mathcal{T}V_\theta - V_\theta)
=
e^{\texttt{T}} D(\mathcal{T}V_\theta-\mathcal{T}V_{\theta^*})
+
e^{\texttt{T}} D(\mathcal{T}V_{\theta^*}-V_{\theta^*})
+
e^{\texttt{T}} D e.
$$

The middle term vanishes by the projected Bellman orthogonality:

$$
e^{\texttt{T}} D(\mathcal{T}V_{\theta^*}-V_{\theta^*})=0.
$$

Also,

$$
e^{\texttt{T}} D e
=
\|e\|_D^2.
$$

Since

$$
\mathcal{T}V_\theta-\mathcal{T}V_{\theta^*}
=
\gamma P(V_\theta-V_{\theta^*})
=
-\gamma P e,
$$

we get

$$
e^{\texttt{T}} D(\mathcal{T}V_\theta-\mathcal{T}V_{\theta^*})
=
-\gamma e^{\texttt{T}} D P e.
$$

Hence

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
=
\|e\|_D^2
-
\gamma e^{\texttt{T}} D P e.
$$

Now use the fact that $P$ is Markov and $\pi$ is stationary. By Jensen's inequality,

$$
\|Pe\|_D
\le
\|e\|_D.
$$

Therefore,

$$
e^{\texttt{T}} D P e
\le
\|e\|_D\|Pe\|_D
\le
\|e\|_D^2.
$$

Thus,

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
\ge
(1-\gamma)\|e\|_D^2.
$$

Since $e=V_{\theta^*}-V_\theta$, we obtain

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
\ge
(1-\gamma)
\|V_{\theta^*}-V_\theta\|_D^2.
$$

This is the first key inequality.

---

## 7. Intuition behind the inequality

The inequality says:

> The expected TD update may not be a gradient, but it has positive correlation with the direction to the TD fixed point.

The factor $1-\gamma$ is natural. When $\gamma$ is close to $1$, the Bellman operator becomes less contractive. Future values matter more, bootstrapping propagates information more slowly, and the TD direction becomes less strongly aligned with the fixed point.

This is why nearly every finite-time discounted RL bound worsens as $\gamma\to 1$.

The inequality also explains why the prediction norm is the right norm. TD is trying to learn values on states visited by the Markov chain. The stationary distribution $D$ is therefore not an arbitrary technical object. It is the geometry in which the Bellman operator and the data distribution naturally match.

---

## 8. The second key inequality: the update is not too large

A progress inequality alone is not enough. To analyze an iterative method, one also needs to control the size of the update.

For gradient descent, the corresponding property is a smoothness bound. For TD, the paper proves an analogous inequality of the form

$$
\|\bar g(\theta)\|_2
\le
2
\|V_{\theta^*}-V_\theta\|_D.
$$

The exact constants depend on feature normalization, but the important message is this:

> The expected TD update cannot be large unless the current value prediction is far from the TD fixed point.

Together, the two inequalities say that the mean TD update is both aligned with the error and controlled by the error.

This is exactly the geometry one needs to mimic a gradient-descent proof.

---

## 9. Mean-path TD: the deterministic recursion

Before analyzing stochastic TD, the paper studies the deterministic recursion

$$
\theta_{t+1}
=
\theta_t+\alpha \bar g(\theta_t).
$$

This is the mean-path TD update. It is what TD would do if we had direct access to the expected update.

Let

$$
\Delta_t
=
\theta^*-\theta_t.
$$

Then

$$
\Delta_{t+1}
=
\theta^*-\theta_{t+1}
=
\Delta_t-\alpha \bar g(\theta_t).
$$

Expanding the squared distance gives

$$
\|\Delta_{t+1}\|_2^2
=
\|\Delta_t\|_2^2
-
2\alpha \Delta_t^{\texttt{T}} \bar g(\theta_t)
+
\alpha^2\|\bar g(\theta_t)\|_2^2.
$$

Now use the two key inequalities:

$$
\Delta_t^{\texttt{T}} \bar g(\theta_t)
\ge
(1-\gamma)
\|V_{\theta^*}-V_{\theta_t}\|_D^2
$$

and

$$
\|\bar g(\theta_t)\|_2^2
\le
4
\|V_{\theta^*}-V_{\theta_t}\|_D^2.
$$

Therefore,

$$
\|\Delta_{t+1}\|_2^2
\le
\|\Delta_t\|_2^2
-
2\alpha(1-\gamma)
\|V_{\theta^*}-V_{\theta_t}\|_D^2
+
4\alpha^2
\|V_{\theta^*}-V_{\theta_t}\|_D^2.
$$

Choose

$$
\alpha=\frac{1-\gamma}{4}.
$$

Then

$$
2\alpha(1-\gamma)-4\alpha^2
=
\frac{(1-\gamma)^2}{4}.
$$

Hence

$$
\|\Delta_{t+1}\|_2^2
\le
\|\Delta_t\|_2^2
-
\frac{(1-\gamma)^2}{4}
\|V_{\theta^*}-V_{\theta_t}\|_D^2.
$$

This is a descent inequality.

If we sum from $t=0$ to $T-1$, the left side telescopes:

$$
\frac{(1-\gamma)^2}{4}
\sum_{t=0}^{T-1}
\|V_{\theta^*}-V_{\theta_t}\|_D^2
\le
\|\theta^*-\theta_0\|_2^2.
$$

Therefore,

$$
\frac{1}{T}
\sum_{t=0}^{T-1}
\|V_{\theta^*}-V_{\theta_t}\|_D^2
\le
\frac{4\|\theta^*-\theta_0\|_2^2}
{T(1-\gamma)^2}.
$$

For the averaged iterate

$$
\bar\theta_T
=
\frac{1}{T}\sum_{t=0}^{T-1}\theta_t,
$$

Jensen's inequality gives

$$
\|V_{\theta^*}-V_{\bar\theta_T}\|_D^2
\le
\frac{4\|\theta^*-\theta_0\|_2^2}
{T(1-\gamma)^2}.
$$

This is an $O(1/T)$ deterministic mean-path prediction-error bound.

---

## 10. The role of feature conditioning

The previous bound controls prediction error. To control parameter error, we need a relationship between

$$
\|\theta-\theta^*\|_2
$$

and

$$
\|V_\theta-V_{\theta^*}\|_D.
$$

Define the feature covariance matrix

$$
\Sigma
=
\Phi^{\texttt{T}} D\Phi.
$$

Then

$$
\|V_\theta-V_{\theta^*}\|_D^2
=
(\theta-\theta^*)^{\texttt{T}} \Sigma(\theta-\theta^*).
$$

If the minimum eigenvalue of $\Sigma$ is

$$
\omega=\lambda_{\min}(\Sigma)>0,
$$

then

$$
\|V_\theta-V_{\theta^*}\|_D^2
\ge
\omega
\|\theta-\theta^*\|_2^2.
$$

This is the feature-conditioning assumption. It says that no nonzero parameter direction is invisible under the stationary distribution.

Using this, the mean-path recursion also gives a geometric parameter convergence bound:

$$
\|\theta_T-\theta^*\|_2^2
\le
\exp\left(
-\frac{(1-\gamma)^2\omega}{4}T
\right)
\|\theta_0-\theta^*\|_2^2.
$$

This bound is very intuitive.

The convergence rate worsens when:

1. $\gamma$ is close to $1$,
2. the feature covariance is ill-conditioned,
3. some feature directions are rarely visited under the stationary distribution.

The first issue is an RL horizon issue. The second and third are representation and exploration issues.

---

## 11. From mean-path TD to stochastic TD

Real TD does not use the expected update $\bar g(\theta_t)$. It uses a random sample update

$$
g_t(\theta_t)
=
\phi(s_t)
\left(
r_t+\gamma\phi(s_t')^{\texttt{T}}\theta_t-\phi(s_t)^{\texttt{T}}\theta_t
\right).
$$

Thus,

$$
\theta_{t+1}
=
\theta_t+\alpha_t g_t(\theta_t).
$$

Under an i.i.d. observation model, the transition tuples are sampled independently from the stationary distribution. Then

$$
\mathbb E[g_t(\theta)\mid \theta]
=
\bar g(\theta).
$$

So the stochastic TD update is an unbiased noisy version of the mean-path update.

This gives the decomposition

$$
g_t(\theta_t)
=
\bar g(\theta_t)
+
\xi_t,
$$

where

$$
\mathbb E[\xi_t\mid \theta_t]=0.
$$

The proof then follows the same structure as SGD.

Expanding the squared distance,

$$
\|\theta^*-\theta_{t+1}\|_2^2
=
\|\theta^*-\theta_t\|_2^2
-
2\alpha_t(\theta^*-\theta_t)^{\texttt{T}} g_t(\theta_t)
+
\alpha_t^2\|g_t(\theta_t)\|_2^2.
$$

Taking conditional expectation removes the martingale-noise term in the linear part:

$$
\mathbb E[g_t(\theta_t)\mid \theta_t]
=
\bar g(\theta_t).
$$

The first-order term gives progress, and the second-order term creates variance.

This is exactly the same bias-variance structure as SGD.

---

## 12. The i.i.d. finite-time picture

The paper proves three types of guarantees under the i.i.d. observation model.

### 12.1 Constant step size of order $1/\sqrt T$

If

$$
\alpha_t=\frac{1}{\sqrt T},
$$

then the averaged iterate satisfies a bound of the form

$$
\mathbb E
\left[
\|V_{\theta^*}-V_{\bar\theta_T}\|_D^2
\right]
\lesssim
\frac{
\|\theta^*-\theta_0\|_2^2+\sigma^2
}
{\sqrt T(1-\gamma)}.
$$

Here

$$
\sigma^2
=
\mathbb E
\left[
\|g_t(\theta^*)\|_2^2
\right]
$$

measures the variance of TD updates at the fixed point.

This is a robust, problem-independent bound. It does not require knowing the feature-conditioning parameter $\omega$.

The cost is the slower $O(1/\sqrt T)$ rate.

---

### 12.2 Small constant step size

With a sufficiently small constant step size,

$$
\alpha
\le
\frac{\omega(1-\gamma)}{8},
$$

the expected parameter error satisfies a bound of the form

$$
\mathbb E
\left[
\|\theta_T-\theta^*\|_2^2
\right]
\le
\exp\left(
-\alpha(1-\gamma)\omega T
\right)
\|\theta_0-\theta^*\|_2^2
+
O\left(
\frac{\alpha\sigma^2}{(1-\gamma)\omega}
\right).
$$

This is the standard constant-step-size story.

There are two terms:

1. an exponentially decaying optimization term,
2. a nonzero noise floor proportional to $\alpha$.

The initial condition is forgotten exponentially fast, but the algorithm does not converge exactly because the step size never vanishes.

This is precisely the same qualitative behavior as constant-step-size SGD.

---

### 12.3 Decaying step size and $O(1/T)$ behavior

With a carefully tuned decaying step size,

$$
\alpha_t
=
\frac{\beta}{\lambda+t},
$$

where the constants depend on $(1-\gamma)$ and $\omega$, the paper obtains an $O(1/T)$ parameter mean-squared error bound.

Ignoring constants, the behavior is

$$
\mathbb E
\left[
\|\theta_T-\theta^*\|_2^2
\right]
=
O\left(
\frac{\sigma^2}
{(1-\gamma)^2\omega^2T}
\right).
$$

This is the fast stochastic-approximation rate. But it requires problem-dependent tuning, especially knowledge of the feature-conditioning parameter $\omega$.

This is an important practical message:

> Fast rates are possible, but aggressive tuning requires knowing how well-conditioned the feature representation is under the stationary distribution.

---

## 13. Why the i.i.d. result is not enough

The i.i.d. model is mathematically clean but not how TD is usually run.

In online reinforcement learning, the samples come from a single trajectory:

$$
s_0,s_1,s_2,\ldots
$$

The update at time $t$ uses

$$
(s_t,r_t,s_{t+1}).
$$

These samples are not independent. Worse, the current parameter $\theta_t$ is itself a function of the past trajectory. Since $s_t$ is correlated with the past, the current sample is correlated with the current parameter.

Thus, unlike in the i.i.d. case,

$$
\mathbb E[g_t(\theta_t)\mid \theta_t]
\neq
\bar g(\theta_t)
$$

in general.

This is the main Markovian-noise difficulty.

The stochastic update is not merely noisy. It is biased.

---

## 14. Projected TD under Markovian sampling

To handle Markovian data, the paper studies projected TD. The update is

$$
\theta_{t+1}
=
\Pi_{\mathcal B_R}
\left(
\theta_t+\alpha_t g_t(\theta_t)
\right),
$$

where

$$
\mathcal B_R
=
\{\theta\in\mathbb R^d:\|\theta\|_2\le R\}
$$

and $R$ is chosen so that

$$
\|\theta^*\|_2\le R.
$$

The projection ensures the iterates remain bounded. This implies the TD update is uniformly bounded.

If rewards are bounded by $r_{\max}$ and features are normalized, then for $\theta\in \mathcal B_R$ the update norm is bounded by a constant of the form

$$
G
=
r_{\max}+2R.
$$

This uniform bound is crucial for controlling the bias introduced by Markovian dependence.

---

## 15. The mixing-time idea

The Markov chain is assumed to mix geometrically. Roughly, there exist constants $m>0$ and $\rho\in(0,1)$ such that

$$
d_{\mathrm{TV}}
\left(
\mathcal L(s_t\mid s_0=s),
\pi
\right)
\le
m\rho^t.
$$

Define the mixing time

$$
\tau_{\mathrm{mix}}(\varepsilon)
=
\min
\{t\ge 0:m\rho^t\le \varepsilon\}.
$$

Since $m\rho^t$ decays geometrically,

$$
\tau_{\mathrm{mix}}(\varepsilon)
\asymp
\frac{\log(1/\varepsilon)}{\log(1/\rho^{-1})}.
$$

The proof idea is very intuitive.

At time $t$, the sample $(s_t,r_t,s_{t+1})$ is correlated with $\theta_t$. But $\theta_t$ changes slowly if the step sizes are small. So compare $\theta_t$ to an older iterate

$$
\theta_{t-\tau}.
$$

If $\tau$ is around the mixing time, then $s_t$ is nearly independent of the distant past, and hence nearly independent of $\theta_{t-\tau}$. The remaining gap between $\theta_{t-\tau}$ and $\theta_t$ is controlled by the cumulative step sizes.

So the Markovian analysis decomposes the bias into two parts:

1. a mixing error,
2. an iterate-drift error.

This is the correct intuition behind many finite-time Markovian stochastic approximation arguments.

---

## 16. Bias under Markovian noise

Define the bias term

$$
\zeta_t(\theta_t)
=
g_t(\theta_t)-\bar g(\theta_t).
$$

Under i.i.d. sampling,

$$
\mathbb E[\zeta_t(\theta_t)\mid \theta_t]=0.
$$

Under Markovian sampling, this is no longer true.

The paper controls the expectation of this bias using mixing and bounded updates. A representative bound has the form

$$
\mathbb E[\zeta_t(\theta_t)]
\lesssim
G^2
\tau_{\mathrm{mix}}(\alpha_T)
\alpha_{t-\tau_{\mathrm{mix}}(\alpha_T)}.
$$

The exact constants are less important than the structure:

$$
\text{Markovian bias}
\approx
\text{update size}
\times
\text{mixing time}.
$$

When the algorithm changes slowly and the chain mixes quickly, the Markovian bias is small.

This is the core reason the final Markovian bounds resemble the i.i.d. bounds up to mixing-time factors.

---

## 17. The Markovian finite-time picture

The projected Markovian TD results mirror the i.i.d. results.

With a constant step size of order

$$
\alpha=\frac{1}{\sqrt T},
$$

one obtains an averaged prediction-error bound with rate roughly

$$
\widetilde O\left(
\frac{\tau_{\mathrm{mix}}}{\sqrt T}
\right),
$$

where the tilde hides logarithmic terms.

With smaller constant step sizes, one obtains geometric convergence up to a noise-and-bias neighborhood.

With a carefully chosen decaying step size, one obtains a faster roughly

$$
\widetilde O\left(\frac{1}{T}\right)
$$

type rate, again with mixing-time dependence.

The message is clean:

> Markovian sampling does not destroy the TD-SGD analogy, but it taxes the rate by the amount of dependence in the trajectory.

This is one of the reasons the paper is influential. It gives a proof template for finite-time stochastic approximation under Markovian noise.

---

## 18. Approximation error: what does TD converge to?

The TD fixed point $\Phi\theta^*$ is not necessarily the best approximation to $V^\mu$ in the feature class. The best approximation is

$$
\Pi_D V^\mu.
$$

TD instead solves

$$
\Phi\theta^*
=
\Pi_D \mathcal{T}(\Phi\theta^*).
$$

How far can $\Phi\theta^*$ be from $V^\mu$?

Because $\mathcal{T}$ is a $\gamma$-contraction and $\Pi_D$ is non-expansive in the $D$-norm, the projected Bellman operator

$$
\Pi_D \mathcal{T}
$$

is also a $\gamma$-contraction in the $D$-norm. This implies

$$
\|V^\mu-\Phi\theta^*\|_D
\le
\frac{1}{\sqrt{1-\gamma^2}}
\|V^\mu-\Pi_D V^\mu\|_D.
$$

This says that the TD fixed point is competitive with the best approximation in the feature class, up to a discount-dependent factor.

This is another important conceptual point.

TD has two errors:

$$
\text{total error}
=
\text{statistical error}
+
\text{approximation error}.
$$

The finite-time analysis controls the statistical error

$$
\|V_{\theta_T}-V_{\theta^*}\|_D.
$$

The projected Bellman theory controls the approximation error

$$
\|V_{\theta^*}-V^\mu\|_D.
$$

Together, they tell us how far the learned value function is from the true value function.

---

## 19. TD($\lambda$): eligibility traces

The paper also extends the analysis to TD($\lambda$).

TD($\lambda$) uses eligibility traces. Define

$$
z_t
=
\phi(s_t)+\gamma\lambda z_{t-1}.
$$

The update becomes

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t z_t
\left(
r_t+\gamma\phi(s_{t+1})^{\texttt{T}}\theta_t
-
\phi(s_t)^{\texttt{T}}\theta_t
\right).
$$

The trace $z_t$ accumulates past features with geometrically decaying weights. The parameter $\lambda\in[0,1]$ interpolates between TD(0) and Monte Carlo-style updates.

The relevant operator for TD($\lambda$) is

$$
\mathcal{T}^{(\lambda)}
=
(1-\lambda)
\sum_{k=0}^{\infty}
\lambda^k \mathcal{T}^{k+1}.
$$

The projected fixed point becomes

$$
\Phi\theta^*
=
\Pi_D \mathcal{T}^{(\lambda)}(\Phi\theta^*).
$$

The operator $\Pi_D \mathcal{T}^{(\lambda)}$ is a contraction with modulus

$$
\kappa
=
\frac{\gamma(1-\lambda)}{1-\gamma\lambda}.
$$

Notice the behavior:

- when $\lambda=0$, $\kappa=\gamma$;
- as $\lambda\to 1$, $\kappa\to 0$.

Thus, larger $\lambda$ improves the contraction modulus and can reduce approximation bias.

But there is a statistical trade-off.

The eligibility trace has size controlled by

$$
\frac{1}{1-\gamma\lambda}.
$$

As $\lambda\to 1$, this quantity grows. The update becomes more variable because it depends on a longer history.

So TD($\lambda$) has a bias-variance trade-off:

$$
\lambda \uparrow
\quad
\Rightarrow
\quad
\text{better Bellman approximation geometry}
\quad
\text{but larger statistical noise}.
$$

This is a beautiful example of why finite-time analysis matters. Asymptotic fixed-point analysis alone suggests taking $\lambda$ close to $1$. Finite-time analysis reveals the cost of doing so.

---

## 20. Why this paper is conceptually important

There are several reasons I like this paper.

### 20.1 It explains why TD behaves like SGD

The paper does not claim that TD is SGD. Instead, it proves the precise weaker statement that matters:

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
\ge
(1-\gamma)
\|V_{\theta^*}-V_\theta\|_D^2.
$$

This is the core replacement for convexity.

And

$$
\|\bar g(\theta)\|_2
\le
2\|V_{\theta^*}-V_\theta\|_D
$$

is the core replacement for smoothness.

Together they allow one to reuse the intuition of SGD without pretending that TD is literally minimizing a simple loss.

---

### 20.2 It separates geometry from probability

The proof has two layers.

The first layer is deterministic geometry:

$$
\text{TD mean update points in the right direction.}
$$

The second layer is stochastic control:

$$
\text{sample updates concentrate around the mean update.}
$$

This separation is extremely useful. Once the geometric part is understood, different sampling models can be handled by changing the probabilistic argument.

For i.i.d. data, the noise is a martingale difference.

For Markovian data, there is an additional mixing bias.

This modular structure is one of the best parts of the paper.

---

### 20.3 It makes the role of $\gamma$ explicit

The factor $1-\gamma$ appears throughout the analysis. This is not an artifact. It reflects the fact that long-horizon problems are intrinsically harder.

When $\gamma$ is close to $1$, information propagates slowly and Bellman contraction becomes weak. The TD update direction becomes less strongly aligned with the fixed point.

This explains why finite-time RL bounds often deteriorate polynomially in

$$
\frac{1}{1-\gamma}.
$$

---

### 20.4 It makes the role of representation explicit

The parameter convergence rates depend on

$$
\omega=\lambda_{\min}(\Phi^{\texttt{T}} D\Phi).
$$

This quantity measures whether the features are well-conditioned under the states the policy actually visits.

If $\omega$ is tiny, there are parameter directions that barely affect values on visited states. Then one cannot hope for fast parameter convergence.

This is a representation issue, not just an algorithmic issue.

---

### 20.5 It clarifies the cost of Markovian data

The Markovian analysis says that temporal dependence is not fatal, but it costs mixing time.

A single trajectory behaves like i.i.d. data only after sufficient forgetting. The proof formalizes this using mixing-time arguments and projection.

This idea appears again and again in modern finite-time RL theory.

---

## 21. What I took away from the paper

The most important lesson is that TD has a hidden geometry.

At the algorithmic level, TD is simple:

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t
\phi(s_t)
\left(
r_t+\gamma\phi(s_{t+1})^{\texttt{T}}\theta_t-\phi(s_t)^{\texttt{T}}\theta_t
\right).
$$

At the proof level, the important object is not the raw update but its expected direction:

$$
\bar g(\theta)
=
\Phi^{\texttt{T}} D(\mathcal{T}\Phi\theta-\Phi\theta).
$$

This direction is not a gradient. But because of the projected Bellman equation, stationarity, and contraction, it satisfies a strong alignment property:

$$
(\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
\ge
(1-\gamma)
\|V_{\theta^*}-V_\theta\|_D^2.
$$

That one inequality is the bridge between TD learning and first-order optimization.

Once the bridge is built, the rest of the paper becomes a finite-time stochastic approximation argument.

---

## 22. Summary

The paper gives a clean finite-time analysis of TD learning with linear function approximation.

The main ideas are:

1. TD converges to the projected Bellman fixed point

   $$
   \Phi\theta^*
   =
   \Pi_D \mathcal{T}(\Phi\theta^*).
   $$

2. The expected TD update is

   $$
   \bar g(\theta)
   =
   \Phi^{\texttt{T}} D(\mathcal{T}\Phi\theta-\Phi\theta).
   $$

3. Although TD is not gradient descent, its expected update satisfies the gradient-like inequality

   $$
   (\theta^*-\theta)^{\texttt{T}} \bar g(\theta)
   \ge
   (1-\gamma)
   \|V_{\theta^*}-V_\theta\|_D^2.
   $$

4. The expected update is also controlled by prediction error:

   $$
   \|\bar g(\theta)\|_2
   \le
   2\|V_{\theta^*}-V_\theta\|_D.
   $$

5. These two facts yield SGD-style finite-time bounds.

6. Under i.i.d. sampling, TD behaves like a stochastic approximation algorithm with unbiased noise.

7. Under Markovian sampling, the noise is biased, but the bias can be controlled by mixing time and projection.

8. TD($\lambda$) fits into the same framework, with a contraction factor

   $$
   \kappa
   =
   \frac{\gamma(1-\lambda)}{1-\gamma\lambda}.
   $$

9. Larger $\lambda$ improves fixed-point approximation geometry but increases statistical noise through longer eligibility traces.

For me, the main conceptual takeaway is:

> TD is not SGD, but it has the right projected Bellman geometry to be analyzed as if it were almost SGD.

That is why this paper is such a clean starting point for understanding finite-time reinforcement learning theory.
