---
title: "Stochastic Gradient Descent: Why Randomness Works"
date: 2026-05-13
categories: [rl-blogs]
tags: [optimization, gradient-descent, stochastic-gradient-descent, machine-learning]
math: true
---

Optimization is one of the quiet engines behind modern machine learning. Whenever we train a neural network, fit a regression model, learn a value function, or tune a policy, we are usually doing some form of optimization.

At the heart of this story lies a deceptively simple idea:

> move in the direction that makes the loss smaller.

This is the philosophy behind **gradient descent**. But in modern machine learning, we often do not use plain gradient descent. Instead, we use its noisy, randomized, and surprisingly powerful cousin:

$$
\textbf{Stochastic Gradient Descent, or SGD.}
$$

The goal of this post is to explain, mathematically and intuitively, why SGD works, how it differs from vanilla gradient descent, and why injecting randomness into optimization is not a bug, but often a feature.

---

## 1. The Optimization Problem

Suppose we are given a loss function

$$
F(w),
$$

where \(w \in \mathbb{R}^d\) denotes the parameter vector of a model. In supervised learning, for example, \(w\) could be the weights of a neural network, and \(F(w)\) measures how badly the model performs on the data.

A standard finite-sum learning objective has the form

$$
F(w)
=
\frac{1}{n}
\sum_{i=1}^n f_i(w),
$$

where \(f_i(w)\) is the loss on the \(i\)-th data point.

For example, if we have training samples \((x_i,y_i)\), then one may define

$$
f_i(w)
=
\ell(h_w(x_i),y_i),
$$

where \(h_w\) is the model and \(\ell\) is a loss function.

The optimization goal is

$$
w^\star
\in
\arg\min_{w \in \mathbb{R}^d} F(w).
$$

So the question is simple:

> How do we find a parameter vector \(w\) that makes \(F(w)\) small?

---

## 2. Vanilla Gradient Descent

Gradient descent uses the local geometry of \(F\). If \(F\) is differentiable, then its gradient

$$
\nabla F(w)
$$

points in the direction of steepest local increase. Therefore, the negative gradient

$$
-\nabla F(w)
$$

points in the direction of steepest local decrease.

The vanilla gradient descent update is

$$
w_{t+1}
=
w_t
-
\eta \nabla F(w_t),
$$

where \(\eta>0\) is the step size, also called the learning rate.

For the finite-sum objective,

$$
F(w)
=
\frac{1}{n}
\sum_{i=1}^n f_i(w),
$$

the full gradient is

$$
\nabla F(w)
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w).
$$

Thus each gradient descent step requires computing gradients over the entire dataset.

This is perfectly reasonable when \(n\) is small. But when \(n\) is very large, computing

$$
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w)
$$

at every iteration becomes expensive.

This is where stochastic gradient descent enters the story.

---

## 3. The Main Idea of SGD

Instead of computing the full gradient over all \(n\) samples, SGD randomly picks one data point \(i_t\) and uses

$$
\nabla f_{i_t}(w_t)
$$

as a cheap estimate of the full gradient.

The SGD update is

$$
w_{t+1}
=
w_t
-
\eta_t \nabla f_{i_t}(w_t),
$$

where \(i_t\) is sampled randomly, often uniformly from \(\{1,\dots,n\}\).

At first glance, this looks reckless. Instead of using the true gradient \(\nabla F(w_t)\), we use only one randomly chosen component gradient. But the key observation is that this stochastic gradient is unbiased:

$$
\mathbb{E}_{i_t}
\left[
\nabla f_{i_t}(w_t)
\mid w_t
\right]
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w_t)
=
\nabla F(w_t).
$$

So SGD does not follow the exact gradient. Rather, it follows a noisy estimate whose average direction is correct.

In other words,

$$
\nabla f_{i_t}(w_t)
=
\nabla F(w_t)
+
\xi_t,
$$

where

$$
\mathbb{E}[\xi_t \mid w_t] = 0.
$$

Therefore, SGD can be written as

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t.
$$

This makes the nature of SGD clear:

$$
\text{SGD}
=
\text{Gradient Descent}
+
\text{Random Noise}.
$$

---

## 4. Why Does This Randomness Not Destroy Learning?

The most important reason SGD works is that the noise is centered. The stochastic gradient may be wrong at any individual step, but it is correct on average.

Imagine walking down a hill in fog. Vanilla gradient descent has a perfect compass that always points downhill. SGD has a noisy compass: sometimes it points slightly left, sometimes slightly right, sometimes too steeply, sometimes not steeply enough. But on average, it points downhill.

So even though the SGD trajectory is more jagged, it still tends to move toward regions of lower loss.

Mathematically, suppose \(F\) is smooth, meaning there exists \(L>0\) such that

$$
F(y)
\le
F(x)
+
\langle \nabla F(x), y-x\rangle
+
\frac{L}{2}\|y-x\|^2
$$

for all \(x,y\).

Using the SGD update

$$
w_{t+1}
=
w_t
-
\eta_t g_t,
$$

where

$$
g_t = \nabla f_{i_t}(w_t),
$$

smoothness gives

$$
F(w_{t+1})
\le
F(w_t)
-
\eta_t \langle \nabla F(w_t), g_t\rangle
+
\frac{L\eta_t^2}{2}\|g_t\|^2.
$$

Now take conditional expectation given \(w_t\). Since

$$
\mathbb{E}[g_t \mid w_t]
=
\nabla F(w_t),
$$

we get

$$
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
\le
F(w_t)
-
\eta_t \|\nabla F(w_t)\|^2
+
\frac{L\eta_t^2}{2}
\mathbb{E}
[
\|g_t\|^2
\mid w_t
].
$$

This inequality captures the entire story.

The term

$$
-\eta_t \|\nabla F(w_t)\|^2
$$

is the descent term. It says SGD wants to reduce the loss.

The term

$$
\frac{L\eta_t^2}{2}
\mathbb{E}
[
\|g_t\|^2
\mid w_t
]
$$

is the price of stochasticity and curvature. It appears because the stochastic step is noisy and the function may be curved.

The first term scales like \(\eta_t\), while the second term scales like \(\eta_t^2\). Therefore, when \(\eta_t\) is small enough, the descent term dominates.

This is the mathematical reason SGD can make progress.

---

## 5. Gradient Descent vs SGD

The difference between gradient descent and SGD is not just computational. They behave differently.

Vanilla gradient descent uses

$$
\nabla F(w_t)
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w_t),
$$

and updates as

$$
w_{t+1}
=
w_t
-
\eta \nabla F(w_t).
$$

SGD uses one randomly sampled gradient

$$
\nabla f_{i_t}(w_t),
$$

and updates as

$$
w_{t+1}
=
w_t
-
\eta_t \nabla f_{i_t}(w_t).
$$

So the main distinction is:

$$
\text{Gradient Descent uses the exact full gradient.}
$$

$$
\text{SGD uses a random unbiased estimate of the gradient.}
$$

The full gradient is accurate but expensive. The stochastic gradient is cheap but noisy.

This gives a fundamental tradeoff:

$$
\text{Gradient Descent: fewer but expensive iterations.}
$$

$$
\text{SGD: many cheap but noisy iterations.}
$$

For large-scale machine learning, the second option is often better.

---

## 6. A Simple Example

Consider the quadratic function

$$
F(w)
=
\frac{1}{2}w^2.
$$

Then

$$
\nabla F(w)=w.
$$

Gradient descent becomes

$$
w_{t+1}
=
w_t
-
\eta w_t
=
(1-\eta)w_t.
$$

If \(0<\eta<2\), this converges to \(0\), which is the minimizer.

Now imagine that instead of observing the exact gradient \(w_t\), we observe a noisy version

$$
g_t
=
w_t + \xi_t,
$$

where

$$
\mathbb{E}[\xi_t \mid w_t]=0.
$$

Then SGD becomes

$$
w_{t+1}
=
w_t
-
\eta_t(w_t+\xi_t)
=
(1-\eta_t)w_t
-
\eta_t \xi_t.
$$

The first term pulls \(w_t\) toward zero. The second term injects noise.

If \(\eta_t\) is fixed, the algorithm may keep fluctuating around zero. But if \(\eta_t\) decreases over time, the noise term becomes smaller and smaller. This is why classical SGD often uses decreasing step sizes such as

$$
\eta_t
=
\frac{c}{t+1}.
$$

Early in training, the algorithm explores aggressively. Later in training, the steps become smaller, allowing the iterates to stabilize.

---

## 7. Mini-Batch SGD

In practice, we often use a compromise between full gradient descent and one-sample SGD. This is called mini-batch SGD.

Instead of choosing one sample, we choose a random batch \(B_t\subseteq \{1,\dots,n\}\) and compute

$$
g_t
=
\frac{1}{|B_t|}
\sum_{i\in B_t}
\nabla f_i(w_t).
$$

The update is

$$
w_{t+1}
=
w_t
-
\eta_t g_t.
$$

Again,

$$
\mathbb{E}[g_t \mid w_t]
=
\nabla F(w_t).
$$

But the variance decreases as the batch size increases. Roughly,

$$
\mathrm{Var}(g_t)
\approx
\frac{1}{|B_t|}
\mathrm{Var}(\nabla f_i(w_t)).
$$

So larger batches give more accurate gradients, but each iteration becomes more expensive.

This creates another tradeoff:

$$
\text{small batch}
=
\text{cheap but noisy};
$$

$$
\text{large batch}
=
\text{expensive but stable}.
$$

In deep learning, mini-batch SGD is the standard because it balances computational efficiency, memory constraints, and statistical stability.

---

## 8. Why SGD Can Be Better Than Gradient Descent

It may seem that full gradient descent should always be better because it uses the exact gradient. But this is not always true in practice.

There are several reasons SGD can be preferable.

### 8.1 One Full Gradient May Be Too Expensive

If \(n\) is huge, one full gradient step costs \(n\) gradient computations.

SGD can make \(n\) small updates for roughly the same cost as one full gradient update.

So even if each SGD step is noisy, the algorithm may make much faster practical progress in terms of computation time.

### 8.2 Noise Can Help Escape Sharp Regions

In nonconvex optimization, such as neural network training, the loss landscape may contain saddle points, flat regions, and sharp minima.

The noise in SGD can help the algorithm avoid getting stuck in certain unstable regions. Informally, the randomness keeps the algorithm moving.

This does not mean noise is always good. Too much noise can prevent convergence. But moderate noise can act as an implicit exploratory force.

### 8.3 SGD Often Has an Implicit Regularization Effect

In modern machine learning, we do not only care about minimizing training loss. We care about generalization: performance on unseen data.

Empirically, SGD often finds solutions that generalize well. One intuitive explanation is that SGD noise biases the trajectory toward wider, flatter regions of the loss landscape, although the full theoretical picture is subtle.

A flat minimum is one where small perturbations of \(w\) do not change the loss too much. Such solutions may be more stable and generalize better.

---

## 9. A Basic Convergence Intuition

Let us write the stochastic gradient as

$$
g_t
=
\nabla F(w_t)+\xi_t,
$$

where

$$
\mathbb{E}[\xi_t\mid w_t]=0.
$$

The SGD update is

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t.
$$

Suppose the noise has bounded second moment:

$$
\mathbb{E}
[
\|g_t\|^2
\mid w_t
]
\le
G^2.
$$

Then from the smoothness inequality,

$$
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
\le
F(w_t)
-
\eta_t \|\nabla F(w_t)\|^2
+
\frac{L\eta_t^2G^2}{2}.
$$

After rearranging,

$$
\eta_t \|\nabla F(w_t)\|^2
\le
F(w_t)
-
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
+
\frac{L\eta_t^2G^2}{2}.
$$

Summing from \(t=0\) to \(T-1\), taking expectations, and telescoping gives

$$
\sum_{t=0}^{T-1}
\eta_t
\mathbb{E}
[
\|\nabla F(w_t)\|^2
]
\le
F(w_0)-F^\star
+
\frac{LG^2}{2}
\sum_{t=0}^{T-1}
\eta_t^2.
$$

This inequality says that the average gradient norm becomes small if the step sizes are chosen properly.

For example, if we choose a constant step size \(\eta\), then

$$
\frac{1}{T}
\sum_{t=0}^{T-1}
\mathbb{E}
[
\|\nabla F(w_t)\|^2
]
\lesssim
\frac{F(w_0)-F^\star}{\eta T}
+
L\eta G^2.
$$

The first term decreases with \(T\). The second term is the noise floor caused by stochastic gradients.

If \(\eta\) is small, the noise floor is small. But if \(\eta\) is too small, progress is slow. This is the central step-size tradeoff in SGD.

---

## 10. The Role of the Learning Rate

The learning rate \(\eta_t\) controls the balance between progress and noise.

If \(\eta_t\) is too large, SGD becomes unstable:

$$
w_{t+1}
=
w_t
-
\eta_t g_t
$$

may jump wildly across the loss landscape.

If \(\eta_t\) is too small, SGD becomes slow and may barely move.

A decreasing learning rate helps resolve this tension:

$$
\eta_t \downarrow 0.
$$

Early iterations use large steps to move quickly. Later iterations use small steps to reduce noise and stabilize near a minimizer.

Classical stochastic approximation often requires

$$
\sum_{t=0}^{\infty} \eta_t = \infty,
$$

and

$$
\sum_{t=0}^{\infty} \eta_t^2 < \infty.
$$

The first condition says the algorithm keeps moving enough to reach the solution. The second says the accumulated noise remains controlled.

A standard example is

$$
\eta_t
=
\frac{c}{t+1}.
$$

---

## 11. Convex Picture: Moving Toward the Optimum

When \(F\) is convex, we can make the intuition even cleaner.

Convexity means

$$
F(w)-F(w^\star)
\le
\langle \nabla F(w), w-w^\star\rangle.
$$

Now consider the squared distance to the minimizer:

$$
\|w_{t+1}-w^\star\|^2.
$$

Using the SGD update,

$$
w_{t+1}
=
w_t-\eta_t g_t,
$$

we get

$$
\|w_{t+1}-w^\star\|^2
=
\|w_t-w^\star\|^2
-
2\eta_t \langle g_t, w_t-w^\star\rangle
+
\eta_t^2\|g_t\|^2.
$$

Taking conditional expectation,

$$
\mathbb{E}
[
\|w_{t+1}-w^\star\|^2
\mid w_t
]
=
\|w_t-w^\star\|^2
-
2\eta_t
\langle \nabla F(w_t), w_t-w^\star\rangle
+
\eta_t^2
\mathbb{E}
[
\|g_t\|^2
\mid w_t
].
$$

Using convexity,

$$
\langle \nabla F(w_t), w_t-w^\star\rangle
\ge
F(w_t)-F(w^\star).
$$

Therefore,

$$
\mathbb{E}
[
\|w_{t+1}-w^\star\|^2
\mid w_t
]
\le
\|w_t-w^\star\|^2
-
2\eta_t
(
F(w_t)-F(w^\star)
)
+
\eta_t^2G^2.
$$

Again, the same structure appears:

$$
\text{progress term}
-
2\eta_t(F(w_t)-F^\star),
$$

plus

$$
\text{noise term}
+
\eta_t^2G^2.
$$

The algorithm moves toward the optimum in expectation, while stochasticity introduces a second-order penalty.

---

## 12. A Geometric Intuition

Gradient descent follows a smooth path down the loss surface.

SGD follows a noisy path.

But the noisy path is not arbitrary. It is biased toward descent.

One can think of SGD as a particle moving under two forces:

$$
\text{deterministic force} = -\nabla F(w_t),
$$

and

$$
\text{random force} = -\xi_t.
$$

So the update

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t
$$

resembles a discretized noisy dynamical system.

When the gradient is large, the deterministic descent dominates. When the gradient is small, the noise becomes more visible. This is why SGD often fluctuates near minimizers instead of stopping exactly.

The learning rate controls the temperature of this motion. A large learning rate creates a hotter, more exploratory process. A small learning rate creates a colder, more stable process.

---

## 13. Summary: What Makes SGD Work?

SGD works because it replaces the full gradient with a cheap random estimate:

$$
g_t
=
\nabla f_{i_t}(w_t).
$$

The crucial property is unbiasedness:

$$
\mathbb{E}[g_t\mid w_t]
=
\nabla F(w_t).
$$

Thus, SGD moves in the correct direction on average.

The price is variance:

$$
g_t
=
\nabla F(w_t)+\xi_t.
$$

This variance makes the trajectory noisy, but with appropriate learning rates, the noise can be controlled.

Vanilla gradient descent is accurate but expensive. SGD is noisy but cheap. In large-scale machine learning, cheap noisy progress often beats expensive exact progress.

The essence of SGD is therefore beautifully simple:

$$
\boxed{
\text{Do not compute the perfect direction. Compute a cheap direction that is correct on average.}
}
$$

That single idea powers much of modern machine learning.

---

## 14. Final Takeaway

Gradient descent asks:

> What is the exact direction of steepest descent?

SGD asks a more practical question:

> Can I cheaply estimate a useful descent direction?

The answer is yes.

And that is why stochastic gradient descent is so powerful. It turns optimization from a deterministic march into a noisy but efficient journey, where randomness is not merely tolerated, but exploited.
