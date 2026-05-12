---
title: "Why Gradient Descent Works: A Small Mathematical Story"
date: 2026-05-12 00:00:00 -0400
categories: [rl-blogs]
tags: [optimization, gradient-descent, machine-learning, theory]
series: Fundamentals of Reinforcement Learning
math: true
---

Gradient descent is one of the simplest and most influential algorithms in modern machine learning. At a high level, the idea is almost embarrassingly simple: if we want to minimize a function, we move in the direction in which the function decreases fastest.

Suppose we want to solve

$$
\min_{x \in \mathbb{R}^d} f(x),
$$

where $f:\mathbb{R}^d \to \mathbb{R}$ is differentiable. Gradient descent generates a sequence

$$
x_{t+1} = x_t - \eta \nabla f(x_t),
$$

where $\eta>0$ is called the step-size or learning rate.

The entire algorithm is contained in this single line. At first glance, this update looks almost too simple to matter. We begin at a point \(x_t\), compute the gradient \(\nabla f(x_t)\), and then take a small step in the opposite direction. The number \(\eta>0\) is called the **learning rate** or **step size**, and it controls how far we move at each iteration.

The intuition is wonderfully geometric. The gradient points in the direction where the function increases the fastest. So, if our goal is to make the function smaller, the most natural thing to do is to walk in the opposite direction. Gradient descent is simply the act of repeatedly asking:

> Where is downhill from here?

and then taking a small step that way.

This is what makes the method so elegant. Gradient descent does not require us to understand the entire landscape of the function. It only uses local information: the slope at the current point. Yet, by repeating this local rule again and again, the iterates can move toward a minimizer.

Of course, the size of the step matters. If \(\eta\) is too large, we may overshoot the minimum and oscillate. If \(\eta\) is too small, we may move in the right direction but painfully slowly. But under suitable assumptions, such as smoothness and convexity, this one-line update can be proved to converge.

That is the beauty of gradient descent: a global optimization problem is attacked through a purely local rule. The convergence proof makes this intuition precise, and reveals why this deceptively simple algorithm sits at the heart of modern machine learning.

---

## The Basic Geometric Idea

The gradient $\nabla f(x)$ points in the direction of steepest local increase of $f$. Therefore, the direction

$$
-\nabla f(x)
$$

is the direction of steepest local decrease. If we take a small step in this direction, we should expect the objective value to go down.

To make this intuition precise, we need a smoothness assumption.

---

## Smoothness and the Descent Lemma

A differentiable function $f$ is called $L$-smooth if its gradient is Lipschitz:

$$
\|\nabla f(x)-\nabla f(y)\| \le L\|x-y\|,
\qquad \forall x,y \in \mathbb{R}^d.
$$

Equivalently, an $L$-smooth function satisfies the upper quadratic bound

$$
f(y)
\le
f(x)
+
\langle \nabla f(x), y-x\rangle
+
\frac{L}{2}\|y-x\|^2.
$$

This inequality says that, near $x$, the function can be upper bounded by a quadratic model.

Now set

$$
y = x - \eta \nabla f(x).
$$

Plugging this into the smoothness inequality gives

$$
f(x-\eta \nabla f(x))
\le
f(x)
-
\eta \|\nabla f(x)\|^2
+
\frac{L\eta^2}{2}\|\nabla f(x)\|^2.
$$

Therefore,

$$
f(x_{t+1})
\le
f(x_t)
-
\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla f(x_t)\|^2.
$$

If we choose

$$
0 < \eta \le \frac{1}{L},
$$

then

$$
1-\frac{L\eta}{2} \ge \frac12,
$$

and hence

$$
f(x_{t+1})
\le
f(x_t)
-
\frac{\eta}{2}\|\nabla f(x_t)\|^2.
$$

This is the key inequality behind gradient descent.

It says: unless the gradient is small, the objective must decrease.

---

## What This Gives in the Nonconvex Case

Assume $f$ is bounded below by $f^\star$. Summing the descent inequality from $t=0$ to $T-1$, we get

$$
\sum_{t=0}^{T-1}
\frac{\eta}{2}\|\nabla f(x_t)\|^2
\le
f(x_0)-f(x_T)
\le
f(x_0)-f^\star.
$$

Therefore,

$$
\sum_{t=0}^{T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta}.
$$

Dividing by $T$,

$$
\frac1T
\sum_{t=0}^{T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta T}.
$$

Thus,

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta T}.
$$

So gradient descent finds an approximate stationary point at rate

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|^2
=
O\left(\frac1T\right).
$$

Equivalently,

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|
=
O\left(\frac1{\sqrt{T}}\right).
$$

This is the standard nonconvex gradient descent guarantee.

---

## Convexity Gives More Structure

Now suppose $f$ is convex. Convexity means

$$
f(y) \ge f(x) + \langle \nabla f(x), y-x\rangle,
\qquad \forall x,y.
$$

Let $x^\star$ be a minimizer. Setting $y=x^\star$, we get

$$
f(x_t)-f(x^\star)
\le
\langle \nabla f(x_t), x_t-x^\star\rangle.
$$

Now look at the squared distance to the optimum:

$$
\|x_{t+1}-x^\star\|^2
=
\|x_t-\eta \nabla f(x_t)-x^\star\|^2.
$$

Expanding,

$$
\|x_{t+1}-x^\star\|^2
=
\|x_t-x^\star\|^2
-
2\eta \langle \nabla f(x_t),x_t-x^\star\rangle
+
\eta^2\|\nabla f(x_t)\|^2.
$$

Using convexity,

$$
\langle \nabla f(x_t),x_t-x^\star\rangle
\ge
f(x_t)-f(x^\star).
$$

Therefore,

$$
2\eta(f(x_t)-f(x^\star))
\le
\|x_t-x^\star\|^2
-
\|x_{t+1}-x^\star\|^2
+
\eta^2\|\nabla f(x_t)\|^2.
$$

For smooth convex functions, a sharper analysis with $\eta=1/L$ gives

$$
f(x_T)-f(x^\star)
\le
\frac{L\|x_0-x^\star\|^2}{2T}.
$$

Thus, in the convex smooth setting, gradient descent converges in function value at rate

$$
f(x_T)-f(x^\star)
=
O\left(\frac1T\right).
$$

---

## Strong Convexity Gives Linear Convergence

The most elegant result appears when $f$ is both smooth and strongly convex.

A function is $\mu$-strongly convex if

$$
f(y)
\ge
f(x)
+
\langle \nabla f(x),y-x\rangle
+
\frac{\mu}{2}\|y-x\|^2.
$$

Strong convexity says that the function has curvature everywhere. It rules out flat directions and ensures a unique minimizer.

For an $L$-smooth and $\mu$-strongly convex function, gradient descent with step-size $\eta=1/L$ satisfies

$$
\|x_{t+1}-x^\star\|
\le
\left(1-\frac{\mu}{L}\right)
\|x_t-x^\star\|.
$$

Iterating,

$$
\|x_t-x^\star\|
\le
\left(1-\frac{\mu}{L}\right)^t
\|x_0-x^\star\|.
$$

Thus, gradient descent converges geometrically:

$$
\|x_t-x^\star\|
=
O\left(
\left(1-\frac{\mu}{L}\right)^t
\right).
$$

The ratio

$$
\kappa = \frac{L}{\mu}
$$

is called the condition number. A smaller condition number means faster convergence; a larger condition number means the objective is poorly conditioned and gradient descent may move slowly.

---

## The Quadratic Case

A particularly transparent example is

$$
f(x)=\frac12 x^\top A x - b^\top x,
$$

where $A$ is symmetric positive definite. Then

$$
\nabla f(x)=Ax-b.
$$

The minimizer satisfies

$$
Ax^\star=b.
$$

Gradient descent becomes

$$
x_{t+1}
=
x_t-\eta(Ax_t-b).
$$

Subtracting $x^\star$, and using $Ax^\star=b$,

$$
x_{t+1}-x^\star
=
(I-\eta A)(x_t-x^\star).
$$

Therefore,

$$
x_t-x^\star
=
(I-\eta A)^t(x_0-x^\star).
$$

The convergence is controlled by the eigenvalues of $I-\eta A$. If the eigenvalues of $A$ lie in

$$
\mu \le \lambda_i(A) \le L,
$$

then choosing $0<\eta<2/L$ ensures convergence. The best fixed step-size is

$$
\eta^\star = \frac{2}{L+\mu},
$$

which gives contraction factor

$$
\frac{L-\mu}{L+\mu}
=
\frac{\kappa-1}{\kappa+1}.
$$

This explains why ill-conditioned problems are hard: when $\kappa$ is large,

$$
\frac{\kappa-1}{\kappa+1}
\approx 1,
$$

so the error shrinks very slowly.

---

## Takeaway

Gradient descent works because smoothness turns the first-order Taylor approximation into a reliable upper bound. Moving against the gradient decreases this upper bound, and therefore decreases the objective.

The basic story is:

$$
x_{t+1}=x_t-\eta\nabla f(x_t)
$$

combined with

$$
f(x_{t+1})
\le
f(x_t)
-
\frac{\eta}{2}\|\nabla f(x_t)\|^2.
$$

From this one inequality, we get:

- for nonconvex smooth functions, convergence to stationary points;
- for convex smooth functions, $O(1/T)$ convergence in function value;
- for strongly convex smooth functions, linear convergence;
- for quadratic functions, convergence determined exactly by eigenvalues.

This is why gradient descent remains one of the central algorithms in optimization and machine learning: it is simple enough to write in one line, yet rich enough to reveal the geometry of learning.
