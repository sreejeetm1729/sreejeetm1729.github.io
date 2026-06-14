---
title: "Why TD Learning Likes the 2-Norm: Mean Directions, Inner Products, and Bellman Geometry"
date: 2026-06-13 
categories: [rl-blogs]
rl_section: rl-fundamentals 
tags: [reinforcement-learning, temporal-difference-learning, stochastic-approximation, bellman-operators, function-approximation, norms]
math: true
---

Temporal-difference learning is often introduced as an incremental algorithm that updates a value estimate using one sampled transition at a time. At first glance, it looks like stochastic gradient descent: we see a random sample, compute a noisy update, and move the parameter in a direction that is hopefully correct on average.

This analogy is useful, but it hides an important mathematical point. TD is not always literal SGD on an ordinary loss function. What is always true in the classical linear setting is that TD is a stochastic approximation algorithm: the random update has a population mean direction, and the analysis tries to show that this mean direction pulls the current iterate toward the TD fixed point.

This is exactly where the 2-norm becomes natural.

The 2-norm is not just a way of measuring length. It comes from an inner product. That inner product lets us compare the current error vector with the expected update direction. In other words, it lets us mathematically express the statement:

> the update is noisy, but on average it points back toward the solution.

The infinity norm is very different. It measures the largest coordinate-wise error. This is extremely powerful for tabular Bellman optimality and Q-learning because the optimal Bellman operator is naturally a contraction in infinity norm. But infinity norm does not have a compatible inner product geometry. So it is less convenient when the analysis is based on mean directions, spectral stability, projections, and stochastic approximation.

This post explains the precise distinction.

---

## 1. Two norms, two questions

Suppose we have an error vector $e \in \mathbb{R}^d.$

In tabular value prediction, we may have

$$
d = |\mathcal{S}|,
$$

and

$$
e(s) = V(s) - V^\star(s).
$$

In tabular control, we may have

$$
d = |\mathcal{S}||\mathcal{A}|,
$$

and

$$
e(s,a) = Q(s,a)-Q^\star(s,a).
$$

The infinity norm is

$$
\|e\|_\infty
=
\max_i |e_i|.
$$

It asks:

> What is the worst error anywhere?

The 2-norm is

$$
\|e\|_2
=
\sqrt{\sum_{i=1}^d e_i^2}.
$$

A weighted 2-norm, very common in RL, is

$$
\|e\|_\nu
=
\sqrt{\sum_i \nu_i e_i^2},
$$

where

$$
\nu_i \ge 0,
\qquad
\sum_i \nu_i = 1.
$$

It asks:

> What is the average squared error under the distribution $\nu$?

So the two norms have different meanings.

The infinity norm gives a uniform guarantee:

$$
\|e\|_\infty \le \varepsilon
\quad
\Longrightarrow
\quad
|e_i| \le \varepsilon
\quad
\text{for every } i.
$$

The weighted 2-norm gives a distributional guarantee:

$$
\|e\|_\nu^2
=
\mathbb{E}_{i\sim \nu}[e_i^2].
$$

This allows large errors on rarely visited states, as long as their probability under $\nu$ is small.

The basic comparison is

$$
\|e\|_\nu \le \|e\|_\infty.
$$

So max-norm control implies weighted 2-norm control. But the reverse direction costs coverage. If

$$
\nu_{\min} = \min_i \nu_i > 0,
$$

then

$$
\|e\|_\infty
\le
\frac{1}{\sqrt{\nu_{\min}}}\|e\|_\nu.
$$

This factor can be terrible when some states are rarely sampled.

This is the first big lesson:

> A 2-norm guarantee is usually easier to obtain in TD and function approximation, but it is an average-distributional guarantee. An infinity-norm guarantee is stronger in a uniform sense, but often harder to prove outside the tabular Bellman contraction setting.

---

## 2. Why TD looks like SGD, but is not always SGD

Consider policy evaluation for a fixed policy $\pi$. The policy-specific Bellman operator is

$$
\mathcal{T}^\pi V
=
r^\pi + \gamma P^\pi V,
$$

where

$$
\gamma \in (0,1),
$$

$r^\pi$ is the expected one-step reward under policy $\pi$, and $P^\pi$ is the transition matrix induced by $\pi$.

In linear value approximation, we write

$$
V_\theta(s)
=
\phi(s)^{\texttt{T}}\theta,
$$

where

$$
\phi(s) \in \mathbb{R}^d,
\qquad
\theta \in \mathbb{R}^d.
$$

Given a transition

$$
(s_t,r_t,s_{t+1}),
$$

the TD error is

$$
\delta_t(\theta_t)
=
r_t + \gamma \phi(s_{t+1})^{\texttt{T}}\theta_t
-
\phi(s_t)^{\texttt{T}}\theta_t.
$$

The TD(0) update is

$$
\theta_{t+1}
=
\theta_t
+
\alpha_t \delta_t(\theta_t)\phi(s_t).
$$

This resembles SGD because the update is random and sample-based. However, for ordinary SGD on a loss $L(\theta)$, the mean update direction is

$$
-\nabla L(\theta).
$$

For TD, the population mean direction has the form

$$
\mathbb{E}[\delta_t(\theta)\phi(s_t)]
=
b-A\theta,
$$

where, under the stationary distribution $\nu$,

$$
A
=
\mathbb{E}_\nu
\left[
\phi(s_t)
\bigl(\phi(s_t)-\gamma \phi(s_{t+1})\bigr)^{\texttt{T}}
\right],
$$

and

$$
b
=
\mathbb{E}_\nu
\left[
\phi(s_t)r_t
\right].
$$

The TD fixed point $\theta^\star$ satisfies

$$
A\theta^\star=b.
$$

Therefore

$$
b-A\theta
=
-A(\theta-\theta^\star).
$$

So the TD recursion can be written morally as

$$
\theta_{t+1}-\theta^\star
=
\theta_t-\theta^\star
-
\alpha_t A(\theta_t-\theta^\star)
+
\alpha_t \xi_{t+1},
$$

where $\xi_{t+1}$ is the stochastic noise around the mean direction.

Let

$$
e_t = \theta_t-\theta^\star.
$$

Then

$$
e_{t+1}
=
e_t-
\alpha_t A e_t
+
\alpha_t \xi_{t+1}.
$$

This is the basic stochastic approximation form behind linear TD.

The key question is:

> Does the mean direction $-Ae_t$ actually move the iterate toward $\theta^\star$?

The 2-norm gives the cleanest way to answer this.

---

## 3. The exact role of the inner product

The squared 2-norm satisfies the identity

$$
\|x+y\|_2^2
=
\|x\|_2^2
+2x^{\texttt{T}}y
+\|y\|_2^2.
$$

This identity is the engine of TD finite-time analysis.

Apply it to

$$
e_{t+1}
=
e_t-
\alpha_t A e_t
+
\alpha_t \xi_{t+1}.
$$

Ignoring noise for one moment,

$$
\|e_t-\alpha_t A e_t\|_2^2
=
\|e_t\|_2^2
-2\alpha_t e_t^{\texttt{T}}Ae_t
+
\alpha_t^2\|Ae_t\|_2^2.
$$

The important term is

$$
e_t^{\texttt{T}}Ae_t.
$$

If we can prove that

$$
e^{\texttt{T}}Ae
\ge
\mu \|e\|_2^2
\qquad
\text{for all } e,
$$

for some $\mu>0$, then the mean TD direction has negative drift:

$$
\langle e,-Ae\rangle
=
-e^{\texttt{T}}Ae
\le
-\mu\|e\|_2^2.
$$

This is the formal version of the intuition:

> the expected TD update direction points back toward the fixed point.

The angle between the error vector $e$ and the mean update direction $-Ae$ is favorable because their inner product is negative.

This is exactly why the 2-norm is natural. The 2-norm is generated by the inner product

$$
\langle x,y\rangle = x^{\texttt{T}}y.
$$

So it allows us to compare direction and error by writing

$$
\langle \text{error},\text{mean update}\rangle.
$$

The infinity norm does not give such an identity. There is no comparable expansion of the form

$$
\|e-\alpha Ae\|_\infty^2
=
\|e\|_\infty^2
-2\alpha \langle e,Ae\rangle
+\cdots.
$$

The infinity norm measures the largest coordinate, but it does not tell us whether the mean update direction is globally aligned against the error.

This is the second big lesson:

> TD analysis wants to show that the noisy update is correct on average. The 2-norm turns this into an inner-product drift inequality.

---

## 4. The stochastic approximation proof skeleton

Let us include the noise term. Write the TD recursion as

$$
e_{t+1}
=
e_t-
\alpha_t A e_t
+
\alpha_t \xi_{t+1},
$$

where

$$
\mathbb{E}[\xi_{t+1}\mid \mathcal{F}_t]=0
$$

in the ideal i.i.d. or conditionally unbiased setting.

Now expand:

$$
\|e_{t+1}\|_2^2
=
\|e_t-\alpha_t A e_t+\alpha_t\xi_{t+1}\|_2^2.
$$

Using the squared-norm identity,

$$
\begin{aligned}
\|e_{t+1}\|_2^2
=&
\|e_t-\alpha_t A e_t\|_2^2
+2\alpha_t\langle e_t-\alpha_t A e_t,\xi_{t+1}\rangle
+\alpha_t^2\|\xi_{t+1}\|_2^2.
\end{aligned}
$$

Taking conditional expectation and using the mean-zero property,

$$
\mathbb{E}
\left[
\langle e_t-\alpha_t A e_t,\xi_{t+1}\rangle
\mid \mathcal{F}_t
\right]
=0.
$$

Therefore

$$
\mathbb{E}
\left[
\|e_{t+1}\|_2^2
\mid \mathcal{F}_t
\right]
=
\|e_t-\alpha_t A e_t\|_2^2
+
\alpha_t^2
\mathbb{E}
\left[
\|\xi_{t+1}\|_2^2
\mid \mathcal{F}_t
\right].
$$

Expanding the deterministic part gives

$$
\begin{aligned}
\|e_t-\alpha_t A e_t\|_2^2
=&
\|e_t\|_2^2
-2\alpha_t e_t^{\texttt{T}}Ae_t
+
\alpha_t^2\|Ae_t\|_2^2.
\end{aligned}
$$

Assume

$$
e^{\texttt{T}}Ae
\ge
\mu\|e\|_2^2,
$$

and

$$
\|Ae\|_2
\le
L\|e\|_2.
$$

Then

$$
\|e_t-\alpha_t A e_t\|_2^2
\le
\left(1-2\alpha_t\mu+
\alpha_t^2L^2\right)
\|e_t\|_2^2.
$$

For sufficiently small $\alpha_t$, this becomes

$$
\|e_t-\alpha_t A e_t\|_2^2
\le
(1-c\alpha_t)\|e_t\|_2^2
$$

for some constant $c>0$. Therefore

$$
\mathbb{E}
\left[
\|e_{t+1}\|_2^2
\mid \mathcal{F}_t
\right]
\le
(1-c\alpha_t)\|e_t\|_2^2
+
\alpha_t^2
\mathbb{E}
\left[
\|\xi_{t+1}\|_2^2
\mid \mathcal{F}_t
\right].
$$

This is the canonical drift-plus-noise recursion.

It says:

- the first term contracts the previous error;
- the second term injects stochastic noise;
- the stepsize $\alpha_t$ balances contraction and noise.

This entire proof is powered by the 2-norm expansion.

---

## 5. Why this is not just a cosmetic choice

One might think the choice of norm is merely aesthetic. It is not.

The squared 2-norm is differentiable and quadratic. It interacts perfectly with linear recursions. If the mean dynamics are

$$
e_{t+1}
=
(I-\alpha A)e_t,
$$

then the 2-norm analysis asks whether

$$
\|(I-\alpha A)e\|_2^2
<
\|e\|_2^2.
$$

This reduces to studying the symmetric part of $A$:

$$
\frac{A+A^{\texttt{T}}}{2}.
$$

Indeed,

$$
e^{\texttt{T}}Ae
=
e^{\texttt{T}}
\left(\frac{A+A^{\texttt{T}}}{2}\right)e,
$$

because the skew-symmetric part contributes zero to the quadratic form.

Thus a sufficient stability condition is

$$
\frac{A+A^{\texttt{T}}}{2}
\succeq
\mu I.
$$

This is a spectral condition. It is exactly the kind of condition that appears naturally in stochastic approximation, linear TD, and least-squares methods.

The infinity norm does not see this geometry directly. It is instead controlled by row sums and coordinate-wise worst-case behavior:

$$
\|Mx\|_\infty
\le
\|M\|_\infty\|x\|_\infty,
$$

where

$$
\|M\|_\infty
=
\max_i \sum_j |M_{ij}|.
$$

This can be useful, but it is often too crude for TD with function approximation because the dynamics are not naturally coordinate-wise contractive.

---

## 6. The projected Bellman equation is a 2-norm object

The 2-norm is also natural because linear TD does not usually solve the exact Bellman equation in the full value-function space.

The true value function $V^\pi$ satisfies

$$
V^\pi
=
\mathcal{T}^\pi V^\pi.
$$

But if we restrict ourselves to the linear class

$$
\mathcal{F}
=
\{\Phi\theta: \theta\in\mathbb{R}^d\},
$$

then $V^\pi$ may not lie in $\mathcal{F}$. We therefore cannot usually solve

$$
\Phi\theta
=
\mathcal{T}^\pi(\Phi\theta).
$$

Instead, linear TD solves the projected Bellman equation

$$
\Phi\theta^\star
=
\Pi_\nu \mathcal{T}^\pi(\Phi\theta^\star),
$$

where $\Pi_\nu$ is the orthogonal projection onto the feature space under the weighted 2-norm

$$
\|V\|_\nu^2
=
\sum_s \nu(s)V(s)^2.
$$

The projection is defined by

$$
\Pi_\nu V
=
\arg\min_{U\in \mathrm{span}(\Phi)}\|U-V\|_\nu.
$$

This definition is inherently 2-norm based. Orthogonality means

$$
\langle V-\Pi_\nu V, U\rangle_\nu=0
\qquad
\text{for all } U\in \mathrm{span}(\Phi),
$$

where

$$
\langle f,g\rangle_\nu
=
\sum_s \nu(s)f(s)g(s).
$$

This gives the normal equations behind TD.

If $D_\nu$ is the diagonal matrix with entries $\nu(s)$, then the projection is

$$
\Pi_\nu
=
\Phi
(\Phi^{\texttt{T}}D_\nu\Phi)^{-1}
\Phi^{\texttt{T}}D_\nu.
$$

Again, this is Hilbert-space geometry. It is not max-norm geometry.

This is another reason TD and the 2-norm are tied together:

> linear TD is not merely using the 2-norm to analyze the algorithm; the fixed point itself is defined through a 2-norm projection.

---

## 7. On-policy Bellman evaluation also has 2-norm stability

For a fixed policy $\pi$, suppose $\nu$ is stationary for $P^\pi$. Then

$$
\nu^{\texttt{T}}P^\pi
=
\nu^{\texttt{T}}.
$$

For any function $f$, Jensen's inequality gives

$$
\left((P^\pi f)(s)\right)^2
=
\left(\mathbb{E}[f(s')\mid s]\right)^2
\le
\mathbb{E}[f(s')^2\mid s].
$$

Taking expectation over $s\sim \nu$,

$$
\|P^\pi f\|_\nu^2
\le
\|f\|_\nu^2.
$$

Therefore

$$
\|P^\pi f\|_\nu
\le
\|f\|_\nu.
$$

As a result,

$$
\begin{aligned}
\|\mathcal{T}^\pi V-\mathcal{T}^\pi W\|_\nu
&=
\gamma\|P^\pi(V-W)\|_\nu\\
&\le
\gamma\|V-W\|_\nu.
\end{aligned}
$$

So for on-policy prediction, the policy Bellman operator is also a contraction in the stationary weighted 2-norm.

This is one reason prediction is geometrically friendlier than control. Policy evaluation has a fixed transition kernel $P^\pi$, a stationary distribution $\nu$, and a natural weighted Hilbert space.

---

## 8. Why Q-learning prefers infinity norm

The story changes for control.

The optimal Bellman operator is

$$
(\mathcal{T}Q)(s,a)
=
r(s,a)
+
\gamma
\mathbb{E}_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'} Q(s',a')
\right].
$$

For any two action-value functions $Q$ and $Q'$,

$$
\begin{aligned}
| (\mathcal{T}Q)(s,a)-(\mathcal{T}Q')(s,a) |
&\le
\gamma
\mathbb{E}_{s'}
\left[
\left|
\max_{a'}Q(s',a')-
\max_{a'}Q'(s',a')
\right|
\right]\\
&\le
\gamma
\mathbb{E}_{s'}
\left[
\max_{a'}|Q(s',a')-Q'(s',a')|
\right]\\
&\le
\gamma\|Q-Q'\|_\infty.
\end{aligned}
$$

Taking the maximum over $(s,a)$,

$$
\|\mathcal{T}Q-\mathcal{T}Q'\|_\infty
\le
\gamma\|Q-Q'\|_\infty.
$$

This is the fundamental reason tabular Q-learning analysis is usually max-norm based.

The Bellman optimality operator is globally contractive in infinity norm. This contraction is:

- distribution-free;
- dimension-free;
- uniform over all state-action pairs;
- compatible with greedy policy improvement.

If we prove

$$
\|Q_t-Q^\star\|_\infty \le \varepsilon,
$$

then every state-action estimate is accurate. This is the right type of guarantee when the final policy may visit states differently from the data-generating distribution.

By contrast, the optimal Bellman operator is not generally a contraction in an arbitrary 2-norm. The max over actions changes the geometry, and the next-state distribution may move mass into regions that the chosen 2-norm does not weight heavily.

This is why there is a clean slogan:

> TD prediction likes 2-norm geometry. Tabular optimal control likes infinity-norm contraction.

---

## 9. Why infinity norm is awkward for TD mean-direction analysis

For TD, the core proof step is something like

$$
\langle e_t, -Ae_t\rangle < 0.
$$

This says the population direction points against the error.

But the infinity norm does not naturally encode angles. The quantity

$$
\|e_t\|_\infty
$$

is determined by whichever coordinate currently has the largest absolute value. That coordinate may change from one step to the next. The max function is nonsmooth. A small update in many coordinates may reduce average error but temporarily increase the largest coordinate.

For example, suppose

$$
e=(10,9,9,\ldots,9).
$$

An update may reduce most coordinates substantially but leave the first coordinate almost unchanged. Then the 2-norm decreases a lot, but the infinity norm barely changes.

Conversely, an update may reduce the largest coordinate but increase many smaller coordinates. Then the infinity norm improves while the 2-norm may worsen.

The infinity norm is therefore excellent when the operator itself is known to be a max-norm contraction. But if the proof only says that the mean direction is globally stable in an inner-product sense, max-norm may not reflect that stability cleanly.

This is the precise issue with TD-style stochastic approximation.

---

## 10. The role of feature covariance

In linear TD, another important matrix is the feature covariance

$$
C
=
\mathbb{E}_\nu[\phi(s_t)\phi(s_t)^{\texttt{T}}].
$$

The prediction error in value space is

$$
\|V_\theta-V_{\theta^\star}\|_\nu^2
=
\mathbb{E}_\nu
\left[
\left(\phi(s)^{\texttt{T}}(\theta-\theta^\star)\right)^2
\right].
$$

Equivalently,

$$
\|V_\theta-V_{\theta^\star}\|_\nu^2
=
(\theta-\theta^\star)^{\texttt{T}}
C
(\theta-\theta^\star).
$$

Thus the value-function 2-norm corresponds to a parameter-space quadratic form.

If

$$
\lambda_{\min}(C)>0,
$$

then

$$
\lambda_{\min}(C)\|\theta-\theta^\star\|_2^2
\le
\|V_\theta-V_{\theta^\star}\|_\nu^2
\le
\lambda_{\max}(C)\|\theta-\theta^\star\|_2^2.
$$

This is another example of 2-norm geometry giving a clean bridge between parameter error and prediction error.

Infinity norm does not provide such a simple covariance-based relationship. To control

$$
\|V_\theta-V_{\theta^\star}\|_\infty,
$$

one needs to control

$$
\max_s |\phi(s)^{\texttt{T}}(\theta-\theta^\star)|.
$$

This depends on the largest feature vector and often introduces a uniform feature bound:

$$
\max_s \|\phi(s)\|_2.
$$

So even when the parameter error is small on average, a single large or rare feature vector can dominate the infinity-norm prediction error.

---

## 11. Average error can hide rare-state failure

The 2-norm's advantage is also its weakness.

Suppose an error function is

$$
e(s)=B
$$

on a rare set $\mathcal{B}$, and

$$
e(s)=0
$$

elsewhere. Let

$$
\nu(\mathcal{B})=p.
$$

Then

$$
\|e\|_\infty = B,
$$

but

$$
\|e\|_\nu = B\sqrt{p}.
$$

If $p$ is tiny, then $\|e\|_\nu$ may be small even though the worst-case error is huge.

This matters in RL because rare states are not always unimportant. They may correspond to:

- unsafe regions;
- high-reward opportunities;
- irreversible failures;
- states induced by the learned policy but not by the behavior policy;
- poorly explored state-action pairs.

Therefore, one should not blindly interpret a 2-norm TD guarantee as a uniform reliability guarantee.

A 2-norm theorem says the algorithm is accurate under the weighting distribution. An infinity-norm theorem says the algorithm is accurate everywhere.

Both are useful, but they answer different questions.

---

## 12. Robust TD: why 2-norm still fits naturally

In robust TD, the update may be corrupted. Instead of observing a clean TD sample, the learner may receive adversarially modified rewards, features, or update vectors.

The clean population direction is still

$$
h(\theta)=b-A\theta.
$$

A robust algorithm tries to construct an estimator $\widehat{h}(\theta)$ such that

$$
\widehat{h}(\theta)
\approx
h(\theta)
$$

in a suitable norm.

The TD recursion becomes

$$
\theta_{t+1}
=
\theta_t+
\alpha_t \widehat{h}(\theta_t).
$$

Write

$$
\widehat{h}(\theta_t)
=
-Ae_t + \zeta_t,
$$

where $\zeta_t$ includes stochastic estimation error and adversarial bias after robust aggregation.

Then

$$
e_{t+1}
=
e_t-
\alpha_tAe_t+
\alpha_t\zeta_t.
$$

The 2-norm expansion gives

$$
\begin{aligned}
\|e_{t+1}\|_2^2
=&
\|e_t\|_2^2
-2\alpha_t e_t^{\texttt{T}}Ae_t
+2\alpha_t e_t^{\texttt{T}}\zeta_t
+\alpha_t^2\|-Ae_t+\zeta_t\|_2^2.
\end{aligned}
$$

The robust error enters through the cross term

$$
2\alpha_t e_t^{\texttt{T}}\zeta_t.
$$

By Cauchy--Schwarz,

$$
|e_t^{\texttt{T}}\zeta_t|
\le
\|e_t\|_2\|\zeta_t\|_2.
$$

So if the robust estimator satisfies

$$
\|\zeta_t\|_2 \le \rho_t,
$$

then the corruption contributes at most

$$
2\alpha_t\rho_t\|e_t\|_2.
$$

This is again a clean inner-product argument. It is exactly the kind of reasoning that becomes much less transparent in infinity norm.

For robust Q-learning, however, max-norm is often more natural because the Bellman optimality recursion is already max-norm contractive, and the corruption can be bounded coordinate-wise. So the norm choice depends on whether the algorithm is closer to TD prediction or Bellman optimality control.

---

## 13. Decentralized and federated TD: Frobenius norm is the matrix version of the same story

The same geometry appears in multi-agent RL.

Suppose $N$ agents each maintain a parameter $\theta_{i,t}$. Stack them as rows of a matrix

$$
\Theta_t
=
\begin{bmatrix}
\theta_{1,t}^{\texttt{T}}\\
\theta_{2,t}^{\texttt{T}}\\
\vdots\\
\theta_{N,t}^{\texttt{T}}
\end{bmatrix}
\in \mathbb{R}^{N\times d}.
$$

Let the network average be

$$
\bar{\theta}_t
=
\frac{1}{N}\sum_{i=1}^N \theta_{i,t}.
$$

Then the total error can be decomposed into an optimization/statistical part and a consensus part:

$$
\Theta_t-
\mathbf{1}(\theta^\star)^{\texttt{T}}
=
\mathbf{1}(\bar{\theta}_t-\theta^\star)^{\texttt{T}}
+
\left(\Theta_t-
\mathbf{1}\bar{\theta}_t^{\texttt{T}}\right).
$$

The Frobenius norm gives the orthogonal decomposition

$$
\|\Theta_t-
\mathbf{1}(\theta^\star)^{\texttt{T}}\|_F^2
=
N\|\bar{\theta}_t-\theta^\star\|_2^2
+
\|\Theta_t-
\mathbf{1}\bar{\theta}_t^{\texttt{T}}\|_F^2.
$$

This identity is extremely useful. It says the total network error splits exactly into:

1. error of the network average from the desired fixed point;
2. disagreement among agents.

This clean Pythagorean decomposition is a 2-norm phenomenon. It is one of the main reasons decentralized stochastic approximation and decentralized TD are usually analyzed using Euclidean or Frobenius norms.

---

## 14. The relationship with finite-time rates

The norm choice also affects finite-time rates and constants.

A typical 2-norm TD finite-time result controls a quantity like

$$
\mathbb{E}\|\theta_T-\theta^\star\|_2^2
$$

or

$$
\mathbb{E}\|V_{\theta_T}-V_{\theta^\star}\|_\nu^2.
$$

The rate is governed by:

- the strong monotonicity parameter $\mu$;
- the noise variance;
- the feature covariance;
- the stepsize schedule;
- Markovian mixing, if the data are dependent;
- corruption radius, if the data are adversarially contaminated.

The proof often reduces to a recursion of the type

$$
E_{t+1}
\le
(1-c\alpha_t)E_t
+
C\alpha_t^2\sigma^2
+
\text{corruption terms}.
$$

Here

$$
E_t=\mathbb{E}\|\theta_t-\theta^\star\|_2^2.
$$

By contrast, a max-norm Q-learning result controls

$$
\mathbb{E}\|Q_T-Q^\star\|_\infty
$$

or a high-probability version. The proof usually relies on:

- the $\gamma$-contraction of $\mathcal{T}$ in infinity norm;
- coordinate-wise concentration of empirical Bellman updates;
- visitation lower bounds for every state-action pair;
- union bounds over $\mathcal{S}\times\mathcal{A}$.

So the two theories have different mathematical skeletons.

For TD:

$$
\text{mean direction} + \text{inner product} + \text{quadratic Lyapunov}.
$$

For Q-learning:

$$
\text{Bellman contraction} + \text{coordinate-wise control} + \text{max norm}.
$$

---

## 15. A simple example showing the difference

Consider a two-state prediction problem with error vector

$$
e=(e_1,e_2).
$$

Suppose the sampling distribution is

$$
\nu=(1-\omega,\omega),
$$

where $\omega\ll 1$. Then

$$
\|e\|_\nu^2
=
(1-\omega)e_1^2+
\omega e_2^2.
$$

If

$$
e_1=0,
\qquad
 e_2=\frac{1}{\sqrt{\omega}},
$$

then

$$
\|e\|_\nu^2
=
1,
$$

but

$$
\|e\|_\infty
=
\frac{1}{\sqrt{\omega}}.
$$

As $\omega\to 0$, the weighted 2-norm stays fixed while the infinity norm diverges.

This example captures the coverage issue perfectly. A rare state can carry enormous pointwise error while contributing only modestly to the average squared error.

This does not mean the weighted 2-norm is bad. It means it answers a different question.

If the rare state is genuinely unimportant under the target distribution, the 2-norm guarantee may be the right one. If the rare state is safety-critical or may be reached by a learned policy, infinity norm or stronger coverage assumptions become important.

---

## 16. A useful mental model

The infinity norm is a worst-case lens:

$$
\|e\|_\infty
=
\text{largest pointwise mistake}.
$$

The 2-norm is an energy lens:

$$
\|e\|_2^2
=
\text{total squared error energy}.
$$

The weighted 2-norm is a distributional energy lens:

$$
\|e\|_\nu^2
=
\text{average squared error under }\nu.
$$

The infinity norm is ideal when the Bellman operator itself reduces the largest error. This is the case for discounted optimal Bellman operators.

The 2-norm is ideal when the algorithm's mean update direction has a stable angle against the error. This is the case for TD-style stochastic approximation.

---

## 17. The main takeaway

Your intuition is exactly right.

TD updates are random, but their population direction is correct in an average sense. The analysis therefore needs a way to compare the current error with the mean update direction. The 2-norm gives exactly that through its inner product:

$$
\langle e_t,-Ae_t\rangle
=
-e_t^{\texttt{T}}Ae_t.
$$

If this quantity is negative enough, then the expected update moves toward the fixed point.

This leads to the fundamental TD drift inequality:

$$
\mathbb{E}
\left[
\|e_{t+1}\|_2^2
\mid \mathcal{F}_t
\right]
\le
(1-c\alpha_t)\|e_t\|_2^2
+
\text{noise}.
$$

That is the mathematical heart of why the 2-norm is so useful in TD.

The infinity norm does something different. It is not about angles or mean directions. It is about worst-coordinate error. It becomes powerful when the Bellman operator itself contracts the worst-coordinate error:

$$
\|\mathcal{T}Q-\mathcal{T}Q'\|_\infty
\le
\gamma\|Q-Q'\|_\infty.
$$

So the clean distinction is:

$$
\boxed{
\text{TD prediction: 2-norm because of inner-product drift.}
}
$$

$$
\boxed{
\text{Tabular Q-learning: infinity norm because of Bellman contraction.}
}
$$

A good way to remember this is:

> The 2-norm helps when the proof is about directions. The infinity norm helps when the proof is about the worst coordinate.

For reinforcement learning theory, both are indispensable. The important thing is to choose the norm that matches the geometry of the algorithm.
