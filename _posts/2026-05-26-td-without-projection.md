---
layout: post
title: "The Beauty of a Simple Proof: TD Learning Without Projection"
date: 2026-05-26
categories: [rl-blogs]
tags: [TD Learning, Function Approximation, Markovian Sampling, Stochastic Approximation]
math: true
---

## The Beauty of a Simple Proof: TD Learning Without Projection

There are some papers whose contribution is not only a theorem, but a way of seeing.

This paper, **"A Simple Finite-Time Analysis of TD Learning with Linear Function Approximation"** by **Aritra Mitra**, is one of those papers. On the surface, the problem is classical: temporal-difference learning with linear function approximation under Markovian sampling. But the beauty of the paper lies in how it takes a technically difficult stability question and resolves it with an argument that feels almost inevitable after one understands it.

This paper is also personally special to me because Aritra Mitra is my Ph.D. supervisor. One thing I have learned from working with him is that good theory is not about making simple objects look complicated. It is often about finding the right viewpoint so that a difficult proof becomes natural. This paper is a very good example of that philosophy.

The central message is simple:

> Can we analyze unprojected TD learning under Markovian sampling with the simplicity of a projected-TD proof?

The paper answers: **yes**.

And the key idea is beautifully clean:

> Treat the Markovian sampling error as a disturbance, prove by induction that the iterates remain bounded, and then show that the disturbance is small enough to behave like ordinary stochastic noise.

Let us unpack this carefully.

---

## 1. Policy Evaluation and the TD Learning Problem

We consider a Markov Decision Process with finite state space and finite action space. Fix a policy $$\mu$$. Under this policy, the MDP induces a Markov reward process with transition matrix $$P_\mu$$ and reward function $$R_\mu$$.

The value function of the policy is

$$
V^\mu(s)
=
\mathbb{E}
\left[
\sum_{t=0}^{\infty} \gamma^t R_\mu(s_t)
\mid s_0=s
\right],
$$

where $$\gamma \in (0,1)$$ is the discount factor.

The value function is the fixed point of the Bellman operator

$$
(T^\mu V)(s)
=
R_\mu(s)
+
\gamma
\sum_{s' \in \mathcal S}
P_\mu(s,s')V(s').
$$

In small tabular problems, one could try to estimate every component of $$V^\mu$$ directly. But in large state spaces, this is not realistic. So we approximate the value function using linear function approximation.

Let $$\phi(s) \in \mathbb{R}^K$$ be the feature vector of state $$s$$. We approximate

$$
\widehat V_\theta(s)
=
\langle \phi(s),\theta\rangle,
$$

where $$\theta \in \mathbb{R}^K$$ is the parameter vector.

The goal is to find the parameter $$\theta^\star$$ corresponding to the best projected Bellman fixed point.

---

## 2. The TD(0) Update

At time $$t$$, the learner observes a transition tuple

$$
X_t = (s_t,s_{t+1},r_t).
$$

The TD(0) update direction is

$$
g_t(\theta)
=
\left(
r_t
+
\gamma \langle \phi(s_{t+1}),\theta\rangle
-
\langle \phi(s_t),\theta\rangle
\right)\phi(s_t).
$$

The TD(0) recursion is

$$
\theta_{t+1}
=
\theta_t
+
\alpha g_t(\theta_t),
$$

where $$\alpha$$ is the step-size.

This algorithm is simple. But the analysis is not.

Why? Because the samples are not independent. They come from a single Markov chain trajectory:

$$
s_0,s_1,s_2,\ldots
$$

So the update noise is temporally correlated. This is what makes finite-time analysis under Markovian sampling difficult.

---

## 3. Why Projection Makes Life Easy

A common trick in earlier analyses is to use a projected TD update:

$$
\theta_{t+1}
=
\Pi_{\mathcal B}
\left(
\theta_t + \alpha g_t(\theta_t)
\right),
$$

where $$\Pi_{\mathcal B}$$ projects the iterate back to some bounded set.

This makes the proof easier because the iterates are automatically bounded:

$$
\|\theta_t\| \leq R.
$$

Once the iterates are bounded, the Markovian error terms can be controlled.

But algorithmically, this projection is artificial. The vanilla TD algorithm does not project. So the natural question is:

> Can we get the clean proof structure of projected TD without actually projecting?

This is exactly the question the paper answers.

---

## 4. The Steady-State TD Direction

The paper introduces the steady-state TD direction

$$
\bar g(\theta)
=
\mathbb{E}_{s \sim \pi,\; s' \sim P_\mu(\cdot \mid s)}
[g(\theta;X)],
$$

where $$\pi$$ is the stationary distribution of the Markov chain induced by the policy.

This is the TD direction one would see if the chain were already perfectly mixed.

The key structural fact is that $$\bar g(\theta)$$ behaves like a descent direction toward $$\theta^\star$$. In particular, one has

$$
\langle \theta^\star - \theta,\bar g(\theta)\rangle
\geq
\omega(1-\gamma)
\|\theta^\star-\theta\|^2,
$$

where $$\omega$$ is the smallest eigenvalue of

$$
\Sigma = \Phi^\top D \Phi.
$$

This inequality is the heart of the optimization analogy. It says that the steady-state TD dynamics contract toward $$\theta^\star$$.

If the update were

$$
\theta_{t+1}
=
\theta_t
+
\alpha \bar g(\theta_t),
$$

then the analysis would look like standard gradient descent.

But the actual update uses $$g_t(\theta_t)$$, not $$\bar g(\theta_t)$$. The difference

$$
g_t(\theta_t)-\bar g(\theta_t)
$$

is where the Markovian difficulty lives.

---

## 5. The Main Recursion

Define the mean-squared error

$$
d_t
=
\mathbb{E}
\left[
\|\theta_t-\theta^\star\|^2
\right].
$$

Also define the Markovian error term

$$
e_t
=
\mathbb{E}
\left[
\left\langle
\theta_t-\theta^\star,
g_t(\theta_t)-\bar g(\theta_t)
\right\rangle
\right].
$$

Starting from the TD update,

$$
\theta_{t+1}
=
\theta_t
+
\alpha g_t(\theta_t),
$$

we expand

$$
\|\theta_{t+1}-\theta^\star\|^2.
$$

This gives

$$
\|\theta_{t+1}-\theta^\star\|^2
=
\|\theta_t-\theta^\star\|^2
+
2\alpha
\langle
\theta_t-\theta^\star,
g_t(\theta_t)
\rangle
+
\alpha^2
\|g_t(\theta_t)\|^2.
$$

Now insert and subtract the steady-state direction:

$$
g_t(\theta_t)
=
\bar g(\theta_t)
+
\left(g_t(\theta_t)-\bar g(\theta_t)\right).
$$

Using the pseudo-gradient inequality and Lipschitz bounds on the TD update direction, the paper obtains the recursion

$$
d_{t+1}
\leq
\left(
1
-
2\alpha\omega(1-\gamma)
+
8\alpha^2
\right)d_t
+
32\alpha^2\sigma^2
+
2\alpha e_t.
$$

This recursion has three parts.

First,

$$
\left(
1
-
2\alpha\omega(1-\gamma)
+
8\alpha^2
\right)d_t
$$

is the steady-state contraction term.

Second,

$$
32\alpha^2\sigma^2
$$

is the ordinary variance term.

Third,

$$
2\alpha e_t
$$

is the Markovian sampling disturbance.

If the third term were absent, the proof would be essentially standard. It would look like a familiar stochastic approximation or SGD-style argument.

So the entire challenge is:

> How do we control $$e_t$$ without projecting the iterates?

---

## 6. The Beautiful Idea: Boundedness by Induction

This is the main conceptual move of the paper.

Instead of forcing boundedness through a projection step, the paper proves boundedness directly.

Define

$$
B
=
10\max
\left\{
\|\theta_0-\theta^\star\|^2,
\sigma^2
\right\}.
$$

The goal is to prove

$$
d_t \leq B
\qquad
\text{for all } t\geq 0.
$$

The proof has an elegant inductive structure.

First, for the initial mixing-time window, if

$$
\alpha \leq \frac{1}{8\tau},
$$

then the iterates cannot move too far, and one obtains

$$
\|\theta_k-\theta^\star\|^2
\leq B,
\qquad
k \in [\tau].
$$

This gives the base case.

Now suppose that for some $$t \geq \tau$$,

$$
d_k \leq B
\qquad
\text{for all } k \leq t.
$$

The goal is to prove

$$
d_{t+1} \leq B.
$$

This is where the induction becomes powerful.

Under the induction hypothesis, the paper first controls how much the iterate can move over one mixing window:

$$
\mathbb{E}
\left[
\|\theta_t-\theta_{t-\tau}\|^2
\right]
\leq
O(\alpha^2\tau^2B).
$$

This says: over the last $$\tau$$ steps, the parameter has not moved too much.

That matters because after $$\tau$$ steps, the Markov chain has mixed enough that the current sample is approximately stationary.

Using this, the paper shows

$$
e_t
\leq
O(\alpha\tau B).
$$

Therefore,

$$
2\alpha e_t
\leq
O(\alpha^2\tau B).
$$

This is the crucial point.

The Markovian error behaves like an $$O(\alpha^2)$$ perturbation, up to mixing-time factors. It is of the same order as the usual noise variance term.

Plugging this back into the main recursion gives

$$
d_{t+1}
\leq
\left(
1
-
2\alpha\omega(1-\gamma)
+
C\alpha^2\tau
\right)B.
$$

If the step-size satisfies

$$
\alpha
\leq
\frac{\omega(1-\gamma)}{C\tau},
$$

then the negative contraction dominates the positive perturbation. Hence,

$$
d_{t+1}
\leq
B.
$$

This closes the induction.

That is the whole magic.

No projection. No heavy stability machinery. Just a carefully designed induction that proves the iterates stay bounded because the recursion itself wants them to stay bounded.

---

## 7. The Final Finite-Time Recursion

Once boundedness is established, the rest becomes straightforward.

Since

$$
d_t \leq B
\qquad
\text{for all } t,
$$

the Markovian disturbance can be uniformly controlled. The paper obtains

$$
d_{t+1}
\leq
\left(
1-\alpha\omega(1-\gamma)
\right)d_t
+
O(\alpha^2\tau B).
$$

This is a clean finite-time recursion.

Unrolling it gives the qualitative behavior

$$
d_t
\lesssim
\exp
\left(
-\alpha\omega(1-\gamma)t
\right)d_0
+
\text{constant-step-size error floor}.
$$

The first term decays exponentially.

The second term is the steady-state error floor caused by constant step-size stochastic approximation under Markovian noise.

So the proof tells a very clean story:

1. The steady-state TD operator contracts.
2. Markovian sampling creates a disturbance.
3. The disturbance depends on the iterates.
4. Induction proves the iterates remain bounded.
5. Bounded iterates imply bounded disturbance.
6. The usual finite-time convergence recursion follows.

This is why the proof is simple.

Not simple because the problem is trivial, but simple because the right decomposition makes the proof almost modular.

---

## 8. Why This Paper Feels Elegant

The elegance of the paper comes from the fact that it does not fight the Markovian noise directly.

Instead, it asks:

> What would make the Markovian term harmless?

The answer is: bounded iterates.

Then it asks:

> Can boundedness be proved without projection?

The answer is: yes, by induction.

This is a very useful proof philosophy in reinforcement learning.

Many RL algorithms are stochastic approximation schemes of the form

$$
\theta_{t+1}
=
\theta_t
+
\alpha g(\theta_t;X_t).
$$

If the data were i.i.d., the proof would be much easier. But in RL, the data usually comes from a trajectory, so the noise is Markovian. This creates extra terms that depend on the current iterate.

The paper shows that these terms can be treated as iterate-dependent disturbances. Once we prove that the iterates stay bounded, the disturbance becomes small.

That viewpoint is powerful beyond TD learning.

---

## 9. Why the Paper Is Simple, But Not Obvious

There is an important difference between a simple proof and an easy proof.

This paper is simple in the sense that, after reading it, the proof strategy can be summarized in one sentence:

> Prove iterate boundedness by induction, then use mixing to show that the Markovian error is only a small perturbation.

But it is not obvious.

The non-obvious part is realizing that one should not first try to directly prove convergence. Instead, one should first prove boundedness. Once boundedness is available, convergence almost falls out of the standard recursion.

This is a very mature way of doing stochastic approximation theory.

The proof separates the problem into two clean layers:

### Layer 1: Stability

Show

$$
d_t \leq B
\qquad
\forall t.
$$

### Layer 2: Convergence

Use stability to prove

$$
d_{t+1}
\leq
\left(
1-\alpha\omega(1-\gamma)
\right)d_t
+
O(\alpha^2\tau B).
$$

This separation is what makes the paper beautiful.

---

## 10. Broader Lesson for RL Theory

A major theme in modern RL theory is that algorithms often look simple, but their stochastic processes are not.

TD learning is a perfect example. The update is one line:

$$
\theta_{t+1}
=
\theta_t
+
\alpha
\left(
r_t
+
\gamma \langle \phi(s_{t+1}),\theta_t\rangle
-
\langle \phi(s_t),\theta_t\rangle
\right)\phi(s_t).
$$

But because the samples come from a Markov chain, the analysis requires care.

The paper teaches a useful lesson:

> The right proof technique can make a Markovian RL algorithm look almost as clean as SGD.

That is not just aesthetically pleasing. It is technically important because many modern RL algorithms involve additional perturbations: delays, compression, asynchrony, communication noise, and adversarial corruption.

The inductive boundedness idea gives a possible template for studying such algorithms.

If the perturbed update can be written as

$$
\theta_{t+1}
=
\theta_t
+
\alpha \bar g(\theta_t)
+
\alpha e_{1,t}
+
\alpha e_{2,t},
$$

where $$e_{1,t}$$ is Markovian noise and $$e_{2,t}$$ is some additional perturbation, then one can try to control both disturbances through a boundedness argument.

This is why the paper is more than a TD learning proof. It is a proof technique.

---

## 11. My Personal Takeaway

What I like most about this paper is that it reflects a style of theory I deeply admire.

The paper does not try to make TD learning look mysterious. It does the opposite. It says: the steady-state object is contractive, the Markovian term is a disturbance, and the only real issue is stability. Then it solves stability using induction.

That is beautiful.

As someone working in reinforcement learning theory, I find this especially inspiring because many of the algorithms I think about also have this structure: a clean ideal operator plus a messy disturbance. The challenge is often not to invent a completely new proof from scratch, but to find the right way to show that the disturbance cannot destabilize the algorithm.

This paper gives one such way.

And perhaps that is the best kind of theoretical contribution: after reading it, the proof feels simple, but only because someone found the right lens first.

---

## 12. One-Line Summary

TD learning with linear function approximation under Markovian sampling is hard to analyze because the noise depends on the trajectory, but this paper shows that a simple inductive boundedness argument can replace projection and recover a clean finite-time analysis.
