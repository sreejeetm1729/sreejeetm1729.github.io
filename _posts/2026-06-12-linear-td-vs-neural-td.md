---
title: "Linear TD vs Neural TD: What really changes ?"
date: 2026-06-12
categories: [rl-blogs]
rl_section: research-papers
tags: [reinforcement-learning, temporal-difference-learning, linear-td, neural-td, function-approximation, mspbe]
math: true
---

## Abstract

This note compares two population temporal-difference methods for policy evaluation under the same i.i.d. stationary sampling model: projected linear TD and projected neural TD with an overparameterized two-layer ReLU network. The emphasis is on the function classes. In linear TD, the Bellman target is projected onto the fixed linear class $$\mathcal{F}_{\mathrm{lin}}=\left\{x\mapsto \phi(x)^\top\theta:\theta\in\Theta\right\}.$$ In neural TD, the paper of Cai, Yang, Lee, and Wang analyzes a projected neural update by comparing it to a frozen random-feature class $$\mathcal{F}_{B,m}=\left\{x\mapsto \frac{1}{\sqrt m}\sum_{r=1}^m b_r\mathbf{1}\{W_r(0)^\top x>0\}W_r^\top x: W\in S_B\right\}.$$ The note focuses only on the population version: no stochastic-semigradient variance and no Markovian sampling bias. For the linear method, the proof is a standard projected strongly-monotone operator argument. For the neural method, the proof has the same projection/telescoping skeleton, but it requires an additional overparameterization step: the true neural network must remain close enough to its frozen linearization so that the population neural semigradient has a one-point monotonicity property relative to the frozen class $\mathcal{F}_{B,m}$.

## Purpose of this note

The goal is to write the two population analyses in a way that makes their difference completely visible. Both algorithms are TD methods. Both use projection. Both can be written as projected operator iterations. But the objects onto which the Bellman target is projected are fundamentally different.

|                         | Projected linear TD                                  | Projected neural TD                                                                |
| :---------------------- | :--------------------------------------------------- | :--------------------------------------------------------------------------------- |
| Approximation           | $Q_\theta(x)=\phi(x)^\top\theta$                     | $\widehat Q(x;W)=m^{-1/2}\sum_{r=1}^m b_r\sigma(W_r^\top x)$                       |
| Trainable parameter     | $\theta\in\mathbb{R}^d$                              | $W=(W_1,\ldots,W_m)\in\mathbb{R}^{md}$                                             |
| Algorithmic projection  | $\theta\in\Theta$                                    | $W\in S_B=\{W:\left\lVert W-W(0) \right\rVert_2\le B\}$                            |
| Function class in MSPBE | $\mathcal{F}_{\mathrm{lin}}$                         | Frozen random-feature class $\mathcal{F}_{B,m}$                                    |
| Feature map in proof    | fixed $\phi(x)$                                      | frozen tangent/random feature $\Phi_0(x)$                                          |
| Population operator     | affine: $A\theta-b$                                  | nonlinear: $\mathbb{E}[\delta_W\nabla_W\widehat Q(x;W)]$                           |
| Why proof works         | global strong monotonicity of the linear TD operator | overparameterization gives implicit local linearization and one-point monotonicity |

Cai, Yang, Lee, and Wang (2019) prove global convergence of neural TD to the global optimum of the mean-squared projected Bellman error (MSPBE) under overparameterization. Its population result is Theorem 4.4. This note does not reproduce every ReLU-concentration estimate from that paper; instead, it states the exact finite-width/local-linearization lemmas needed and then proves the population recursion in full. This is the right level if the purpose is to understand how the neural proof differs from the linear projected TD proof.

## Common policy-evaluation setup

Let $\mathcal{X}$ denote the state-action space. We write $$x=(s,a),\qquad x'=(s',a')$$ when we want to emphasize the reinforcement-learning interpretation. A fixed policy $\pi$ induces a Markov chain over state-action pairs. Let $\mu$ be its stationary distribution. In this note we impose the population/i.i.d. abstraction: $$(x,r,x')\sim\mathcal{D},
    \qquad x\sim\mu,
    \qquad x'\mid x \sim P^\pi(\cdot\mid x),$$ and the population update is obtained by taking expectation with respect to this distribution. Since we focus on population updates, there is no sample noise term.

For any measurable value function $f:\mathcal{X}\to\mathbb{R}$, define the Bellman operator $$\label{eq:bellman-operator}
    (T^\pi f)(x)
    :=
    \mathbb{E}[r+\gamma f(x')\mid x],
    \qquad \gamma\in(0,1).$$ Equivalently, the Bellman residual is $$\label{eq:bellman-residual-function}
    \delta_f(x,r,x')
    :=
    f(x)-r-\gamma f(x').$$ We use the $L_2(\mu)$ inner product and norm $$\label{eq:l2mu}
    \left\langle f,g\right\rangle_\mu:=\mathbb{E}_{x\sim\mu}[f(x)g(x)],
    \qquad
    \left\lVert f \right\rVert_\mu^2:=\mathbb{E}_{x\sim\mu}[f(x)^2].$$ When the argument is a Euclidean vector, $\left\lVert \cdot \right\rVert_2$ denotes the Euclidean norm.

<div id="ass:stationary-marginal" class="assumption">

**Assumption 1** (Stationarity of the next-state marginal). *If $x\sim\mu$ and $x'\sim P^\pi(\cdot\mid x)$, then $x'\sim\mu$.*

</div>

This assumption is automatic when $\mu$ is the stationary distribution of the policy-induced Markov chain. It implies the contraction inequality $$\label{eq:stationarity-cauchy}
    \mathbb{E}[h(x)h(x')]
    \le
    \sqrt{\mathbb{E}[h(x)^2] \mathbb{E}[h(x')^2]}
    =
    \left\lVert h \right\rVert_\mu^2.$$ This one-line inequality is the source of the factor $(1-\gamma)$ in both the linear and neural population analyses.

## Bellman projection, MSPBE, and the role of the Function Class

The symbol $\mathcal{F}$ denotes the approximation class used in the projected Bellman equation. Given a closed convex subset $\mathcal{F}\subset L_2(\mu)$, the $L_2(\mu)$ projection of a function $g$ onto $\mathcal{F}$ is $$\label{eq:function-projection-general}
    \Pi_{\mathcal{F}}g
    :=
    \operatorname*{arg\,min}_{f\in\mathcal{F}} \left\lVert f-g \right\rVert_\mu^2.$$ The projected Bellman equation is $$\label{eq:pbe-general}
    f^\star
    =
    \Pi_{\mathcal{F}}T^\pi f^\star.$$ The MSPBE associated with $\mathcal{F}$ is $$\label{eq:mspbe-general}
    \mathrm{MSPBE}_{\mathcal{F}}(f)
    :=
    \left\lVert f-\Pi_{\mathcal{F}}T^\pi f \right\rVert_\mu^2.$$ Thus, the function class is not a cosmetic detail. Changing $\mathcal{F}$ changes the projected Bellman fixed point, the MSPBE objective, and the meaning of “convergence.”

### Variational characterization of the projection

For a closed convex set $\mathcal{F}$ in a Hilbert space, $f_\mathcal{F}=\Pi_\mathcal{F}g$ if and only if $$\label{eq:projection-vi-function}
    \left\langle f_\mathcal{F}-g,f-f_\mathcal{F}\right\rangle_\mu\ge 0,
    \qquad \forall f\in\mathcal{F}.$$ Therefore, a projected Bellman fixed point $f^\star=\Pi_\mathcal{F}T^\pi f^\star$ satisfies $$\label{eq:pbe-vi-function}
    \left\langle f^\star-T^\pi f^\star,f-f^\star\right\rangle_\mu\ge 0,
    \qquad \forall f\in\mathcal{F}.$$ This variational inequality is the cleanest way to compare linear TD and neural TD. In the linear case, $\mathcal{F}$ is a fixed finite-dimensional span. In the neural case, the class appearing in the proof is the frozen random-feature class $\mathcal{F}_{B,m}$.

## Projected linear TD: function class and population update

### Linear value approximation

Let $$\phi:\mathcal{X}\to\mathbb{R}^d$$ be a fixed feature map. The linear approximation is $$\label{eq:linear-model}
    Q_\theta(x)=\phi(x)^\top\theta,
    \qquad \theta\in\mathbb{R}^d.$$ Let $\Theta\subset\mathbb{R}^d$ be a compact convex feasible set. The corresponding linear function class is $$\label{eq:F-linear}
    \boxed{
    \mathcal{F}_{\mathrm{lin}}
    :=
    \left\{x\mapsto \phi(x)^\top\theta:\theta\in\Theta\right\}.
    }$$ This is the linear analogue of the neural class $\mathcal{F}_{B,m}$ defined later.

If $\Theta=\mathbb{R}^d$, then $\mathcal{F}_{\mathrm{lin}}$ is a linear subspace. If $\Theta$ is a compact convex set, then $\mathcal{F}_{\mathrm{lin}}$ is a closed convex subset of the linear span of $\phi$. The algorithmic projection $\Pi_\Theta$ in parameter space induces the restriction that the Bellman projection is onto the feasible linear class $\mathcal{F}_{\mathrm{lin}}$.

### Linear TD residual

For $Q_\theta(x)=\phi(x)^\top\theta$, define $$\label{eq:linear-residual}
    \delta_\theta(x,r,x')
    :=
    \phi(x)^\top\theta-r-\gamma\phi(x')^\top\theta.$$ The population linear TD semigradient is $$\label{eq:linear-pop-gradient}
    \bar g_{\mathrm{lin}}(\theta)
    :=
    \mathbb{E}[\delta_\theta(x,r,x')\phi(x)].$$ Since the model is linear, this population direction is affine: $$\label{eq:linear-A-b}
    \bar g_{\mathrm{lin}}(\theta)=A\theta-b,$$ where $$\label{eq:A-and-b}
    A:=\mathbb{E}[\phi(x)(\phi(x)-\gamma\phi(x'))^\top],
    \qquad
    b:=\mathbb{E}[r\phi(x)].$$ This is the key simplification in linear TD. The feature map is fixed, so the population TD operator is a fixed affine map.

### Population projected linear TD

The population version of projected linear TD is $$\label{eq:linear-pop-update}
    \boxed{
    \theta_{t+1}
    =
    \Pi_\Theta\bigl(\theta_t-\eta\bar g_{\mathrm{lin}}(\theta_t)\bigr)
    =
    \Pi_\Theta\bigl(\theta_t-\eta(A\theta_t-b)\bigr).
    }$$

**Population projected linear TD.**

```text
Input: feasible set Θ, stepsize η, initialization θ₀ ∈ Θ
For t = 0, 1, ..., T-1:
    Compute ḡ_lin(θ_t) = E[δ_{θ_t}(x,r,x') φ(x)]
    Update θ_{t+1} = Π_Θ(θ_t - η ḡ_lin(θ_t))
Output: Q_{θ_T}, or the averaged function T^{-1} Σ_{t=0}^{T-1} Q_{θ_t}
```

### Projected Bellman fixed point for the linear class

Suppose first that the fixed point is unconstrained and lies in $\Theta$. Then $\theta^\star$ solves $$\label{eq:linear-fixed-point}
    A\theta^\star=b.$$ Equivalently, $$\label{eq:linear-pbe}
    Q_{\theta^\star}
    =
    \Pi_{\mathcal{F}_{\mathrm{lin}}}T^\pi Q_{\theta^\star}.$$ Indeed, the normal equation for the projection of $T^\pi Q_\theta$ onto the linear span of $\phi$ is $$\mathbb{E} \left[\phi(x) \left(\phi(x)^\top\theta - (T^\pi Q_\theta)(x)\right)\right]=0.$$ Since $$(T^\pi Q_\theta)(x)=\mathbb{E}[r+\gamma\phi(x')^\top\theta\mid x],$$ the normal equation is exactly $$\mathbb{E}[\phi(x)(\phi(x)^\top\theta-r-\gamma\phi(x')^\top\theta)]=0,$$ i.e., $A\theta=b$.

If the feasible set $\Theta$ is active, the projected solution can instead be written as the variational inequality $$\label{eq:linear-vi}
    \left\langle A\theta^\star-b,\theta-\theta^\star\right\rangle\ge 0,
    \qquad \forall \theta\in\Theta.$$ For most finite-time TD presentations, one chooses $\Theta$ large enough so that the unconstrained fixed point belongs to $\Theta$; then $A\theta^\star=b$ and the projection is used only to keep iterates bounded. This is the regime we use for the cleanest side-by-side comparison.

## Population proof for projected linear TD

### Assumptions

<div id="ass:linear-bounded" class="assumption">

**Assumption 2** (Bounded features and rewards). *There exist constants $L_\phi,R_{\max}<\infty$ such that $$\left\lVert \phi(x) \right\rVert_2\le L_\phi,
    \qquad
    \left\lvert r \right\rvert\le R_{\max}$$ for all samples $(x,r,x')$.*

</div>

<div id="ass:linear-fixed-in-theta" class="assumption">

**Assumption 3** (Linear TD fixed point inside the projection set). *There exists $\theta^\star\in\Theta$ such that $$A\theta^\star=b.$$*

</div>

<div id="ass:linear-strong-monotone" class="assumption">

**Assumption 4** (Strong monotonicity of the linear TD operator). *There exists $\mu_{\mathrm{lin}}>0$ such that $$\label{eq:linear-strong-monotone}
    \left\langle \theta-\theta^\star,A(\theta-\theta^\star)\right\rangle
    \ge
    \mu_{\mathrm{lin}} \left\lVert \theta-\theta^\star \right\rVert_2^2,
    \qquad \forall \theta\in\Theta.$$*

</div>

In classical TD with linear function approximation, this condition follows from the positive definiteness induced by the feature covariance and the contraction of the Bellman operator. For example, if features are normalized and the feature covariance has minimum eigenvalue $\omega>0$, then one often obtains a monotonicity constant of order $$\mu_{\mathrm{lin}}\asymp \omega(1-\gamma).$$ This is exactly the linear phenomenon that is absent globally in neural TD.

<div id="ass:linear-lipschitz" class="assumption">

**Assumption 5** (Lipschitzness). *Let $$L_A:=\left\lVert A \right\rVert_2.$$ The stepsize satisfies $$\label{eq:linear-stepsize}
    0<\eta\le \frac{\mu_{\mathrm{lin}}}{L_A^2}.$$*

</div>

### One-step contraction

<div id="lem:linear-one-step" class="lemma">

**Lemma 1** (Projected population linear TD contraction). *Under Assumption 3,Assumption 4,Assumption 5, the population projected linear TD update the displayed equation satisfies $$\label{eq:linear-one-step-contraction}
    \left\lVert \theta_{t+1}-\theta^\star \right\rVert_2^2
    \le
    \left(1-\eta\mu_{\mathrm{lin}}\right)
    \left\lVert \theta_t-\theta^\star \right\rVert_2^2.$$*

</div>

<div class="proof">

*Proof.* Since Euclidean projection onto a closed convex set is nonexpansive and $\theta^\star\in\Theta$, $$\begin{aligned}
    \left\lVert \theta_{t+1}-\theta^\star \right\rVert_2^2
    &=
    \left\lVert \Pi_\Theta(\theta_t-\eta(A\theta_t-b))-\Pi_\Theta(\theta^\star) \right\rVert_2^2 \\
    &\le
    \left\lVert \theta_t-\eta(A\theta_t-b)-\theta^\star \right\rVert_2^2.
\end{aligned}$$ Using $A\theta^\star=b$, we rewrite $$A\theta_t-b=A(\theta_t-\theta^\star).$$ Therefore $$\begin{aligned}
    \left\lVert \theta_{t+1}-\theta^\star \right\rVert_2^2
    &\le
    \left\lVert \theta_t-\theta^\star-\eta A(\theta_t-\theta^\star) \right\rVert_2^2 \\
    &=
    \left\lVert \theta_t-\theta^\star \right\rVert_2^2
    -2\eta\left\langle \theta_t-\theta^\star,A(\theta_t-\theta^\star)\right\rangle
    +\eta^2 \left\lVert A(\theta_t-\theta^\star) \right\rVert_2^2.
\end{aligned}$$ By strong monotonicity, $$\left\langle \theta_t-\theta^\star,A(\theta_t-\theta^\star)\right\rangle
    \ge
    \mu_{\mathrm{lin}} \left\lVert \theta_t-\theta^\star \right\rVert_2^2.$$ By Lipschitzness, $$\left\lVert A(\theta_t-\theta^\star) \right\rVert_2^2
    \le
    L_A^2 \left\lVert \theta_t-\theta^\star \right\rVert_2^2.$$ Combining these two estimates gives $$\begin{aligned}
    \left\lVert \theta_{t+1}-\theta^\star \right\rVert_2^2
    &\le
    \left(1-2\eta\mu_{\mathrm{lin}}+\eta^2L_A^2\right)
    \left\lVert \theta_t-\theta^\star \right\rVert_2^2.
\end{aligned}$$ The stepsize condition $\eta L_A^2\le \mu_{\mathrm{lin}}$ implies $$1-2\eta\mu_{\mathrm{lin}}+\eta^2L_A^2
    \le
    1-\eta\mu_{\mathrm{lin}},$$ which proves the result. ◻

</div>

### Finite-time population theorem

<div id="thm:linear-population" class="theorem">

**Theorem 1** (Population projected linear TD). *Under the assumptions of Lemma 1, for all $T\ge 0$, $$\label{eq:linear-final-parameter}
    \left\lVert \theta_T-\theta^\star \right\rVert_2^2
    \le
    (1-\eta\mu_{\mathrm{lin}})^T
    \left\lVert \theta_0-\theta^\star \right\rVert_2^2.$$ Consequently, $$\label{eq:linear-final-function}
    \left\lVert Q_{\theta_T}-Q_{\theta^\star} \right\rVert_\mu^2
    \le
    \lambda_{\max}(C_\phi)
    (1-\eta\mu_{\mathrm{lin}})^T
    \left\lVert \theta_0-\theta^\star \right\rVert_2^2,$$ where $$C_\phi:=\mathbb{E}[\phi(x)\phi(x)^\top].$$*

</div>

<div class="proof">

*Proof.* Iterating Lemma 1 gives the displayed equation. For the function error, $$\begin{aligned}
    \left\lVert Q_{\theta_T}-Q_{\theta^\star} \right\rVert_\mu^2
    &=
    \mathbb{E} \left[(\phi(x)^\top(\theta_T-\theta^\star))^2\right] \\
    &=
    (\theta_T-\theta^\star)^\top C_\phi(\theta_T-\theta^\star) \\
    &\le
    \lambda_{\max}(C_\phi) \left\lVert \theta_T-\theta^\star \right\rVert_2^2.
\end{aligned}$$ Substituting the displayed equation completes the proof. ◻

</div>

<div class="remark">

**Remark 1** (Why population linear TD is easy). *There are three reasons the projected population linear proof is short.*

1.  *The features $\phi(x)$ are fixed.*

2.  *The population TD direction is affine: $\bar g_{\mathrm{lin}}(\theta)=A\theta-b$.*

3.  *Strong monotonicity is global in parameter space.*

*The proof does not need overparameterization, local linearization, random-feature concentration, or a lazy-training argument.*

</div>

## Projected neural TD: network, projection set, and function classes

We now turn to the neural version. This section is the heart of the comparison. The important point is that neural TD has several related function classes:

1.  the actual nonlinear neural-network class;

2.  the local linearization class around a point $W^\dagger$, denoted $\mathcal{F}_{B,m}^\dagger$;

3.  the initialization-frozen random-feature class, denoted $\mathcal{F}_{B,m}$.

The population proof in Cai, Yang, Lee, and Wang (2019) uses the third class as the clean reference object.

### Two-layer ReLU model

Let $x\in\mathcal{X}\subseteq\mathbb{R}^d$. The two-layer ReLU value approximator is $$\label{eq:neural-model}
    \widehat Q(x;W)
    :=
    \frac{1}{\sqrt m}
    \sum_{r=1}^m b_r\sigma(W_r^\top x),$$ where $$\sigma(z)=\max\{z,0\},
    \qquad
    W=(W_1,\ldots,W_m)\in\mathbb{R}^{md},
    \qquad
    b_r\in\{-1,+1\}.$$ The output signs $b_r$ are fixed, while the hidden weights $W_r$ are trained.

The neural TD residual is $$\label{eq:neural-residual}
    \delta_W(x,r,x')
    :=
    \widehat Q(x;W)-r-\gamma\widehat Q(x';W).$$ The population neural TD semigradient is $$\label{eq:neural-pop-gradient}
    \bar g_{\mathrm{NN}}(W)
    :=
    \mathbb{E}[\delta_W(x,r,x')\nabla_W\widehat Q(x;W)].$$ Unlike the linear case, $\bar g_{\mathrm{NN}}(W)$ is not affine in $W$.

### Algorithmic projection set

Let $W(0)$ be the random initialization. For a radius $B>0$, define $$\label{eq:SB}
    \boxed{
    S_B
    :=
    \left\{W\in\mathbb{R}^{md}:\left\lVert W-W(0) \right\rVert_2\le B\right\}.
    }$$ The population projected neural TD update is $$\label{eq:neural-pop-update}
    \boxed{
    W(t+1)
    =
    \Pi_{S_B} \left(W(t)-\eta\bar g_{\mathrm{NN}}(W(t))\right).
    }$$ This is the population analogue of the neural TD update in Cai, Yang, Lee, and Wang (2019).

**Population projected neural TD.**

```text
Input: initialization W(0), fixed signs b₁,...,b_m, radius B, stepsize η
For t = 0, 1, ..., T-1:
    Compute ḡ_NN(W(t)) = E[δ_{W(t)}(x,r,x') ∇_W Q̂(x;W(t))]
    Update W(t+1) = Π_{S_B}(W(t) - η ḡ_NN(W(t)))
Output: an averaged network Q̂_out, usually formed by averaging iterates/functions
```

### The actual neural-network class

The most literal neural-network class is $$\label{eq:F-neural-nonlinear}
    \mathcal{F}_{\mathrm{NN},B}^{\mathrm{nonlin}}
    :=
    \left\{x\mapsto \widehat Q(x;W):W\in S_B\right\}.$$ This is nonlinear in $W$ because the gates $\mathbf{1}\{W_r^\top x>0\}$ change with $W$. If one tried to analyze projection directly onto this class, the geometry would be nonconvex and difficult.

The key idea in the neural TD proof is not to treat the displayed equation as an arbitrary nonconvex class. Instead, overparameterization makes the network behave like its local linearization.

### Local linearization class around $W^{\dagger}$

Fix a reference point $$W^\dagger=(W_1^\dagger,\ldots,W_m^\dagger).$$ The activation pattern at $W^\dagger$ defines the local linearization $$\label{eq:local-linearized-model}
    \widehat Q^\dagger_0(x;W)
    :=
    \frac{1}{\sqrt m}
    \sum_{r=1}^m
    b_r\mathbf{1}\{(W_r^\dagger)^\top x>0\}W_r^\top x.$$ The corresponding function class is $$\label{eq:F-dagger}
    \boxed{
    \mathcal{F}_{B,m}^\dagger
    :=
    \left\{
    x\mapsto
    \frac{1}{\sqrt m}
    \sum_{r=1}^m
    b_r\mathbf{1}\{(W_r^\dagger)^\top x>0\}W_r^\top x
    : W\in S_B
    \right\}.
    }$$ This class is linear in the trainable weights $W$ after the gates are frozen at $W^\dagger$.

The stationarity condition of the projected neural population update at a point $W^\dagger$ can be related to the projected Bellman equation over this local class. Roughly, $$\widehat Q(\cdot;W^\dagger)
    \approx
    \Pi_{\mathcal{F}_{B,m}^\dagger}T^\pi\widehat Q(\cdot;W^\dagger).$$ The exact statement in the neural paper uses the variational inequality form.

### Frozen random-feature class $\mathcal{F}_{B,m}$

The class that is most useful for global analysis is obtained by freezing the gates at initialization $W(0)$: $$\label{eq:Q0-frozen}
    \boxed{
    \widehat Q_0(x;W)
    :=
    \frac{1}{\sqrt m}
    \sum_{r=1}^m
    b_r\mathbf{1}\{W_r(0)^\top x>0\}W_r^\top x.
    }$$ The associated function class is $$\label{eq:F-B-m}
    \boxed{
    \mathcal{F}_{B,m}
    :=
    \left\{
    x\mapsto
    \frac{1}{\sqrt m}
    \sum_{r=1}^m
    b_r\mathbf{1}\{W_r(0)^\top x>0\}W_r^\top x
    : W\in S_B
    \right\}.
    }$$ This is the class denoted $\mathcal{F}_{B,m}$ in the neural TD analysis.

### Why $\mathcal{F}_{B,m}$ is a linear class

Define the frozen random-feature map $$\label{eq:Phi0}
    \Phi_0(x)
    :=
    \frac{1}{\sqrt m}
    \begin{bmatrix}
    b_1\mathbf{1}\{W_1(0)^\top x>0\}x\\
    b_2\mathbf{1}\{W_2(0)^\top x>0\}x\\
    \vdots\\
    b_m\mathbf{1}\{W_m(0)^\top x>0\}x
    \end{bmatrix}
    \in\mathbb{R}^{md}.$$ Then $$\label{eq:Q0-inner-product}
    \widehat Q_0(x;W)=\Phi_0(x)^\top W.$$ Therefore $$\label{eq:F-Bm-linear-form}
    \mathcal{F}_{B,m}
    =
    \left\{x\mapsto \Phi_0(x)^\top W:W\in S_B\right\}.$$ This is why neural TD can be compared to linear TD: $\mathcal{F}_{B,m}$ is a random linear class. Its feature map is not hand-designed; it is induced by the random initialized ReLU gates.

<div class="remark">

**Remark 2** (Precise comparison with linear TD). *The side-by-side feature-class comparison is $$\boxed{
    \mathcal{F}_{\mathrm{lin}}
    =
    \left\{x\mapsto \phi(x)^\top\theta:\theta\in\Theta\right\}
    }$$ and $$\boxed{
    \mathcal{F}_{B,m}
    =
    \left\{x\mapsto \Phi_0(x)^\top W:W\in S_B\right\}.
    }$$ Thus, once the ReLU gates are frozen, neural TD becomes linear TD in the random feature map $\Phi_0(x)$. The difficult part is proving that the actual neural update using $\widehat Q(x;W)$ behaves like the frozen update using $\widehat Q_0(x;W)$.*

</div>

### Frozen residual and frozen population semigradient

Define the frozen TD residual $$\label{eq:frozen-residual}
    \delta_0(x,r,x';W)
    :=
    \widehat Q_0(x;W)-r-\gamma\widehat Q_0(x';W).$$ The frozen population semigradient is $$\label{eq:frozen-pop-gradient}
    \bar g_0(W)
    :=
    \mathbb{E}[\delta_0(x,r,x';W)\nabla_W\widehat Q_0(x;W)].$$ Since $\widehat Q_0(x;W)=\Phi_0(x)^\top W$, we have $$\label{eq:grad-Q0}
    \nabla_W\widehat Q_0(x;W)=\Phi_0(x),$$ and hence $$\label{eq:frozen-pop-gradient-linear}
    \bar g_0(W)
    =
    \mathbb{E} \left[
    \left(\Phi_0(x)^\top W-r-\gamma\Phi_0(x')^\top W\right)\Phi_0(x)
    \right].$$ Thus $$\label{eq:frozen-A-b}
    \bar g_0(W)=A_0W-b_0,$$ where $$\label{eq:A0-b0}
    A_0:=\mathbb{E}[\Phi_0(x)(\Phi_0(x)-\gamma\Phi_0(x'))^\top],
    \qquad
    b_0:=\mathbb{E}[r\Phi_0(x)].$$ This is exactly the same algebraic structure as linear TD, but with $\phi$ replaced by $\Phi_0$ and $\theta$ replaced by $W$.

## The neural projected Bellman point over $\mathcal{F}_{B,m}$

### Approximate stationary point $W^\star$

The reference point in the neural population theorem is not defined by the actual nonlinear class the displayed equation. It is defined using the frozen class $\mathcal{F}_{B,m}$.

<div id="def:Wstar" class="definition">

**Definition 1** (Frozen approximate stationary point). *A point $W^\star\in S_B$ is a frozen approximate stationary point if $$\label{eq:Wstar-VI}
    \left\langle \bar g_0(W^\star),W-W^\star\right\rangle
    \ge 0,
    \qquad \forall W\in S_B.$$ Equivalently, $$\label{eq:Wstar-VI-expanded}
    \mathbb{E}[\delta_0(x,r,x';W^\star)\Phi_0(x)]^\top(W-W^\star)
    \ge 0,
    \qquad \forall W\in S_B.$$*

</div>

This is the exact analogue of the projected linear TD variational inequality. It says that $W^\star$ is a projected Bellman fixed point in the frozen random-feature class $\mathcal{F}_{B,m}$.

### Equivalence to projection onto $\mathcal{F}_{B,m}$

<div id="prop:frozen-vi-pbe" class="proposition">

**Proposition 1** (Frozen VI equals projected Bellman equation). *Let $f^\star_0(x)=\widehat Q_0(x;W^\star)$. Then the displayed equation is equivalent to $$\label{eq:frozen-pbe}
    f^\star_0
    =
    \Pi_{\mathcal{F}_{B,m}}T^\pi f^\star_0.$$ Consequently, $f^\star_0$ is a global minimizer of $\mathrm{MSPBE}_{\mathcal{F}_{B,m}}$.*

</div>

<div class="proof">

*Proof.* For any $W\in S_B$, let $$f_W(x)=\widehat Q_0(x;W)=\Phi_0(x)^\top W.$$ Using the definition of the Bellman operator, $$\begin{aligned}
    \left\langle f^\star_0-T^\pi f^\star_0,f_W-f^\star_0\right\rangle_\mu
    &=
    \mathbb{E} \left[
        \left(f^\star_0(x)-r-\gamma f^\star_0(x')\right)
        \left(f_W(x)-f^\star_0(x)\right)
    \right] \\
    &=
    \mathbb{E} \left[
        \delta_0(x,r,x';W^\star)
        \Phi_0(x)^\top(W-W^\star)
    \right] \\
    &=
    \left\langle \bar g_0(W^\star),W-W^\star\right\rangle.
\end{aligned}$$ Thus the displayed equation is exactly $$\left\langle f^\star_0-T^\pi f^\star_0,f_W-f^\star_0\right\rangle_\mu\ge 0,
    \qquad \forall f_W\in\mathcal{F}_{B,m}.$$ By the variational characterization of Hilbert-space projection, this is equivalent to $$f^\star_0=\Pi_{\mathcal{F}_{B,m}}T^\pi f^\star_0.$$ Since the MSPBE is the squared distance to the projected Bellman image, any projected Bellman fixed point has MSPBE value zero relative to its projected equation and is a global minimizer of the corresponding projected Bellman residual objective. ◻

</div>

<div class="remark">

**Remark 3** (What $W^\star$ is and is not). *The point $W^\star$ is not introduced as a stationary point of the original nonlinear neural-network class. It is the projected Bellman point for the frozen random-feature class $\mathcal{F}_{B,m}$. The neural TD proof then shows that the actual neural population iterates approach $\widehat Q_0(\cdot;W^\star)$ in $L_2(\mu)$, up to finite-width errors.*

</div>

## Linearized neural TD is exactly linear TD

Before analyzing the actual neural update, it is useful to analyze the frozen update $$\label{eq:frozen-pop-update}
    W(t+1)=\Pi_{S_B} \left(W(t)-\eta\bar g_0(W(t))\right).$$ This is projected linear TD with feature map $\Phi_0$ and parameter $W$.

### One-point monotonicity in function norm

The most important identity is the following.

<div id="lem:frozen-monotonicity" class="lemma">

**Lemma 2** (Frozen one-point monotonicity). *Let $W,W^\star\in S_B$, and suppose $W^\star$ satisfies the displayed equation. Define $$h_W(x):=\widehat Q_0(x;W)-\widehat Q_0(x;W^\star).$$ Then $$\label{eq:frozen-monotonicity}
    \left\langle \bar g_0(W),W-W^\star\right\rangle
    \ge
    (1-\gamma) \left\lVert h_W \right\rVert_\mu^2.$$*

</div>

<div class="proof">

*Proof.* Decompose $$\label{eq:g0-decomp}
    \left\langle \bar g_0(W),W-W^\star\right\rangle
    =
    \left\langle \bar g_0(W)-\bar g_0(W^\star),W-W^\star\right\rangle
    +
    \left\langle \bar g_0(W^\star),W-W^\star\right\rangle.$$ The second term is nonnegative by the displayed equation. For the first term, using the linearity of $\widehat Q_0$, $$\begin{aligned}
    \left\langle \bar g_0(W)-\bar g_0(W^\star),W-W^\star\right\rangle
    &=
    \mathbb{E} \left[
    \left(h_W(x)-\gamma h_W(x')\right)
    \Phi_0(x)^\top(W-W^\star)
    \right] \\
    &=
    \mathbb{E} \left[
    \left(h_W(x)-\gamma h_W(x')\right)h_W(x)
    \right] \\
    &=
    \mathbb{E}[h_W(x)^2]-\gamma\mathbb{E}[h_W(x')h_W(x)].
\end{aligned}$$ By Assumption 1 and Cauchy–Schwarz, $$\mathbb{E}[h_W(x')h_W(x)]
    \le
    \left\lVert h_W \right\rVert_\mu^2.$$ Therefore $$\mathbb{E}[h_W(x)^2]-\gamma\mathbb{E}[h_W(x')h_W(x)]
    \ge
    (1-\gamma) \left\lVert h_W \right\rVert_\mu^2.$$ Combining this with the decomposition above proves the claim. ◻

</div>

<div class="remark">

**Remark 4** (This is the neural analogue of linear strong monotonicity). *The linear proof used $$\left\langle A(\theta-\theta^\star),\theta-\theta^\star\right\rangle
    \ge
    \mu_{\mathrm{lin}} \left\lVert \theta-\theta^\star \right\rVert_2^2.$$ The frozen neural proof uses $$\left\langle \bar g_0(W),W-W^\star\right\rangle
    \ge
    (1-\gamma) \left\lVert \widehat Q_0(\cdot;W)-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.$$ The latter is a function-space monotonicity statement. It controls prediction error, not necessarily raw Euclidean distance between weights. This distinction matters because the frozen random-feature map can be highly overparameterized and non-identifiable in parameter space.*

</div>

### A useful norm bound

The projection/telescoping proof also needs a bound on the size of the frozen semigradient. In the cleanest presentation, assume the following.

<div id="ass:frozen-gradient-bound" class="assumption">

**Assumption 6** (Frozen semigradient norm bound). *For every $W\in S_B$, $$\label{eq:frozen-gradient-bound}
    \left\lVert \bar g_0(W) \right\rVert_2^2
    \le
    G_0^2 \left\lVert \widehat Q_0(\cdot;W)-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.$$*

</div>

In many normalized neural TD presentations, the constant $G_0^2$ is an absolute constant. The paper’s proof effectively uses a bound of this type, with constants that lead to the coefficient $2\eta(1-\gamma)-8\eta^2$ in the population recursion.

### Population convergence of the frozen linearized update

<div id="lem:frozen-telescope" class="lemma">

**Lemma 3** (Frozen population TD telescoping inequality). *Suppose Lemma 2 holds and Assumption 6 holds. Let $$W(t+1)=\Pi_{S_B} \left(W(t)-\eta\bar g_0(W(t))\right).$$ If $0<\eta<2(1-\gamma)/G_0^2$, then $$\label{eq:frozen-one-step}
    \left\lVert W(t+1)-W^\star \right\rVert_2^2
    \le
    \left\lVert W(t)-W^\star \right\rVert_2^2
    -
    \left(2\eta(1-\gamma)-\eta^2G_0^2\right)
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.$$ Consequently, $$\label{eq:frozen-average}
    \frac1T\sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    \le
    \frac{\left\lVert W(0)-W^\star \right\rVert_2^2}
    {T\left(2\eta(1-\gamma)-\eta^2G_0^2\right)}.$$*

</div>

<div class="proof">

*Proof.* By nonexpansiveness of projection and $W^\star\in S_B$, $$\begin{aligned}
    \left\lVert W(t+1)-W^\star \right\rVert_2^2
    &\le
    \left\lVert W(t)-\eta\bar g_0(W(t))-W^\star \right\rVert_2^2 \\
    &=
    \left\lVert W(t)-W^\star \right\rVert_2^2
    -2\eta\left\langle \bar g_0(W(t)),W(t)-W^\star\right\rangle
    +\eta^2 \left\lVert \bar g_0(W(t)) \right\rVert_2^2.
\end{aligned}$$ Using Lemma 2 for the inner-product term and Assumption 6 for the squared-gradient term gives $$\begin{aligned}
    \left\lVert W(t+1)-W^\star \right\rVert_2^2
    &\le
    \left\lVert W(t)-W^\star \right\rVert_2^2 \\
    &\quad
    -2\eta(1-\gamma)
\left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2 \\
    &\quad
    +\eta^2G_0^2
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.
\end{aligned}$$ This proves the claimed one-step inequality. Summing the one-step inequality from $t=0$ to $T-1$ yields $$\begin{aligned}
    &\left(2\eta(1-\gamma)-\eta^2G_0^2\right)
    \sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2 \\
    &\qquad\le
    \left\lVert W(0)-W^\star \right\rVert_2^2-\left\lVert W(T)-W^\star \right\rVert_2^2
    \le
    \left\lVert W(0)-W^\star \right\rVert_2^2.
\end{aligned}$$ Dividing by $T(2\eta(1-\gamma)-\eta^2G_0^2)$ proves the averaged bound. ◻

</div>

This is exactly the same projection/telescoping skeleton as the actual neural population proof. The only missing issue is that the true neural update uses $\bar g_{\mathrm{NN}}$ and $\widehat Q$, not $\bar g_0$ and $\widehat Q_0$.

## From frozen linear TD to actual neural TD

### What must be proved beyond the linear proof

The actual population neural update is $$W(t+1)=\Pi_{S_B}(W(t)-\eta\bar g_{\mathrm{NN}}(W(t))),$$ where $$\bar g_{\mathrm{NN}}(W)=\mathbb{E}[\delta_W(x,r,x')\nabla_W\widehat Q(x;W)].$$ The frozen proof would be immediate if $$\bar g_{\mathrm{NN}}(W)=\bar g_0(W)
    \quad\text{and}\quad
    \widehat Q(x;W)=\widehat Q_0(x;W).$$ But this equality is false. The ReLU gates change with $W$. The neural proof therefore needs overparameterization to show that this mismatch is small.

The two essential statements are:

1.  **Local linearization**: for all iterates in $S_B$, $$\widehat Q(\cdot;W)
            \approx
            \widehat Q_0(\cdot;W),
            \qquad
            \bar g_{\mathrm{NN}}(W)
            \approx
            \bar g_0(W).$$

2.  **One-point monotonicity**: despite nonconvexity, the population neural semigradient points toward the frozen projected Bellman point $W^\star$ up to finite-width error.

### Finite-width error shorthand

To avoid hiding the structure under long ReLU concentration estimates, define $$\label{eq:neural-width-error}
    \Delta_{m,B}
    :=
    C\left(B^3m^{-1/2}+B^{5/2}m^{-1/4}\right),$$ where $C$ hides constants depending on regularity of the input distribution. The population theorem of Cai, Yang, Lee, and Wang (2019) has exactly this type of finite-width error term.

<div id="ass:neural-one-point" class="assumption">

**Assumption 7** (Population neural one-point inequality). *Along the projected neural population iterates $W(t)\in S_B$, the following inequality holds: $$\label{eq:neural-one-point}
    \left\langle \bar g_{\mathrm{NN}}(W(t)),W(t)-W^\star\right\rangle
    \ge
    (1-\gamma)
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    -
    \Delta_{m,B}.$$*

</div>

This is the nonlinear replacement for Lemma 2. In the exact frozen linear case, $\Delta_{m,B}=0$.

<div id="ass:neural-grad-bound" class="assumption">

**Assumption 8** (Population neural semigradient norm inequality). *Along the projected neural population iterates, $$\label{eq:neural-grad-bound}
    \left\lVert \bar g_{\mathrm{NN}}(W(t)) \right\rVert_2^2
    \le
    8
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    +
    \Delta_{m,B}.$$*

</div>

The numerical constant $8$ is chosen to match the common presentation of the population neural TD recursion, where the useful descent coefficient becomes $2\eta(1-\gamma)-8\eta^2$.

<div class="remark">

**Remark 5** (Where these assumptions come from). *These assumptions are not generic facts about arbitrary neural networks. They are consequences of overparameterization, random initialization, projection to $S_B$, and regularity of the input distribution. The proof strategy is:*

1.  *Show that few ReLU gates change when $W$ remains in $S_B$ and $m$ is large.*

2.  *Use this to show $\widehat Q(\cdot;W)$ is close to $\widehat Q_0(\cdot;W)$.*

3.  *Use this to show $\bar g_{\mathrm{NN}}(W)$ is close to $\bar g_0(W)$.*

4.  *Transfer the exact frozen monotonicity of Lemma 2 to the actual neural semigradient.*

*This is precisely the point at which projected neural TD differs from projected linear TD.*

</div>

## Population proof for projected neural TD

### One-step recursion

<div id="lem:neural-one-step" class="lemma">

**Lemma 4** (Population neural TD one-step inequality). *Suppose Assumption 7 and Assumption 8 hold. Then the population projected neural TD update above satisfies $$\begin{aligned}
\label{eq:neural-one-step-final}
    \left\lVert W(t+1)-W^\star \right\rVert_2^2
    &\le
    \left\lVert W(t)-W^\star \right\rVert_2^2 \\
    &\quad
    -
    \left(2\eta(1-\gamma)-8\eta^2\right)
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2 \\
    &\quad
    +
    (2\eta+\eta^2)\Delta_{m,B}.
\end{aligned}$$*

</div>

<div class="proof">

*Proof.* By nonexpansiveness of projection and $W^\star\in S_B$, $$\begin{aligned}
    \left\lVert W(t+1)-W^\star \right\rVert_2^2
    &\le
    \left\lVert W(t)-\eta\bar g_{\mathrm{NN}}(W(t))-W^\star \right\rVert_2^2 \\
    &=
    \left\lVert W(t)-W^\star \right\rVert_2^2
    -2\eta\left\langle \bar g_{\mathrm{NN}}(W(t)),W(t)-W^\star\right\rangle
    +\eta^2 \left\lVert \bar g_{\mathrm{NN}}(W(t)) \right\rVert_2^2.
\end{aligned}$$ Using Assumption 7, $$\begin{aligned}
    -2\eta\left\langle \bar g_{\mathrm{NN}}(W(t)),W(t)-W^\star\right\rangle
    &\le
    -2\eta(1-\gamma)
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    +2\eta\Delta_{m,B}.
\end{aligned}$$ Using Assumption 8, $$\begin{aligned}
    \eta^2 \left\lVert \bar g_{\mathrm{NN}}(W(t)) \right\rVert_2^2
    &\le
    8\eta^2
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    +\eta^2\Delta_{m,B}.
\end{aligned}$$ Combining the two bounds gives the displayed equation. ◻

</div>

### Telescoping

<div id="thm:neural-pop-modular" class="theorem">

**Theorem 2** (Population projected neural TD, modular form). *Suppose Assumption 7 and Assumption 8 hold and choose $$\label{eq:neural-eta-choice}
    \eta=\frac{1-\gamma}{8}.$$ Then $$\label{eq:neural-average-bound-modular}
    \frac1T\sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    \le
    \frac{8 \left\lVert W(0)-W^\star \right\rVert_2^2}{(1-\gamma)^2T}
    +
    C_\gamma\Delta_{m,B},$$ where $C_\gamma$ is a constant depending at most polynomially on $(1-\gamma)^{-1}$. Since $W^\star\in S_B$, this implies $$\label{eq:neural-average-bound-B}
    \frac1T\sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    \le
    \frac{8B^2}{(1-\gamma)^2T}
    +
    C_\gamma\Delta_{m,B}.$$*

</div>

<div class="proof">

*Proof.* From Lemma 4, summing from $t=0$ to $T-1$ gives $$\begin{aligned}
    &\left(2\eta(1-\gamma)-8\eta^2\right)
    \sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2 \\
    &\quad\le
    \left\lVert W(0)-W^\star \right\rVert_2^2-\left\lVert W(T)-W^\star \right\rVert_2^2
    +T(2\eta+\eta^2)\Delta_{m,B} \\
    &\quad\le
    \left\lVert W(0)-W^\star \right\rVert_2^2
    +T(2\eta+\eta^2)\Delta_{m,B}.
\end{aligned}$$ With $\eta=(1-\gamma)/8$, $$\begin{aligned}
    2\eta(1-\gamma)-8\eta^2
    &=
    2\cdot\frac{1-\gamma}{8}(1-\gamma)
    -8\cdot\frac{(1-\gamma)^2}{64} \\
    &=
    \frac{(1-\gamma)^2}{4}-\frac{(1-\gamma)^2}{8} \\
    &=
    \frac{(1-\gamma)^2}{8}.
\end{aligned}$$ Therefore $$\begin{aligned}
    \frac1T\sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    &\le
    \frac{8 \left\lVert W(0)-W^\star \right\rVert_2^2}{(1-\gamma)^2T} \\
    &\quad
    +
    \frac{8(2\eta+\eta^2)}{(1-\gamma)^2}\Delta_{m,B}.
\end{aligned}$$ This proves the claimed one-step inequality. Since $W^\star\in S_B$ and $W(0)$ is the center of $S_B$, $\left\lVert W(0)-W^\star \right\rVert_2\le B$, giving the stated $B$-radius bound. ◻

</div>

### From averaged frozen prediction error to output error

The neural TD paper states the theorem for an output function $\widehat Q_{\mathrm{out}}$. If the output is formed by averaging the functions, i.e., $$\label{eq:Qout-average}
    \widehat Q_{0,\mathrm{out}}(x)
    :=
    \frac1T\sum_{t=0}^{T-1}\widehat Q_0(x;W(t)),$$ then Jensen’s inequality gives $$\begin{aligned}
    \left\lVert \widehat Q_{0,\mathrm{out}}-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    &=
    \left\lVert \frac 1T\sum_{t=0}^{T-1}
    \left(\widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star)\right) \right\rVert_\mu^2 \\
    &\le
    \frac1T\sum_{t=0}^{T-1}
    \left\lVert \widehat Q_0(\cdot;W(t))-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.
\end{aligned}$$ Combining this with Theorem 2 yields $$\label{eq:Qout-frozen-bound}
    \left\lVert \widehat Q_{0,\mathrm{out}}-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    \le
    \frac{8B^2}{(1-\gamma)^2T}+C_\gamma\Delta_{m,B}.$$ The actual paper states the result for the actual neural output $\widehat Q_{\mathrm{out}}$, not merely the frozen output. The additional step is another local-linearization estimate: $$\label{eq:actual-output-vs-frozen-output}
    \left\lVert \widehat Q_{\mathrm{out}}-\widehat Q_{0,\mathrm{out}} \right\rVert_\mu^2
    \le
    C\Delta_{m,B}.$$ Hence $$\label{eq:neural-final-style}
    \boxed{
    \left\lVert \widehat Q_{\mathrm{out}}-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    \le
    \frac{C B^2}{(1-\gamma)^2T}
    +
    C'\left(B^3m^{-1/2}+B^{5/2}m^{-1/4}\right).
    }$$ With the constants in Cai, Yang, Lee, and Wang (2019), the leading population term is written as $$\label{eq:neural-paper-style}
    \mathbb{E}_{\mathrm{init},\mu}
    \left[
    \left(\widehat Q_{\mathrm{out}}(x)-\widehat Q_0(x;W^\star)\right)^2
    \right]
    \le
    \frac{16B^2}{(1-\gamma)^2T}
    +
    O\left(B^3m^{-1/2}+B^{5/2}m^{-1/4}\right).$$ The expectation over initialization appears because $\Phi_0$, $\mathcal{F}_{B,m}$, and $W^\star$ are random objects induced by $W(0)$.

## Exact side-by-side proof comparison

### The class $\mathcal{F}$

|                                | Linear TD                                                                   | Neural TD                                                                              |
| :----------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| Generic MSPBE projection class | $\mathcal{F}=\mathcal{F}_{\mathrm{lin}}$                                    | $\mathcal{F}=\mathcal{F}_{B,m}$ in the proof                                           |
| Class definition               | $\mathcal{F}_{\mathrm{lin}}=\{x\mapsto\phi(x)^\top\theta:\theta\in\Theta\}$ | $\mathcal{F}_{B,m}=\{x\mapsto\Phi_0(x)^\top W:W\in S_B\}$                              |
| Who chooses features?          | User/designer chooses $\phi$                                                | Random initialization chooses $\Phi_0$ through ReLU gates                              |
| Is the class linear?           | Yes                                                                         | Yes after freezing gates                                                               |
| Is the actual model linear?    | Yes                                                                         | No; actual $\widehat Q(x;W)$ is nonlinear                                              |
| Why projection is easy         | Projection onto a fixed convex parameter set                                | Projection keeps network near initialization so frozen class remains accurate          |
| Fixed point                    | $Q_{\theta^\star}=\Pi_{\mathcal{F}_{\mathrm{lin}}}T^\pi Q_{\theta^\star}$   | $\widehat Q_0(\cdot;W^\star)=\Pi_{\mathcal{F}_{B,m}}T^\pi \widehat Q_0(\cdot;W^\star)$ |

</div>

### Population direction

Linear TD: $$\bar g_{\mathrm{lin}}(\theta)
    =
    \mathbb{E}[(Q_\theta(x)-r-\gamma Q_\theta(x'))\phi(x)]
    =A\theta-b.$$

Frozen neural TD: $$\bar g_0(W)
    =
    \mathbb{E}[(\widehat Q_0(x;W)-r-\gamma\widehat Q_0(x';W))\Phi_0(x)]
    =A_0W-b_0.$$

Actual neural TD: $$\bar g_{\mathrm{NN}}(W)
    =
    \mathbb{E}[(\widehat Q(x;W)-r-\gamma\widehat Q(x';W))\nabla_W\widehat Q(x;W)].$$ The third object is nonlinear and is not equal to $A_0W-b_0$. The neural proof works by proving it is close enough to the frozen object inside $S_B$.

### One-point inequalities

Linear TD uses parameter-space strong monotonicity: $$\left\langle \bar g_{\mathrm{lin}}(\theta),\theta-\theta^\star\right\rangle
    \ge
    \mu_{\mathrm{lin}} \left\lVert \theta-\theta^\star \right\rVert_2^2.$$

Frozen neural TD uses function-space monotonicity: $$\left\langle \bar g_0(W),W-W^\star\right\rangle
    \ge
    (1-\gamma)
    \left\lVert \widehat Q_0(\cdot;W)-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2.$$

Actual neural TD uses approximate function-space monotonicity: $$\left\langle \bar g_{\mathrm{NN}}(W),W-W^\star\right\rangle
    \ge
    (1-\gamma)
    \left\lVert \widehat Q_0(\cdot;W)-\widehat Q_0(\cdot;W^\star) \right\rVert_\mu^2
    -
    \Delta_{m,B}.$$

### Final population rates

For projected population linear TD, under global strong monotonicity, $$\left\lVert \theta_T-\theta^\star \right\rVert_2^2
    \le
    (1-\eta\mu_{\mathrm{lin}})^T
    \left\lVert \theta_0-\theta^\star \right\rVert_2^2.$$ This is exponential because the population operator is a deterministic contraction in parameter space.

For projected population neural TD, the theorem is usually stated in averaged function error: $$\mathbb{E}_{\mathrm{init},\mu}
    \left[
    (\widehat Q_{\mathrm{out}}(x)-\widehat Q_0(x;W^\star))^2
    \right]
    \le
    \frac{16B^2}{(1-\gamma)^2T}
    +
    O\left(B^3m^{-1/2}+B^{5/2}m^{-1/4}\right).$$ This is an $O(1/T)$ averaged population rate plus finite-width error. It is not a simple exponential parameter contraction because the proof controls prediction error relative to the frozen random-feature projected Bellman point, not Euclidean distance to a unique neural parameter.

## Why the two population results look different

### Linear TD has a fixed finite-dimensional geometry

In linear TD, once $\phi$ is chosen, the geometry is fixed. The matrix $$A=\mathbb{E}[\phi(x)(\phi(x)-\gamma\phi(x'))^\top]$$ determines everything. If $A$ is strongly monotone, the population update contracts. Projection is only a stability device.

### Neural TD creates a random feature class

In neural TD, the actual features are the tangent features $$\nabla_W\widehat Q(x;W).$$ These depend on $W$, so they move during training. The proof freezes them at initialization: $$\Phi_0(x)=\nabla_W\widehat Q_0(x;W)
    =
    \frac1{\sqrt m}
    \begin{bmatrix}
    b_1\mathbf{1}\{W_1(0)^\top x>0\}x\\
    \vdots\\
    b_m\mathbf{1}\{W_m(0)^\top x>0\}x
    \end{bmatrix}.$$ The class $\mathcal{F}_{B,m}$ is therefore a random feature class generated by initialization.

### Projection plays a different conceptual role

In projected linear TD, projection keeps $\theta_t$ in $\Theta$. Since the operator is already globally affine and strongly monotone, projection is not conceptually responsible for linearity.

In projected neural TD, projection keeps $W(t)$ close to $W(0)$. This is essential because closeness to initialization is what makes the ReLU gates stable and validates the approximation $$\widehat Q(x;W(t))\approx \widehat Q_0(x;W(t)).$$ Thus, in neural TD, projection is part of the mechanism that makes the nonlinear system behave like a linear random-feature TD method.

## A compact “mental model”

The cleanest way to remember the comparison is: $$\boxed{
    \text{Linear TD is projected TD in a fixed feature class }\mathcal{F}_{\mathrm{lin}}.
    }$$ $$\boxed{
    \text{Neural TD is approximately projected TD in a random frozen feature class }\mathcal{F}_{B,m}.
    }$$ The full neural proof is the proof that “approximately” is true when the network is sufficiently wide.

## Bibliographic notes

Mitra Mitra (2024) studies linear TD with linear function approximation under Markovian sampling and emphasizes the role of projection-based analyses as a simpler baseline. The note here uses the easier population/i.i.d. projected version as the clean linear reference point.

Cai, Yang, Lee, and Wang Cai, Yang, Lee, and Wang (2019) prove that neural TD converges to the global optimum of MSPBE under overparameterization. Their population theorem gives an $O(1/T)$ rate plus finite-width error, and the key structural object is the frozen random-feature class $\mathcal{F}_{B,m}$ induced by the initialized ReLU gates.


## References

- Qi Cai, Zhuoran Yang, Jason D. Lee, and Zhaoran Wang. Neural Temporal-Difference Learning Converges to Global Optima. *Advances in Neural Information Processing Systems*, 32, 2019. <https://arxiv.org/abs/1905.10027>.
- Aritra Mitra. A Simple Finite-Time Analysis of TD Learning with Linear Function Approximation. arXiv:2403.02476, 2024. <https://arxiv.org/abs/2403.02476>.
- Jalaj Bhandari, Daniel Russo, and Raghav Singal. A finite time analysis of temporal difference learning with linear function approximation. *Conference on Learning Theory*, 2018.
- John N. Tsitsiklis and Benjamin Van Roy. An analysis of temporal-difference learning with function approximation. *IEEE Transactions on Automatic Control*, 42(5):674–690, 1997.
