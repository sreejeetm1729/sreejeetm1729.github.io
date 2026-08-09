---
title: "Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates"
date: 2026-08-08 
categories: [rl-blogs]
rl_section: robust-rl
tags: [q-learning, robust reinforcement learning, asynchronous sampling, markovian data, heavy-tailed rewards]
math: true
description: "How robust reward estimation, adaptive thresholding, and refined concentration yield near-optimal asynchronous Q-Learning under adversarial corruption."
---

Real $Q$-Learning does not receive a fresh sample for every state-action pair at every iteration. It follows a behavior policy, observes one transition, and updates only the coordinate that was visited. The data are asynchronous, unevenly distributed across state-action pairs, and correlated along a Markov trajectory. If the observed rewards can also be heavy-tailed and adversarially corrupted, even defining a stable empirical Bellman update becomes a nontrivial problem.

The paper *Corruption-Tolerant Asynchronous Q-Learning with Near-Optimal Rates* by Sreejeet Maity and Aritra Mitra develops a robust $Q$-Learning framework for precisely this setting. Its contribution is not merely to show convergence. It establishes a high-probability finite-time rate, proves a nearly matching information-theoretic lower bound, removes the need to know the reward moments, and extends the result to single-trajectory Markovian data.

## The asynchronous learning problem

Consider a discounted finite Markov decision process

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma),
\qquad \gamma\in(0,1).
$$

The clean reward distribution for $(s,a)$ has mean $R(s,a)$ and variance $\sigma^2(s,a)$. The only tail assumption is a finite second moment:

$$
|R(s,a)|\leq \overline R,
\qquad
\sigma^2(s,a)\leq \overline\sigma^2.
$$

A stochastic behavior policy $\mu$ generates the data. If $\pi$ denotes the stationary state distribution under $\mu$, then the stationary visitation probability of $(s,a)$ is

$$
\lambda(s,a)=\pi(s)\mu(a\mid s),
$$

and the least-covered coordinate is measured by

$$
\lambda_{\min}
=
\min_{(s,a)\in\mathcal S\times\mathcal A}
\lambda(s,a).
$$

This quantity is central in asynchronous learning. A coordinate with visitation probability $\lambda(s,a)$ receives only about $\lambda(s,a)T$ observations during $T$ iterations. Any uniform error guarantee must pay for the least frequently observed coordinate.

## Huber contamination of the rewards

At time $t$, the learner observes $(s_t,a_t,s_{t+1})$, but the reward is

$$
y_t(s_t,a_t)
=
(1-Y_t)r_t(s_t,a_t)+Y_t z_t,
$$

where

$$
Y_t\sim\operatorname{Bernoulli}(\varepsilon).
$$

When $Y_t=0$, the reward is a clean finite-variance sample. When $Y_t=1$, the adversarial value $z_t$ can be arbitrary, unbounded, history dependent, and state-action dependent. The indicator sequence is independent of the past, so the adversary controls what value is injected but not the realization of whether the current sample is corrupted.

The goal is to estimate the optimal $Q$-function, the fixed point of

$$
(\mathcal TQ)(s,a)
=
R(s,a)
+
\gamma\,
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q(s',a')
\right],
$$

using only the corrupted asynchronous trajectory.

## Why the asynchronous setting is harder

There are three interacting difficulties. First, the number of observations available for a given $(s,a)$ is random. A robust mean theorem stated for a deterministic sample size cannot be applied blindly to a randomly sized reward history. Second, the $Q$-update uses the current random iterate inside a bootstrap target, coupling transition noise with all previous corruptions. Third, high-probability robust estimators can fail on rare events, and one such failure can destabilize the recursive learning dynamics.

These issues explain why replacing the current reward by a trimmed mean is not, by itself, a complete algorithm. The estimator must be coupled to a visitation analysis and a deterministic stability mechanism.

## Robust reward histories

For each state-action pair, the algorithm maintains a dynamic reward data set

$$
\mathcal D_t(s,a)
=
\{y_k(s_k,a_k):(s_k,a_k)=(s,a),\ 0\leq k\leq t\}.
$$

Whenever $(s_t,a_t)$ is visited, a robust trimmed-mean estimator is applied to $\mathcal D_t(s_t,a_t)$. If $M$ samples of a scalar random variable with variance at most $\sigma^2$ are Huber contaminated, the estimator satisfies

$$
|\widehat\mu-\mu|
\leq
C\sigma
\left(
\sqrt\varepsilon
+
\sqrt{\frac{\log(1/\delta)}{M}}
\right)
$$

with high probability. Thus, after $(s,a)$ has been visited about $\lambda(s,a)t$ times, its reward-estimation error should behave like

$$
\overline\sigma
\left(
\sqrt\varepsilon
+
\sqrt{
\frac{\log(1/\delta)}{\lambda(s,a)t}
}
\right).
$$

The paper turns this heuristic into a uniform statement by using Bernstein's inequality to show that, after a burn-in period, every visitation count is bounded below by a constant fraction of its expectation.

## Adaptive thresholding

Let $\overline r_t(s_t,a_t)$ denote the trimmed-mean reward estimate. The algorithm defines a burn-in time of order

$$
\overline T
\asymp
\frac{1}{\lambda_{\min}}
\log\left(\frac{|\mathcal S||\mathcal A|T}{\delta}\right)
$$

and an adaptive threshold

$$
G_t
=
\begin{cases}
0,
& t\leq\overline T,\\[4pt]
C\overline\sigma
\left(
\sqrt{
\dfrac{\log(1/\delta_1)}{\lambda_{\min}t}
}
+
\sqrt\varepsilon
\right)
+
\widetilde\sigma,
& t>\overline T,
\end{cases}
$$

where

$$
\widetilde\sigma=\max\{\overline R,\overline\sigma\}.
$$

If $|\overline r_t(s_t,a_t)|>G_t$, the estimate is rejected and replaced by zero. Otherwise it is accepted. Denote the resulting reward proxy by $\widetilde r_t(s_t,a_t)$.

The threshold is carefully balanced. It is small enough to prevent a rare catastrophic estimator output from entering the recursion, but large enough that, after burn-in, an accurate robust estimate is accepted on the main high-probability event. The proof shows that thresholding eventually becomes inactive on this good event.

## The Robust Async-Q update

Only the visited state-action coordinate is updated:

$$
Q_{t+1}(s,a)
=
\begin{cases}
Q_t(s,a),
& (s,a)\neq(s_t,a_t),\\[4pt]
(1-\alpha)Q_t(s_t,a_t)
+
\alpha
\left[
\widetilde r_t(s_t,a_t)
+
\gamma\max_{a'}Q_t(s_{t+1},a')
\right],
& (s,a)=(s_t,a_t).
\end{cases}
$$

This is recognizably $Q$-Learning, but its reward input is a robust historical estimate rather than the newest observed reward. The algorithm assumes knowledge of $\varepsilon$ and $\lambda_{\min}$ for tuning. Exact knowledge of $\varepsilon$ is not essential: any valid upper bound can be substituted, with the theorem then depending on that upper bound.

## Finite-time rate with known reward scales

Let

$$
d_t=Q_t-Q^\star.
$$

With a constant step size of order

$$
\alpha
\asymp
\frac{\log T}{\lambda_{\min}(1-\gamma)T},
$$

the main i.i.d.-sampling theorem gives, with high probability,

$$
\|d_T\|_\infty
\leq
\frac{\|d_0\|_\infty}{T}
+
\widetilde O\left(
\frac{\widetilde\sigma}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}\sqrt T}
\right)
+
O\left(
\frac{\overline\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

The suppressed logarithms depend on $T$, $|\mathcal S||\mathcal A|$, and $1/\delta$. When $\varepsilon=0$, the dominant statistical term recovers the known $T^{-1/2}$ behavior of asynchronous $Q$-Learning, up to logarithmic and problem-dependent factors. Under corruption, the attack magnitude disappears entirely; only $\sqrt\varepsilon$ remains.

The factor $1/\lambda_{\min}$ in the corruption term is also intuitive. If a state-action pair is rarely visited, the learner has fewer clean rewards for that pair, so an $\varepsilon$ fraction of corruption is harder to distinguish from the tails of the clean distribution.

## A fundamental lower bound

Could a different algorithm eliminate the $\sqrt\varepsilon$ term? The paper answers no. It proves that even under a more favorable synchronous observation model, there is a universal constant $c>0$ such that

$$
\inf_{\widehat Q_T}
\sup_{Q^\star\in\mathfrak H(\varepsilon,\overline\sigma,\mathcal Q)}
\mathbb P\left(
\|\widehat Q_T-Q^\star\|_\infty
\geq
\frac{c\overline\sigma\sqrt\varepsilon}{1-\gamma}
\right)
\geq
\frac14.
$$

The construction uses two MDPs with optimal $Q$-functions separated by order $\overline\sigma\sqrt\varepsilon/(1-\gamma)$ and attack distributions that make their corrupted observations statistically indistinguishable. Therefore, no estimator can be accurate on both instances.

The upper and lower bounds match in their dependence on $\varepsilon$, $\overline\sigma$, and $1-gamma$. The remaining gap concerns asynchronous coverage factors and logarithms, not the fundamental corruption scaling.

## A useful noise-free exception

The lower bound scales with the clean reward variance. If rewards are deterministic, the situation changes. Every clean observation of $(s,a)$ equals exactly $R(s,a)$. Since $\varepsilon<1/2$, clean observations are eventually in the majority. A simple median can then recover each mean reward exactly after sufficiently many visits.

Once the true reward table is recovered, the algorithm evolves like ordinary $Q$-Learning and pays no corruption floor. This does not contradict the lower bound: when $\overline\sigma=0$, the lower-bound separation is zero. The example clarifies that the unavoidable $\sqrt\varepsilon$ term comes from the statistical ambiguity between heavy-tailed clean samples and adversarial outliers, not from corruption in isolation.

## Removing knowledge of reward moments

The threshold above uses $\widetilde\sigma$, an upper bound on the reward mean and standard deviation. The reward-agnostic version, called Robust Async-RAQ, replaces this unknown constant by a slowly growing proxy

$$
m(t)=t^p,
\qquad p\in\{1,2,\ldots\}.
$$

The new threshold is obtained by substituting $m(t)$ for $\widetilde\sigma$. Since $m(t)$ eventually exceeds every fixed reward scale, the threshold ultimately contains the typical reward estimates even though the learner does not know when this domination first occurs.

This modification creates a serious proof problem. Deterministically, the threshold and hence the iterates can now grow like $T^p$. A standard Azuma-Hoeffding bound would insert this crude worst-case scale into every martingale increment and produce a vacuous result. Yet with high probability the iterates remain of constant order.

The analysis resolves this tension using a refined inequality of Shamir and Spencer for martingales whose increments admit both a coarse deterministic bound and a much smaller high-probability bound. This concentration result allows the proof to exploit the typical behavior without ignoring the rare worst case.

For any fixed positive integer $p$, the reward-agnostic theorem yields

$$
\|d_T\|_\infty
\leq
\frac{\|d_0\|_\infty}{T}
+
\widetilde O\left(
\frac{\widetilde\sigma^{1+1/(2p)}}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}\sqrt T}
\right)
+
O\left(
\frac{\overline\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

Thus the learner can remain agnostic to the reward distribution while preserving the same $T^{-1/2}$ and $\sqrt\varepsilon$ scalings, with only a mild change in the reward-scale dependence.

## Extension to a single Markov trajectory

The preceding discussion initially samples $s_t$ independently from stationarity to isolate the asynchronous and corruption difficulties. The paper then returns to the actual single-trajectory model. The augmented process

$$
Z_t=(s_t,a_t,s_{t+1})
$$

is itself an ergodic Markov chain. Let $\overline\tau$ denote its mixing time. The algorithm retains one sample every

$$
\tau
\asymp
\overline\tau\log(T/\delta)
$$

steps and discards the intervening samples. This blocking construction produces a nearly independent subsequence to which the robust analysis can be coupled.

The Markovian theorem has the schematic form

$$
\|d_T\|_\infty
\leq
\frac{\|d_0\|_\infty}{T}
+
\widetilde O\left(
\frac{\widetilde\sigma^{1+1/(2p)}}
{\lambda_{\min}^{3/2}(1-\gamma)^{5/2}}
\sqrt{\frac{\tau}{T}}
\right)
+
O\left(
\frac{\overline\sigma\sqrt\varepsilon}
{\lambda_{\min}(1-\gamma)}
\right).
$$

The statistical term is inflated by approximately $\sqrt\tau$, reflecting an effective sample size of $T/\tau$. The corruption floor is unchanged. This mirrors the way mixing affects vanilla $Q$-Learning and shows that robustness does not introduce an additional asymptotic penalty beyond the expected loss of effective samples.

## Proof architecture

The proof begins with the error recursion for asynchronous stochastic approximation. It separates a transition-sampling noise term from the robust reward-estimation error. These terms are coupled because the transition noise contains $Q_t$, while $Q_t$ depends on every previous corrupted reward estimate.

The first key step is a uniform visitation event. Bernstein's inequality guarantees that, after burn-in, every state-action pair has accumulated enough samples for trimmed-mean estimation. Conditioning on the realized visitation count then permits a careful application of the robust mean theorem despite the random history length.

The second key step is deterministic boundedness through thresholding. This makes the transition martingale increments controllable on every sample path. The third step is a high-probability event on which all accepted reward proxies are close to the true reward means. Finally, the asynchronous Bellman recursion is unrolled, and contraction converts the one-step perturbation bounds into the final infinity-norm guarantee.

For Markovian data, a coupling argument compares the blocked trajectory with an independent reference sequence. For reward-agnostic thresholds, the refined almost-martingale inequality replaces the standard bounded-increment argument.

## The broader lesson

The paper shows that robust $Q$-Learning is not obtained from a single robust estimator. It requires a complete system: historical filtering to suppress arbitrary reward magnitudes, visitation analysis to validate the estimator under asynchronous data, adaptive thresholding to stabilize rare events, and concentration tools matched to the actual dependence structure.

The final picture is sharp. Heavy-tailed reward noise and Huber contamination impose a $\sqrt\varepsilon$ error floor; asynchronous coverage introduces $\lambda_{\min}$; Markovian dependence reduces the effective sample size through the mixing time; and none of these effects prevents the ordinary $T^{-1/2}$ statistical decay from being recovered up to the appropriate problem-dependent factors.
