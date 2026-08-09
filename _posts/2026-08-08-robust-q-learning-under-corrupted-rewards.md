---
title: "Robust Q-Learning under Corrupted Rewards"
date: 2026-08-08 
categories: [rl-blogs]
rl_section: robust-rl
tags: [robust reinforcement learning, q-learning, adversarial corruption, robust statistics]
math: true
description: "Why ordinary Q-Learning can fail under a small number of corrupted rewards, and how robust mean estimation restores a near-optimal finite-time guarantee."
---

Reward observations are the feedback through which a reinforcement-learning agent learns which decisions are useful. This makes rewards an especially attractive target for an adversary. A single corrupted reward can be arbitrarily large, and a persistent but small fraction of such observations can bias every later decision because $Q$-Learning repeatedly bootstraps from its own estimates. The central question of this post is therefore not whether corruption adds some extra noise. It is whether learning remains statistically meaningful when an adversary is allowed to replace a small fraction of the rewards by arbitrary values.

This post develops the main ideas in the paper *Robust Q-Learning under Corrupted Rewards* by Sreejeet Maity and Aritra Mitra. The paper first shows that vanilla synchronous $Q$-Learning can be driven arbitrarily far from the correct solution. It then combines historical reward data, a robust trimmed-mean estimator, and a carefully designed threshold to recover the usual $T^{-1/2}$ statistical rate up to an unavoidable corruption-dependent error.

## The Bellman fixed-point viewpoint

Consider a discounted Markov decision process

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma),
\qquad \gamma\in(0,1).
$$

When action $a$ is selected in state $s$, the learner observes a next state $s'\sim P(\cdot\mid s,a)$ and a random reward $r(s,a)$ with mean

$$
R(s,a)=\mathbb E[r(s,a)].
$$

For a policy $\pi$, the state-action value is the expected discounted return

$$
Q^\pi(s,a)
=
\mathbb E\left[
\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)
\,middle|\,
(s_0,a_0)=(s,a),\pi
\right].
$$

The optimal $Q$-function $Q^\star$ is the unique fixed point of the Bellman optimality operator

$$
(\mathcal T^\star Q)(s,a)
=
R(s,a)
+
\gamma\,
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'\in\mathcal A}Q(s',a')
\right].
$$

The essential analytical fact is that this operator is a contraction in the infinity norm:

$$
\|\mathcal T^\star Q_1-\mathcal T^\star Q_2\|_\infty
\leq
\gamma\|Q_1-Q_2\|_\infty.
$$

If the model were known, repeatedly applying $\mathcal T^\star$ would converge geometrically to $Q^\star$. In reinforcement learning the model is unknown, so $Q$-Learning replaces the exact operator with a random empirical operator constructed from sampled rewards and transitions.

## Synchronous $Q$-Learning and the attack model

The paper studies synchronous sampling. At iteration $t$, a generative model supplies an independent next-state sample and reward for every state-action pair. Vanilla synchronous $Q$-Learning performs

$$
Q_{t+1}(s,a)
=
(1-\alpha_t)Q_t(s,a)
+
\alpha_t
\left[
r_t(s,a)
+
\gamma\max_{a'}Q_t(s_t(s,a),a')
\right].
$$

The benign reward $r_t(s,a)$ is not necessarily observed. Instead, the learner receives $y_t(s,a)$, and an omniscient adversary may replace the reward set at an $\varepsilon$-fraction of the first $t$ iterations. The replacement values can be arbitrary and unbounded. This is a sequential version of the strong-contamination model from robust statistics.

The distinction between ordinary stochastic noise and adversarial corruption is important. A zero-mean noise term tends to cancel when averaged. An adversary instead chooses the sign, magnitude, and timing of its perturbations to create a systematic bias. Increasing the number of samples does not automatically eliminate such a bias if the learner keeps using the ordinary empirical mean.

## Why vanilla $Q$-Learning can fail completely

To see the basic mechanism, first consider the slightly weaker Huber model. At each iteration, with probability $1-\varepsilon$, the reward is drawn from the true distribution; with probability $\varepsilon$, it is drawn from an adversarial distribution. The mean reward seen by the learner becomes

$$
\widetilde R_c(s,a)
=
(1-\varepsilon)R(s,a)
+
\varepsilon C(s,a),
$$

where $C(s,a)$ is the mean of the adversarial component. Vanilla $Q$-Learning therefore does not track the original Bellman operator. It tracks the perturbed operator

$$
(\widetilde{\mathcal T}_c^\star Q)(s,a)
=
\widetilde R_c(s,a)
+
\gamma\,
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q(s',a')
\right].
$$

Under the usual stochastic-approximation step-size conditions, the iterates converge almost surely to the fixed point $\widetilde Q_c^\star$ of this corrupted operator, not to $Q^\star$. Convergence by itself is therefore not a robustness guarantee. The algorithm may converge perfectly to the wrong object.

The paper gives a finite MDP in which the attacker corrupts rewards only at one state. By choosing a corruption magnitude proportional to $1/\varepsilon$, the attacker makes

$$
\|\widetilde Q_c^\star-Q^\star\|_\infty
$$

arbitrarily large, even when $\varepsilon$ is very small. The attack can also reverse the identity of the optimal action. This example exposes a general principle: a bound on the *fraction* of corruptions provides no protection to an estimator whose output can scale linearly with the *magnitude* of an outlier.

## Robust reward estimation from history

The defense begins by refusing to use the newest observed reward directly. For every $(s,a)$, the learner retains the history

$$
\mathcal D_t(s,a)=\{y_k(s,a):0\leq k\leq t\}
$$

and estimates $R(s,a)$ robustly from this entire collection. The estimator used in the paper is a split-sample trimmed mean. The data are divided into two halves. Empirical lower and upper quantiles are computed from the first half. Samples in the second half are clipped to the resulting interval, and the clipped samples are averaged.

For a clean scalar random variable $Z$ with variance $\sigma_Z^2$, the robust estimate $\widehat\mu_Z$ satisfies, with high probability,

$$
|\widehat\mu_Z-\mu_Z|
\leq
C\sigma_Z
\left(
\sqrt{\varepsilon}
+
\sqrt{\frac{\log(1/\delta)}{M}}
\right),
$$

provided the contamination fraction is sufficiently small and $M$ is large enough. The two terms have different meanings. The second is the usual statistical error and decreases with the sample size. The first is a contamination floor: with only a finite-variance assumption, an adversary can hide corruptions in the tails of the clean distribution, making a $\sqrt\varepsilon$ error unavoidable in general.

## Why robust estimation alone is not enough

The trimmed-mean guarantee is probabilistic. On a rare failure event, its output may still be extreme. Such an event cannot simply be ignored in a recursive algorithm because one extreme update can make $Q_t$ enormous, and this enormous iterate then appears inside future bootstrap targets.

The algorithm therefore applies a second safety mechanism. Let $\widetilde r_t(s,a)$ be the trimmed-mean estimate. It is clipped using a time-dependent threshold $G_t$. During an initial period, the threshold uses the known reward bound $\bar r$. Once enough observations have accumulated, $G_t$ is chosen using the high-probability radius of the robust estimator. Schematically,

$$
G_t
\asymp
\bar r
+
\bar r
\left(
\sqrt{\frac{\log(1/\delta_1)}{t}}
+
\sqrt\varepsilon
\right).
$$

If the robust estimate lies outside $[-G_t,G_t]$, it is truncated to the boundary. This design has two simultaneous effects. It bounds the iterates deterministically, including on rare estimator-failure events, and it remains loose enough that a statistically valid estimate is not altered after the burn-in period.

The robust $Q$-update is then

$$
Q_{t+1}(s,a)
=
(1-\alpha)Q_t(s,a)
+
\alpha
\left[
\widetilde r_t(s,a)
+
\gamma\max_{a'}Q_t(s_t(s,a),a')
\right].
$$

This resembles ordinary $Q$-Learning, but the immediate reward is now a robust estimate built from history and protected by thresholding.

## The finite-time guarantee

Let

$$
d_t=\|Q_t-Q^\star\|_\infty.
$$

For a suitable constant step size of order $\log(T)/((1-\gamma)T)$, the main theorem gives, up to logarithmic factors,

$$
d_T
\leq
\frac{d_0}{T}
+
\widetilde O\left(
\frac{\bar r}{(1-\gamma)^{5/2}\sqrt T}
\right)
+
O\left(
\frac{\bar r\sqrt\varepsilon}{1-\gamma}
\right)
$$

with high probability. The theorem assumes $\varepsilon<1/16$, a condition inherited from the particular robust mean estimator used by the algorithm.

The first term is the contracting memory of the initialization. The second is the ordinary statistical term; when $\varepsilon=0$, it matches the state-of-the-art $T^{-1/2}$ rate for synchronous $Q$-Learning up to logarithmic factors. The final term is the price of adversarial reward corruption. Crucially, it depends on the corruption fraction but not on the attack magnitude.

The factor $1/(1-\gamma)$ in the corruption term has a direct dynamic interpretation. A persistent reward bias of size $b$ is accumulated through the discounted horizon as

$$
b+\gamma b+\gamma^2b+\cdots
=
\frac{b}{1-\gamma}.
$$

Thus, even a robustly controlled one-step reward error is amplified by the effective planning horizon.

## Anatomy of the proof

The proof separates the random transition error from the adversarial reward-estimation error. Writing $E_t$ for the reward error and $D_t$ for the centered transition-sampling error, the update admits the decomposition

$$
Q_{t+1}
=
(1-\alpha)Q_t
+
\alpha\left(\mathcal T^\star Q_t+E_t+D_t\right).
$$

Subtracting $Q^\star=\mathcal T^\star Q^\star$, applying the Bellman contraction, and unrolling the recursion yields three components: a decaying initial error, a geometrically weighted sum of the transition noise, and a geometrically weighted sum of the robust reward errors.

The transition term is handled using martingale concentration after deterministic boundedness of the iterates has been established. The adversarial term is controlled by splitting time into two regimes. Before enough reward samples have accumulated, the algorithm relies on its deterministic threshold. After that point, the robust mean estimate is accurate simultaneously for all state-action pairs and all relevant iterations on a high-probability event. A union bound makes this event uniform, while the geometric weights ensure that the accumulated perturbation remains controlled.

The proof strategy is instructive beyond this particular algorithm. Robust statistics controls individual estimates; contraction controls how estimation errors propagate through dynamic programming; and deterministic safeguards control the rare events that high-probability statistics alone cannot rule out.

## Why a corruption floor is unavoidable

The paper gives a simple reduction to robust mean estimation. Consider an MDP with one state and one action. Learning $Q^\star$ is then equivalent to estimating the mean reward, followed by multiplication by $1/(1-\gamma)$. Under finite variance and adversarial contamination, robust mean estimation has a fundamental $\sqrt\varepsilon$ error. Therefore no algorithm can generally remove the term

$$
\frac{\sqrt\varepsilon}{1-\gamma}
$$

without additional assumptions on the clean reward distribution or the attack model.

This is the right way to interpret the theorem. The algorithm does not promise exact recovery under noisy finite-variance rewards. It promises that arbitrary attack magnitudes are reduced to the smallest corruption dependence one should expect from the underlying statistical problem.

## What this paper teaches us

The first lesson is that standard convergence theory can conceal extreme fragility. Vanilla $Q$-Learning still converges under the Huber mixture, but it converges to the fixed point of the wrong Bellman operator. The second lesson is that robust mean estimation must be integrated into the dynamics of the learning algorithm. Dropping a robust estimator into an update rule without controlling rare failures is not enough. The third lesson is that the Bellman contraction provides a clean interface between statistics and sequential decision-making: once the empirical operator error is uniformly controlled, its effect on the final $Q$-function follows through a stable recursion.

The main limitation is the synchronous generative sampling model, which gives fresh data for every state-action pair at every iteration. Real trajectories are asynchronous and often Markovian. Extending the same robustness principle to that setting requires controlling random visitation counts, temporal dependence, and online data arrival. Those issues lead naturally to the later asynchronous and Markovian developments discussed in the other posts in this series.
