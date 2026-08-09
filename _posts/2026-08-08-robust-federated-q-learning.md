---
title: "Robust Federated Q-Learning with Almost No Communication"
date: 2026-08-08 
categories: [rl-blogs]
rl_section: robust-rl
tags: [federated reinforcement learning, q-learning, byzantine robustness, communication efficiency]
math: true
description: "How low-variance Bellman operators and median-of-means aggregation produce collaborative Q-Learning gains despite Byzantine agents."
---

Federated reinforcement learning promises a simple statistical benefit: if many agents interact with the same environment, pooling their information should reduce the number of samples each agent needs. This promise hides two practical obstacles. Communication may be far more expensive than local sampling, and a small fraction of agents may be faulty or malicious. Naively averaging their messages can allow one arbitrarily large message to destroy the global estimate.

The paper *Robust Federated Q-Learning with Almost No Communication* by Sreejeet Maity and Aritra Mitra addresses both issues simultaneously. Its algorithm, Robust Fed-Q, combines local model estimation, robust median-of-means aggregation, and infrequent server updates. The result is a collaborative $1/\sqrt{MT}$ statistical rate in the benign case, graceful degradation under Byzantine agents, and only logarithmically many communication rounds.

## Federated tabular reinforcement learning

There are $M$ agents, each interacting with the same discounted finite MDP

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma).
$$

The reward is bounded by

$$
|R(s,a)|\leq\overline R.
$$

Each agent has access to a synchronous generative model. For every state-action pair and every query, it receives an independent next-state sample from $P(\cdot\mid s,a)$. Samples are independent across agents. A central server coordinates learning, but raw transitions are kept local.

The optimal $Q$-function is the fixed point of

$$
(\mathcal T^\star Q)(s,a)
=
R(s,a)
+
\gamma
\mathbb E_{s'\sim P(\cdot\mid s,a)}
\left[
\max_{a'}Q(s',a')
\right].
$$

If every agent uses $T$ samples per state-action pair, the system contains $MT$ independent samples per coordinate. A statistically efficient federated method should therefore achieve an error of order $1/\sqrt{MT}$ rather than the single-agent rate $1/\sqrt T$.

## Byzantine agents

An $\varepsilon$-fraction of the agents may be adversarial, with

$$
0\leq\varepsilon<\frac12.
$$

The adversaries know the MDP, the honest agents' data, and the algorithm. They may collude and send arbitrary messages to the server. Their messages are not required to correspond to any trajectory or Bellman update.

The problem is stronger than ordinary noisy federated learning. Honest agents already produce different random estimates because their sampled transitions differ. The server must distinguish this legitimate statistical variation from malicious deviation. During early learning, when honest updates are noisy, Byzantine messages can hide inside the natural spread of the honest population.

## Why lower-variance local messages matter

A standard model-free $Q$-Learning update uses one sampled next state. Such a direction has high variance. Robust Fed-Q instead asks every honest agent to estimate the transition kernel over a full epoch before sending anything.

Divide the $T$ local samples into $K$ epochs of length $H$:

$$
T=KH.
$$

During epoch $k$, honest agent $i$ uses its $H$ next-state observations to form

$$
\widehat P_{i,k}(s'\mid s,a)
=
\frac1H
\sum_{j=1}^{H}
\mathbf 1\{s_{i,k,j}(s,a)=s'\}.
$$

It then constructs the empirical Bellman operator

$$
(\widehat{\mathcal T}_{i,k}Q)(s,a)
=
R(s,a)
+
\gamma
\sum_{s'\in\mathcal S}
\widehat P_{i,k}(s'\mid s,a)
\max_{a'}Q(s',a').
$$

The message sent to the server is the direction

$$
d_{i,k}(s,a)
=
(\widehat{\mathcal T}_{i,k}Q_k)(s,a).
$$

Because $\widehat P_{i,k}$ averages $H$ transitions, the variance of an honest message is reduced by a factor of $H$. Honest directions concentrate tightly around $\mathcal T^\star Q_k$, making it harder for adversaries to masquerade as ordinary sampling noise.

The $Q$-table is not updated locally inside an epoch. Each agent only estimates the operator at the common server iterate $Q_k$. This keeps all honest messages centered on the same target.

## Median-of-means across agents

For a fixed coordinate $(s,a)$ and epoch $k$, the server receives

$$
d_{1,k}(s,a),\ldots,d_{M,k}(s,a).
$$

Some are arbitrary. The server partitions the agents into $P$ buckets, computes the mean within each bucket, and takes the median of the bucket means. The number of buckets is chosen on the order of

$$
P
\asymp
\varepsilon M
+
\log\left(
\frac{|\mathcal S||\mathcal A|T}{\delta}
\right).
$$

The construction balances two requirements. There must be enough buckets that the corrupted agents cannot contaminate a majority of them. Yet each bucket must contain enough honest samples for its mean to concentrate. Under the paper's explicit population and corruption condition, the median remains controlled.

Let $\widetilde d_k(s,a)$ be the robust aggregate. Conditional on the past, honest directions are independent and sub-Gaussian around $(\mathcal T^\star Q_k)(s,a)$ with scale $O(\overline R/((1-\gamma)\sqrt H))$. The median-of-means analysis yields

$$
|\widetilde d_k(s,a)-(\mathcal T^\star Q_k)(s,a)|
\lesssim
\frac{\overline R}{(1-\gamma)\sqrt H}
\left(
\sqrt\varepsilon
+
\sqrt{
\frac{\log(1/\delta)}{M}
}
\right).
$$

The important point is the factor $1/\sqrt H$ multiplying both the statistical and adversarial terms. Local operator refinement causes even the corruption effect to vanish as the epoch length grows.

## The global update

The server performs

$$
Q_{k+1}(s,a)
=
(1-\alpha)Q_k(s,a)
+
\alpha\widetilde d_k(s,a)
$$

and broadcasts $Q_{k+1}$ to the agents. This is the only communication event in the epoch. The server maintains the unique evolving $Q$-table; agents contribute lower-variance operator evaluations.

The algorithm therefore blends model-based and model-free ideas. Agents estimate transition kernels locally, but the server updates a $Q$-function through an approximate Bellman fixed-point iteration. This hybrid structure is what creates both robustness and communication efficiency.

## Main convergence theorem

Choose

$$
K
=
\left\lceil
\frac{c_1\log(MT)}{1-\gamma}
\right\rceil,
\qquad
\alpha
=
\frac{\log(MT)}{(1-\gamma)K},
$$

with $H=T/K$. Let

$$
e_k=\|Q_k-Q^\star\|_\infty.
$$

Then the main theorem gives, with probability at least $1-\delta$,

$$
e_K
\leq
\frac{e_0}{MT}
+
\widetilde O\left(
\frac{\overline R}
{(1-\gamma)^{5/2}\sqrt{MT}}
\right)
+
\widetilde O\left(
\frac{\overline R\sqrt\varepsilon}
{(1-\gamma)^{5/2}\sqrt T}
\right).
$$

The logarithmic factors depend on $M$, $T$, $|\mathcal S||\mathcal A|$, and $1/\delta$.

When $\varepsilon=0$, the dominant rate is $\widetilde O(1/\sqrt{MT})$. This is the desired linear sample-complexity speedup: to achieve the same accuracy, each of $M$ agents needs roughly $1/M$ as many samples as a single learner.

When $\varepsilon>0$, the corruption term decreases as $1/\sqrt T$. Thus the adversaries do not create a permanent error floor. With enough local samples, the effect of a fixed Byzantine fraction vanishes and the algorithm converges exactly to $Q^\star$.

## Why exact recovery is possible here

In Huber-contaminated reward learning, a fixed fraction of individual reward samples can create an unavoidable $\sqrt\varepsilon$ floor under finite variance. Here the adversaries corrupt *agents*, while every honest agent can refine its message using an increasingly long clean local epoch. The honest-message distribution therefore collapses around the true Bellman direction at rate $1/\sqrt H$.

Robust aggregation identifies the center of this increasingly concentrated honest population. Since the honest agents remain in the majority, an arbitrary but fixed Byzantine fraction cannot prevent the uncertainty radius from shrinking to zero. More local data genuinely makes honest and malicious behavior easier to separate.

## Communication complexity

There is one server exchange per epoch, and the number of epochs is

$$
K
=
O\left(
\frac{\log(MT)}{1-\gamma}
\right).
$$

Thus the communication-round complexity is logarithmic in the sample horizon. Standard federated $Q$-Learning methods often communicate after every local step or every short local-update block, producing communication that grows linearly with $T$. Robust Fed-Q instead spends almost all of its time collecting local samples and communicates only when a refined operator estimate is ready.

The phrase "almost no communication" refers to this polylogarithmic dependence. Communication does not disappear, but it becomes negligible compared with the number of environment interactions as $T$ grows.

## Proof intuition

The server update gives

$$
Q_{k+1}-Q^\star
=
(1-\alpha)(Q_k-Q^\star)
+
\alpha(\mathcal T^\star Q_k-\mathcal T^\star Q^\star)
+
\alpha(\widetilde d_k-\mathcal T^\star Q_k).
$$

Using the Bellman contraction,

$$
e_{k+1}
\leq
\left(1-\alpha(1-\gamma)\right)e_k
+
\alpha
\|\widetilde d_k-\mathcal T^\star Q_k\|_\infty.
$$

The first technical step proves that all iterates and honest directions are uniformly bounded by a multiple of $\overline R/(1-\gamma)$. The second step analyzes the median-of-means estimator under bounded honest inputs and Byzantine contamination. Conditional on the past, the $H$ samples used by each honest agent are fresh, so its operator error is centered and sub-Gaussian. A union bound makes the robust aggregation guarantee simultaneous across coordinates and epochs.

Finally, the scalar recursion is unrolled:

$$
e_K
\leq
\left(1-\alpha(1-\gamma)\right)^K e_0
+
\frac{1}{1-\gamma}
\max_{k<K}
\|\widetilde d_k-\mathcal T^\star Q_k\|_\infty.
$$

The chosen $K$ makes the transient at most $e_0/(MT)$, while $H=T/K$ turns the empirical-operator radius into the two statistical terms in the theorem.

## What collaboration buys

The honest statistical term scales with $1/\sqrt M$ because the server aggregates information across agents. The Byzantine term scales with $\sqrt\varepsilon$ but not with $1/\sqrt M$ in the same way, because increasing the number of agents while keeping a fixed corrupted fraction also increases the number of adversaries. More agents are beneficial when the collaborative statistical reduction dominates the robust-aggregation cost.

The theorem therefore separates two resources. Increasing $M$ supplies more independent honest replicas per epoch. Increasing $T$ improves the accuracy of each honest replica. Robust aggregation needs both a sufficiently large honest majority and sufficiently concentrated honest messages.

## Scope and limitations

The result uses tabular $Q$-Learning, a common MDP across agents, synchronous generative sampling, deterministic bounded rewards, independent honest data, a known upper bound on the Byzantine fraction, and a central server. The population must also be large enough for the chosen bucket structure to be valid.

These assumptions isolate the central statistical and communication question. The paper establishes that Byzantine robustness need not destroy the $1/\sqrt{MT}$ collaborative gain and need not require communication proportional to the sample horizon. Extending the same principle to asynchronous trajectories, heterogeneous environments, partial state-action coverage, or serverless networks requires new mechanisms, but the organizing idea remains powerful: refine local Bellman information before communicating it, then robustly aggregate the resulting low-variance directions.
