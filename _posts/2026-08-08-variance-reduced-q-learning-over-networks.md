---
title: "Variance-Reduced Q-Learning over Static and Time-Varying Networks"
date: 2026-08-08 
categories: [rl-blogs]
rl_section: robust-rl
tags: [decentralized reinforcement learning, q-learning, consensus, variance reduction, time-varying networks]
math: true
description: "How local Bellman-operator estimation and network diffusion deliver collaborative Q-Learning speedups with polylogarithmic communication."
---

In decentralized reinforcement learning, several agents interact with the same environment but communicate only with neighbors in a graph. There is no central server that can instantly average their data. The statistical goal is still clear: $N$ agents collecting $T$ samples each should behave like a learner with $NT$ samples. The systems question is harder: how much network communication is required to realize that gain?

The paper *Variance-Reduced Q-Learning over Static and Time-Varying Networks* by Sreejeet Maity, Feng Zhu, Aritra Mitra, and Robert W. Heath Jr. introduces Variance-Reduced Diffused $Q$-Learning, abbreviated VRDQ. The method makes only a logarithmic number of $Q$-updates, estimates a low-variance Bellman operator locally between updates, and diffuses the resulting directions through average consensus. It achieves a near-optimal $1/\sqrt{NT}$ statistical rate with polylogarithmic communication on both static and time-varying networks.

## The decentralized learning model

There are $N$ agents connected by an undirected graph

$$
\mathcal G=(\mathcal V,\mathcal E),
\qquad
\mathcal V=[N].
$$

Every agent interacts with the same finite discounted MDP

$$
\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma),
$$

with bounded deterministic rewards

$$
|R(s,a)|\leq\overline R.
$$

Each agent has a synchronous generative model that supplies an independent next-state observation for every state-action pair at each query. Samples are independent across agents. The optimal $Q$-function satisfies

$$
Q^\star=\mathcal T^\star Q^\star,
$$

where

$$
(\mathcal T^\star Q)(s,a)
=
R(s,a)
+
\gamma
\sum_{s'\in\mathcal S}
P(s'\mid s,a)
\max_{a'}Q(s',a').
$$

If agents act separately, each obtains the single-agent rate $\widetilde O(1/\sqrt T)$. A fully efficient collaborative method should reduce this to $\widetilde O(1/\sqrt{NT})$ at every node.

## Network mixing

Communication is represented by a symmetric doubly stochastic matrix $W\in\mathbb R^{N\times N}$. Agent $i$ assigns weight $W_{ij}$ to agent $j$'s message, and $W_{ij}=0$ when the agents are not neighbors. Repeated multiplication by $W$ drives vectors toward their network average.

For a connected, aperiodic static graph, there exist $C_1>0$ and $\rho\in(0,1)$ such that

$$
\left\|
[W^\ell]_i-\frac1N\mathbf 1^\top
\right\|
\leq
C_1\rho^\ell.
$$

The parameter $\rho$ measures how slowly the network mixes. A small $\rho$ means consensus is fast; a value near one means information requires many rounds to spread.

## Why communicating every sample is wasteful

Many distributed stochastic-approximation methods update and communicate after each new observation. Their communication cost is therefore $O(T)$. This can be prohibitive when wireless links, bandwidth, or energy are the limiting resources.

The paper starts from a different observation. Agents need to exchange information primarily when they update their $Q$-tables. If accurate learning requires only a small number of well-designed $Q$-updates, then communication can be reduced by the same factor. The challenge is to make each infrequent update accurate enough to preserve the $1/\sqrt{NT}$ rate.

## Epoch-based local operator estimation

Each agent partitions its $T$ samples into $K$ epochs of length $H$:

$$
T=KH.
$$

During epoch $k$, agent $i$ estimates the transition kernel by

$$
\widehat P_{i,k}(s'\mid s,a)
=
\frac1H
\sum_{u=0}^{H-1}
\mathbf 1\{s_{i,k,u}(s,a)=s'\}.
$$

It then constructs a local empirical Bellman operator

$$
(\widehat{\mathcal T}_{i,k}f)(s,a)
=
R(s,a)
+
\gamma
\sum_{s'\in\mathcal S}
\widehat P_{i,k}(s'\mid s,a)
\max_{a'}f(s',a').
$$

Because the operator averages $H$ next-state samples, its stochastic error is smaller by order $1/\sqrt H$ than a one-sample $Q$-Learning direction. A longer epoch gives a more accurate local estimate without requiring any communication.

## Diffusing the previous epoch's direction

At the beginning of epoch $k$, agent $i$ initializes a direction using the operator estimated in the previous epoch:

$$
d_{i,k}^{(0)}
=
\widehat{\mathcal T}_{i,k-1}Q_{i,k}.
$$

The convention is $\widehat{\mathcal T}_{i,-1}f=0$. Agents run $L$ consensus steps:

$$
d_{i,k}^{(\ell+1)}(s,a)
=
\sum_{j\in\mathcal N_i}
W_{ij}d_{j,k}^{(\ell)}(s,a),
\qquad
\ell=0,\ldots,L-1.
$$

In stacked form,

$$
d_k^{(L)}(s,a)
=
W^L d_k^{(0)}(s,a).
$$

After enough rounds, $d_{i,k}^{(L)}$ approximates the average of all agents' empirical Bellman directions. Every node then updates

$$
Q_{i,k+1}
=
(1-\alpha)Q_{i,k}
+
\alpha d_{i,k}^{(L)}.
$$

The one-epoch delay is deliberate. During epoch $k$, an agent can estimate the next local operator using fresh samples while simultaneously diffusing the already constructed direction from epoch $k-1$. Computation, sampling, and communication are overlapped rather than serialized.

## Where the collaborative gain comes from

Conditioned on the current iterates, the local empirical operators are independent across agents. Averaging them reduces variance by a factor of $N$. Since each agent already averages $H$ samples inside the epoch, the ideal network-average direction has stochastic scale

$$
\frac{1}{\sqrt{NH}}.
$$

Consensus does not create the statistical gain; it transports the average that contains the gain. The role of $L$ is to make the network disagreement smaller than the $1/\sqrt{NH}$ statistical uncertainty. Once diffusion error is below that scale, further consensus rounds do not improve the leading learning rate.

## The static-network theorem

Define the local terminal error

$$
e_{i,K}=\|Q_{i,K}-Q^\star\|_\infty.
$$

The algorithm chooses

$$
K
=
\left\lceil
\frac{c_1\log(NT)}{1-\gamma}
\right\rceil,
\qquad
\alpha
=
\frac{\log(NT)}{(1-\gamma)K},
$$

and a diffusion length of logarithmic order

$$
L
\asymp
\frac{
\log\left(N^{3/2}\sqrt{T(1-\gamma)}\right)
}{
\log(1/\rho)
}.
$$

Then, with probability at least $1-\delta$, every agent satisfies

$$
e_{i,K}
\leq
\frac{e_{i,0}}{NT}
+
\widetilde O\left(
\frac{\overline R}
{(1-\gamma)^{5/2}\sqrt{NT}}
\right).
$$

The logarithmic factors include $\log(NT)$ and $\log(|\mathcal S||\mathcal A|T/\delta)$. The result exhibits the desired linear speedup: the error at each node depends on the total system sample size $NT$, even though no node can directly access all samples.

## Communication complexity

There are $K$ epochs and $L$ consensus rounds per epoch, so the communication cost per agent is

$$
KL
=
\widetilde O(1),
$$

more explicitly polylogarithmic in $N$ and $T$. With the theorem's choices, the dependence is of order $\log^2(NT)$ up to graph and discount factors.

This is the key systems-level result. The sample horizon may be very large, but the number of messages per agent grows only logarithmically. VRDQ therefore avoids the usual tradeoff in which collecting more samples automatically forces proportionally more communication.

## How topology enters the guarantee

The final leading error does not explicitly contain $\rho$. This does not mean topology is irrelevant. A slowly mixing graph requires a larger $L$ before the consensus error falls below the statistical error. Because consensus must fit inside the epoch, the algorithm needs

$$
L\leq H=\frac{T}{K}.
$$

Thus the graph controls the burn-in sample horizon required for the theorem to apply. Once $T$ is large enough and $L$ is chosen appropriately, topology no longer appears in the leading statistical rate.

This separation is useful conceptually. The network determines how long agents must communicate to emulate centralized averaging; the environment samples determine how accurately that average estimates the Bellman operator.

## Extension to time-varying networks

Suppose the graph and mixing matrix change with time, producing matrices

$$
W(0),W(1),W(2),\ldots.
$$

Each $W(t)$ is doubly stochastic and respects the graph available at time $t$. The paper assumes that there is a block length $B$ such that every product over a block contracts disagreement uniformly. If

$$
W_B(t)
=
W(t)W(t-1)\cdots W(t-B+1),
$$

then the assumption requires

$$
\sup_{t\geq B-1}
\left\|
W_B(t)-\frac1N\mathbf 1\mathbf 1^\top
\right\|_2
\leq
\omega
<1.
$$

The static power $W^L$ is replaced by a product of time-varying matrices. Choosing a logarithmic number of contracting blocks makes the disagreement as small as before. The same $1/\sqrt{NT}$ terminal bound follows, while communication remains polylogarithmic. The additional block length $B$ may depend on the network sequence but not on the sample horizon $T$.

## Proof architecture: average plus disagreement

The analysis decomposes the network dynamics into a centralized average and a disagreement term. Define

$$
\overline Q_k
=
\frac1N\sum_{i=1}^{N}Q_{i,k},
\qquad
\overline d_k
=
\frac1N\sum_{i=1}^{N}d_{i,k}^{(L)}.
$$

Double stochasticity preserves averages, so

$$
\overline Q_{k+1}
=
(1-\alpha)\overline Q_k
+
\alpha\overline d_k.
$$

The average error obeys the Bellman-type recursion

$$
\|\overline Q_{k+1}-Q^\star\|_\infty
\leq
\left(1-\alpha(1-\gamma)\right)
\|\overline Q_k-Q^\star\|_\infty
+
\alpha
\|\overline d_k-\mathcal T^\star\overline Q_k\|_\infty.
$$

The last term is split into local operator-estimation error and a consensus gap caused by agents evaluating the Bellman operator at slightly different $Q_{i,k}$. Concentration across $NH$ independent transition samples produces the $1/\sqrt{NH}$ term.

Separately, the mixing inequality gives

$$
\left\|
d_{i,k}^{(L)}
-
\frac1N\sum_{j=1}^{N}d_{j,k}^{(0)}
\right\|_\infty
\lesssim
\frac{N\overline R\rho^L}{1-\gamma}.
$$

This bounds both direction disagreement and, after the local recursions are unrolled, $Q$-table disagreement. The selected $L$ makes these network terms no larger than the statistical term. Finally, the Bellman recursion is iterated over $K$ epochs, and the choice of $\alpha K$ reduces the initialization error to order $1/(NT)$.

## Why variance reduction and diffusion fit together

Local operator estimation and consensus solve different problems. Operator estimation turns $H$ noisy samples into a stable local direction. Diffusion turns $N$ stable local directions into an approximation of their average. If the directions remained high variance, consensus would faithfully average noisy objects but frequent updates would still be required. If agents estimated accurate directions but never diffused them, each would retain only a $1/\sqrt T$ rate. VRDQ obtains the product benefit $1/\sqrt{NH}$ by combining the two stages at the epoch level.

The decoupling also simplifies the proof. Statistical error can be bounded conditionally using fresh samples, while network error is controlled deterministically through mixing. Their contributions are balanced only at the end by the choice of $L$.

## Scope and future directions

The paper studies a common tabular MDP, synchronous generative sampling, bounded deterministic rewards, full state-action access, and doubly stochastic undirected communication. These assumptions make it possible to isolate the fundamental relationship between collaborative sample efficiency and communication.

Natural extensions include asynchronous trajectory data, stochastic or heavy-tailed rewards, heterogeneous behavior policies, partial state-action coverage, directed graphs, and adversarial agents. The central design principle should remain useful in those settings: communicate low-variance Bellman information only when a meaningful $Q$-update is ready, and run consensus just long enough that network error is dominated by statistical uncertainty.
