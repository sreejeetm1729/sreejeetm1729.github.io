---
title: "Turbo the Snail as a Robust POMDP"
date: 2026-06-11
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [reinforcement-learning, robust-rl, pomdp, exploration, minimax, gridworld]
math: true
---

A good puzzle is often a small theorem in disguise. The Turbo the Snail problem **[Problem 5, IMO 2024]** is a particularly nice example: it looks like a deterministic olympiad-style grid problem, but it can be read as a clean model of robust exploration under partial information.

Turbo moves on a board with hidden monsters. He does not know where the monsters are, but the monster configuration is fixed. Every failed attempt reveals information. The key question is not whether a safe path exists; by the structural assumptions, some safe structure must exist. The question is how quickly Turbo can force himself to find enough information to exploit it.

That is exactly the language of reinforcement learning: an agent interacts with an unknown environment, receives partial feedback, updates its information state, and acts again. The original problem is not vanilla stochastic RL. It is closer to a **robust partially observable Markov decision process** with a static latent environment and episodic resets.

The point of this note is to formalize that connection.

## 1. The deterministic game

Let the board have rows

$$
1,2,\ldots,R
$$

and columns

$$
1,2,\ldots,C.
$$

In the original problem,

$$
R=2024,
\qquad
C=2023.
$$

The monster set is a hidden subset

$$
M\subseteq \{2,3,\ldots,R-1\}\times \{1,2,\ldots,C\}.
$$

The assumptions are

$$
\left|M\cap \bigl(\{r\}\times [C]\bigr)\right|=1
\qquad
\text{for every } r\in \{2,\ldots,R-1\},
$$

and

$$
\left|M\cap \bigl([R]\times \{c\}\bigr)\right|\leq 1
\qquad
\text{for every } c\in [C].
$$

So every intermediate row has exactly one monster, and every column has at most one monster. Let

$$
\mathcal M
$$

be the class of all monster configurations satisfying these constraints.

Turbo knows $\mathcal M$, but he does not know the true $M\in \mathcal M$.

On each attempt, Turbo starts at any cell in the first row. He moves to adjacent cells sharing a side. If he reaches a monster, the attempt ends and he is reset to the first row. If he reaches any cell in row $R$, the game ends successfully. Turbo remembers all cells he has tested.

The original contest question asks for the smallest $n$ such that Turbo has a strategy guaranteeing success by attempt $n$, uniformly over all $M\in\mathcal M$.

In minimax notation, define

$$
T_\pi(M)
=
\text{the first attempt on which policy }\pi\text{ reaches row }R\text{ under monster map }M.
$$

The deterministic minimax value is

$$
N^\star
=
\inf_\pi \sup_{M\in\mathcal M} T_\pi(M).
$$

The answer to the original problem is

$$
\boxed{N^\star=3.}
$$

The rest of the note explains why this is naturally a robust RL/POMDP statement.

## 2. Why this is not just a shortest-path problem

If the monster configuration $M$ were known, the problem would be a standard deterministic planning problem: find a path from the top row to the bottom row avoiding the blocked cells.

But $M$ is not known. Turbo only learns by stepping on cells. The environment has a hidden static parameter, namely the entire monster map $M$. Therefore the true state is not merely Turbo's current cell. A fully observed state would have to include the hidden map:

$$
s_t=(x_t,M),
\qquad
x_t=(r_t,c_t).
$$

Turbo observes $x_t$, but not $M$. He also observes whether a tested cell is safe or contains a monster. Therefore the process is partially observable.

The important point is that the hidden map does not change across attempts. This means the agent is not solving independent episodes. It is solving an adaptive exploration problem where information carries over from one episode to the next.

This is the first RL interpretation:

> Turbo's game is an episodic POMDP with a static latent environment parameter and reset after failure.

The second interpretation is robust:

> Since the guarantee must hold for every admissible monster map, the objective is minimax rather than average-case.

So a precise phrase is:

> Turbo's game is a finite deterministic robust POMDP with goal-reaching objective and episodic resets.

## 3. The information state

The natural Markov state for the agent is not the physical position alone. It is the physical position together with the set of monster maps still consistent with the observations.

Let

$$
\mathcal B_t\subseteq \mathcal M
$$

denote Turbo's belief set at time $t$. This is not necessarily a probability distribution. It is a version space: the set of all monster configurations that have not yet been ruled out.

If Turbo tests cell $x$ and finds it safe, then

$$
\mathcal B_{t+1}
=
\{M\in\mathcal B_t:x\notin M\}.
$$

If Turbo tests cell $x$ and finds a monster, then

$$
\mathcal B_{t+1}
=
\{M\in\mathcal B_t:x\in M\}.
$$

Thus the belief update is deterministic once the observation is known.

This is the standard POMDP trick: convert a partially observable problem into a fully observed belief-state problem. The belief state is combinatorial, but it is Markov.

A history-dependent policy

$$
\pi(a_t\mid h_t)
$$

can therefore be replaced by a policy of the form

$$
\pi(a_t\mid x_t,\mathcal B_t).
$$

The price is that $\mathcal B_t$ may be exponentially large.

## 4. A Boolean Bellman equation for guaranteed reachability

Because the original puzzle asks for a deterministic guarantee, it is useful to write the dynamic programming recursion in Boolean form.

For a cell $x$, let $\mathcal A(x)$ be the set of legal moves. For a move $a\in\mathcal A(x)$, let

$$
y=f(x,a)
$$

be the next cell.

For a belief set $\mathcal B$ and a cell $y$, define the safe and monster branches

$$
\mathcal B^0(y)
=
\{M\in\mathcal B:y\notin M\},
$$

and

$$
\mathcal B^1(y)
=
\{M\in\mathcal B:y\in M\}.
$$

Let

$$
V_j(x,\mathcal B)\in\{0,1\}
$$

denote whether Turbo can force success starting from cell $x$, with belief $\mathcal B$, using at most $j$ remaining attempts including the current one.

Let

$$
U_j(\mathcal B)
=
\max_{c\in[C]} V_j((1,c),\mathcal B)
$$

be the corresponding value when a fresh attempt may start in any top-row cell.

The boundary condition is

$$
V_j(x,\mathcal B)=1
\qquad
\text{if }x\text{ lies in row }R.
$$

For nonterminal cells, Turbo chooses a move. The adversarial hidden map determines whether the next cell is safe or a monster, subject to the current belief set. Therefore a move is winning only if every possible branch is winning. This gives the Boolean recursion

$$
V_j(x,\mathcal B)
=
\bigvee_{a\in\mathcal A(x)}
\left[
\Bigl(\mathcal B^0(y)=\varnothing
\ \lor\
V_j(y,\mathcal B^0(y))=1\Bigr)
\land
\Bigl(\mathcal B^1(y)=\varnothing
\ \lor\
U_{j-1}(\mathcal B^1(y))=1\Bigr)
\right],
$$

where $y=f(x,a)$.

This is a reachability-game analogue of the Bellman equation.

The original minimax value is

$$
N^\star
=
\min\{j:U_j(\mathcal M)=1\}.
$$

For the Turbo puzzle,

$$
U_2(\mathcal M)=0,
\qquad
U_3(\mathcal M)=1.
$$

So the theorem $N^\star=3$ is exactly a Bellman-style robust reachability statement.

## 5. Why two attempts are impossible

We now prove the lower bound

$$
N^\star\geq 3.
$$

Fix any deterministic strategy. On the first attempt, Turbo must eventually enter some cell in row $2$ if he is to reach the last row. Let

$$
(2,c_1)
$$

be the first row-$2$ cell he visits.

There exists an admissible monster configuration with a monster at $(2,c_1)$. Under that configuration, Turbo's first attempt ends immediately when he enters $(2,c_1)$.

Now Turbo knows that row $2$ has its monster in column $c_1$. In particular, column $c_1$ contains no other monster. However, the cell $(2,c_1)$ itself blocks direct passage through that column from the top.

On the second attempt, if Turbo is to reach the last row, he must eventually enter row $3$. Since he cannot safely pass through $(2,c_1)$, the first row-$3$ cell he enters has the form

$$
(3,c_2)
$$

with

$$
c_2\neq c_1.
$$

There exists an admissible monster configuration with monsters at both

$$
(2,c_1)
\qquad\text{and}\qquad
(3,c_2).
$$

The column constraint is satisfied because $c_2\neq c_1$. The remaining rows can be assigned monsters in distinct unused columns. There are enough columns to do this in the original board.

Under this configuration, Turbo fails on the second attempt as well. Therefore no strategy can guarantee success by the second attempt.

Hence

$$
N^\star\geq 3.
$$

## 6. A three-attempt robust strategy

We now describe a strategy that guarantees success by attempt $3$.

The first attempt is used to identify the monster in row $2$.

Turbo starts in the first row and enters row $2$. Then he walks along row $2$ until he hits the unique monster in that row. Suppose the monster is found at

$$
(2,c).
$$

After this, Turbo knows two important facts:

1. every cell in row $2$ except $(2,c)$ is safe;
2. every cell in column $c$ except $(2,c)$ is safe.

Thus column $c$ is a safe vertical highway below row $2$. The only problem is that the entrance cell $(2,c)$ is blocked. Turbo must get around it and enter column $c$ from the side at some lower row.

### Interior case

Suppose

$$
1<c<C.
$$

On attempt $2$, Turbo starts above column $c+1$, moves to $(2,c+1)$, and then tries to enter $(3,c+1)$.

If $(3,c+1)$ is safe, then Turbo moves left to $(3,c)$ and then straight down column $c$ to the last row. This succeeds because column $c$ has no monster below row $2$.

If $(3,c+1)$ is a monster, the second attempt fails. But now Turbo knows that row $3$ has its monster at $(3,c+1)$. On attempt $3$, he starts above column $c-1$, moves to $(2,c-1)$, then to $(3,c-1)$, then right to $(3,c)$, and then straight down column $c$.

This is safe because row $3$ has no monster at $(3,c-1)$ or $(3,c)$, and column $c$ has no monster below row $2$.

So the interior case is solved in at most three attempts.

### Left-edge case

Now suppose

$$
c=1.
$$

Then column $1$ is safe below row $2$, but the cell $(2,1)$ blocks direct access to it.

Turbo uses attempt $2$ to follow a staircase path starting from column $3$:

$$
(1,3)\to (2,3)\to (3,3)\to (3,4)\to (4,4)\to (4,5)\to\cdots\to (C,C)\to (R,C).
$$

If this path reaches row $R$, Turbo wins on attempt $2$.

Otherwise, it hits a monster at some cell $(r,s)$ with $r\geq 3$. We show that attempt $3$ then succeeds.

There are two possibilities.

First, suppose the failed cell was entered horizontally from the left. Then the previous cell $(r,s-1)$ is known to be safe. Since the only monster in row $r$ is at $(r,s)$, every cell

$$
(r,1),(r,2),\ldots,(r,s-1)
$$

is safe. On attempt $3$, Turbo follows the already verified safe prefix of the staircase to $(r,s-1)$, moves left to column $1$, and then moves straight down column $1$. Column $1$ is safe below row $2$, so he reaches the last row.

Second, suppose the failed cell was entered vertically from above. In the staircase, this means Turbo tried to move from $(r-1,s)$ to $(r,s)$. The prefix of the staircase has already verified safe access to $(r-1,s)$, and also to $(r-1,s-1)$, except in the initial case where $r=3$, where row $2$ is already known to be safe outside column $1$. Since the monster in row $r$ is at $(r,s)$, the cell $(r,s-1)$ is safe. On attempt $3$, Turbo reaches $(r-1,s-1)$ through known safe cells, moves down to $(r,s-1)$, then moves left along row $r$ to column $1$, and finally moves straight down column $1$.

Again this reaches the last row safely.

Thus the left-edge case is solved in at most three attempts.

### Right-edge case

If

$$
c=C,
$$

Turbo uses the mirror image of the left-edge strategy. Column $C$ is safe below row $2$, and the staircase is reflected horizontally. The same argument guarantees success by the third attempt.

Combining the interior and edge cases gives

$$
N^\star\leq 3.
$$

Together with the lower bound, this proves

$$
\boxed{N^\star=3.}
$$

## 7. RL interpretation of the proof

The proof has a very RL-flavored structure.

The first attempt is pure exploration. Turbo deliberately searches row $2$ until he discovers a monster. This is costly because it sacrifices the first episode, but the information gained is extremely valuable: it certifies an entire safe column below row $2$.

The second attempt tries to exploit that safe column. If the local bypass succeeds, Turbo wins immediately. If it fails, the failure itself is informative: it reveals the unique monster in another row, creating enough structure to construct a guaranteed bypass on the third attempt.

So the proof is not merely path planning. It is adaptive information acquisition.

The key invariant is:

> A discovered monster at $(r,c)$ certifies that the rest of row $r$ is safe and the rest of column $c$ is safe.

The strategy is designed so that every possible failure creates a certificate strong enough to complete the task on the next attempt.

This is a deterministic analogue of exploration bonuses and information-directed exploration: the agent chooses actions whose failures are useful.

## 8. A stochastic RL version

The original puzzle is worst-case and deterministic. A more standard RL version would randomize the hidden monster map.

Let

$$
M\sim \mu
$$

for some unknown distribution $\mu$ over $\mathcal M$. The agent interacts over episodes and wants to minimize the expected number of failures before success:

$$
\inf_\pi \mathbb E_{M\sim\mu}^\pi[T_\pi(M)].
$$

If $\mu$ is known, the problem is a Bayesian POMDP. The belief is no longer just a set $\mathcal B_t$; it is a posterior distribution

$$
b_t(M)=\mathbb P(M\mid h_t).
$$

The Bayes update is

$$
b_{t+1}(M)
\propto
b_t(M)\mathbf 1\{x_t\notin M\}
$$

after a safe observation, and

$$
b_{t+1}(M)
\propto
b_t(M)\mathbf 1\{x_t\in M\}
$$

after hitting a monster.

A value function can then be written as

$$
V(x,b)
=
\max_{a\in\mathcal A(x)}
\mathbb E\left[ r(x,a,Y)+\gamma V(x',b')\mid x,b,a\right],
$$

where $Y$ is the safety observation. This is the standard belief-MDP formulation.

But the deterministic puzzle corresponds to the stronger robust objective

$$
\inf_\pi \sup_{M\in\mathcal M} T_\pi(M).
$$

The Bayesian objective asks for good average performance. The robust objective asks for a uniform certificate.

## 9. An adversarially corrupted version

One can make the model even closer to adversarially robust RL by corrupting the feedback.

Suppose that when Turbo tests a cell $x$, the true hazard indicator is

$$
H_M(x)=\mathbf 1\{x\in M\}.
$$

Instead of observing $H_M(x)$ exactly, Turbo observes

$$
Y_t
=
\begin{cases}
H_M(x_t), & \text{with probability }1-\varepsilon,\\
Z_t, & \text{with probability }\varepsilon,
\end{cases}
$$

where $Z_t$ may be chosen adversarially.

This is a Huber-contaminated observation model. Now a single observation no longer certifies a cell as safe or unsafe. The belief update must be robustified. Turbo may need repeated testing, majority votes, median-of-means style aggregation, or confidence sets.

The robust objective could be

$$
\inf_\pi \sup_{M\in\mathcal M}\sup_{\text{adversary}}
\mathbb P\left(T_\pi(M)>n\right).
$$

Or one may ask for a high-probability guarantee:

$$
\mathbb P\left(T_\pi(M)\leq n\right)
\geq 1-\delta
\qquad
\text{for all }M\in\mathcal M.
$$

This version has the same conceptual core as the puzzle, but now the information itself is unreliable. That makes it a natural toy model for adversarially robust exploration.

## 10. What this toy problem teaches

The Turbo puzzle compresses several important RL ideas into a tiny deterministic model.

| Puzzle feature                 | RL interpretation                    |
| ------------------------------ | ------------------------------------ |
| Hidden monsters                | latent environment parameter         |
| Attempts                       | episodes                             |
| Reset after monster            | terminal failure and restart         |
| Memory across attempts         | history-dependent policy             |
| Structural monster constraints | model class / prior knowledge        |
| Testing cells                  | exploration                          |
| Reaching the bottom row        | goal-reaching objective              |
| Guarantee for all monster maps | robust/minimax RL                    |
| The value $N^\star=3$          | worst-case sample-complexity theorem |

The most important lesson is that exploration is not only about finding a good path. It is about finding information that makes a good path certifiable.

Turbo's first failure is not a failure in the learning sense. It is a measurement. The discovered monster converts an unknown board into one with a certified safe column. The second attempt either exploits this certificate or produces another certificate. By the third attempt, the accumulated information is enough to force success.

In this sense, the puzzle is a finite, exact, minimax version of a question that appears throughout reinforcement learning:

> How many interactions are needed before an agent can act safely and successfully in every environment consistent with its observations?

For Turbo, the answer is exactly three.
