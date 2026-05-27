---
title: "Function Approximation in RL: From Tables to Linear Models to Neural Networks"
date: 2026-05-27 00:00:00 -0500
categories: [rl-blogs]
tags: [rl, function-approximation, q-learning, deep-rl, neural-networks]
math: true
---

# Function Approximation in Reinforcement Learning: From Tables to Linear Models to Neural Networks

Reinforcement learning is about learning how to act through interaction. An agent observes a state, takes an action, receives a reward, and moves to a new state. Over time, the agent tries to learn a policy that collects large long-term reward.

Mathematically, we usually model this interaction as a Markov Decision Process, or MDP,

$$
\mathcal M = (\mathcal S,\mathcal A,P,r,\gamma),
$$

where $$\mathcal S$$ is the state space, $$\mathcal A$$ is the action space, $$P$$ is the transition kernel, $$r$$ is the reward function, and $$\gamma \in [0,1)$$ is the discount factor.

A policy $$\pi$$ tells the agent how to act:

$$
\pi(a \mid s) = \mathbb P(a_t=a \mid s_t=s).
$$

For a fixed policy $$\pi$$, the state-value function is

$$
V^\pi(s)
=
\mathbb E^\pi
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)
\mid s_0=s
\right].
$$

The action-value function is

$$
Q^\pi(s,a)
=
\mathbb E^\pi
\left[
\sum_{t=0}^{\infty} \gamma^t r(s_t,a_t)
\mid s_0=s,\ a_0=a
\right].
$$

The optimal action-value function is

$$
Q^\star(s,a)
=
\sup_\pi Q^\pi(s,a).
$$

Once we know $$Q^\star$$, an optimal policy can be obtained greedily:

$$
\pi^\star(s) \in \arg\max_{a\in \mathcal A} Q^\star(s,a).
$$

So one of the central problems in RL is value-function estimation. The important question is:

**How do we represent the value function?**

This is exactly where function approximation enters.

---

## 1. The Tabular World

The simplest setting is the tabular setting. Here, the state and action spaces are finite and small enough that we can store one number for each state-action pair:

$$
Q(s,a), \qquad (s,a)\in \mathcal S\times \mathcal A.
$$

If the state space has size

$$
|\mathcal S|
$$

and the action space has size

$$
|\mathcal A|,
$$

then the tabular $$Q$$-function contains one entry for every state-action pair. Therefore, the table has size


$$
|\mathcal S||\mathcal A|.
$$

For small problems, such as a tiny gridworld, this is perfectly reasonable. If the agent has four actions, say up, down, left, and right, then each state has four stored action-values.

The beauty of the tabular setting is that the Bellman equations become finite-dimensional fixed-point equations.

For a fixed policy $$\pi$$, the Bellman operator is

$$
(T^\pi Q)(s,a)
=
r(s,a)
+
\gamma
\mathbb E
\left[
Q(s',a')\mid s,a
\right],
$$

where

$$
s' \sim P(\cdot\mid s,a),
\qquad
a' \sim \pi(\cdot\mid s').
$$

The value function $$Q^\pi$$ is the fixed point of this operator:

$$
Q^\pi = T^\pi Q^\pi.
$$

For optimal control, the Bellman optimality operator is

$$
(T^\star Q)(s,a)
=
r(s,a)
+
\gamma
\mathbb E
\left[
\max_{a'\in\mathcal A} Q(s',a')\mid s,a
\right].
$$

The optimal action-value function satisfies

$$
Q^\star = T^\star Q^\star.
$$

In the tabular case, $$Q$$ is just a vector in

$$
\mathbb R^{|\mathcal S||\mathcal A|}.
$$

So the RL problem becomes the problem of learning the entries of this vector.

---

## 2. Tabular $$Q$$-Learning

The classical tabular $$Q$$-Learning update is

$$
Q_{t+1}(s_t,a_t)
=
(1-\eta_t)Q_t(s_t,a_t)
+
\eta_t
\left[
r_t
+
\gamma
\max_{a'\in\mathcal A}Q_t(s_{t+1},a')
\right].
$$

Equivalently,

$$
Q_{t+1}(s_t,a_t)
=
Q_t(s_t,a_t)
+
\eta_t
\left[
r_t
+
\gamma
\max_{a'}Q_t(s_{t+1},a')
-
Q_t(s_t,a_t)
\right].
$$

All other entries remain unchanged:

$$
Q_{t+1}(s,a)=Q_t(s,a),
\qquad
(s,a)\neq (s_t,a_t).
$$

The temporal-difference error is

$$
\delta_t
=
r_t
+
\gamma
\max_{a'}Q_t(s_{t+1},a')
-
Q_t(s_t,a_t).
$$

Thus the update becomes

$$
Q_{t+1}(s_t,a_t)
=
Q_t(s_t,a_t)+\eta_t\delta_t.
$$

This is the basic mechanism of bootstrapping. The estimate at $$(s_t,a_t)$$ is moved toward the target

$$
r_t+
\gamma
\max_{a'}Q_t(s_{t+1},a').
$$

The target says that the value of taking action $$a_t$$ at state $$s_t$$ should be the immediate reward plus the discounted value of the best future action.

In the tabular case, each update changes one coordinate of the table. This locality makes the analysis clean. Under standard assumptions, tabular $$Q$$-Learning converges to $$Q^\star$$.

---

## 3. Why the Tabular Representation Breaks Down

The tabular representation is simple, but it has a serious limitation: it requires one parameter per state-action pair.

If the state space is large, then

$$
|\mathcal S||\mathcal A|
$$

can be enormous. If the state space is continuous, then maintaining a table is impossible.

For example, in robotics, a state may include positions, velocities, angles, forces, and sensor readings. In autonomous driving, a state may include the positions and velocities of surrounding cars. In Atari, the state may be an image. In control, the state may be a vector in $$\mathbb R^d$$.

If

$$
\mathcal S = \mathbb R^d,
$$

there are infinitely many states. A table is no longer meaningful.

Even when the state space is finite but huge, tabular methods are inefficient because they do not generalize. Learning the value of one state-action pair tells us nothing about another state-action pair, even if the two are very similar.

This is wasteful. If two states are similar, we would like learning in one state to help us in the other.

This motivates function approximation.

---

## 4. The Main Idea of Function Approximation

Function approximation means replacing the table by a parameterized function.

Instead of storing

$$
Q(s,a)
$$

for every $$(s,a)$$, we write

$$
Q_\theta(s,a)\approx Q^\star(s,a),
$$

where $$\theta$$ is a finite-dimensional parameter vector.

So the learning problem becomes

$$
\text{learn }\theta
\qquad
\text{instead of learning every table entry separately.}
$$

More generally, we choose a function class

$$
\mathcal F
=
\{Q_\theta:\theta\in\mathbb R^d\}.
$$

Then we try to find a parameter vector $$\theta$$ such that

$$
Q_\theta \approx Q^\star.
$$

The same idea applies to value functions and policies:

$$
V_\theta(s)\approx V^\pi(s),
$$

$$
Q_\theta(s,a)\approx Q^\pi(s,a),
$$

and

$$
\pi_\theta(a\mid s)\approx \pi^\star(a\mid s).
$$

The main benefit is generalization. The function approximator shares parameters across states and actions. Therefore, learning from one state-action pair can improve estimates for other related state-action pairs.

---

## 5. Tabular RL as a Special Case of Linear Approximation

It is useful to see that tabular RL is actually a special case of function approximation.

For each state-action pair $$(s,a)$$, define a one-hot feature vector

$$
\phi(s,a)\in \mathbb R^{|\mathcal S||\mathcal A|}.
$$

This vector has a single 1 in the coordinate corresponding to $$(s,a)$$ and 0 everywhere else.

Now define

$$
Q_\theta(s,a)=\theta^T \phi(s,a).
$$

Since $$\phi(s,a)$$ is one-hot, this simply selects one coordinate of $$\theta$$:

$$
Q_\theta(s,a)=\theta_{s,a}.
$$

Thus a table is a linear function approximator with one-hot features.

This perspective is important. It shows that the movement from tabular RL to function approximation is not a complete break. It is a generalization.

In tabular RL, the features do not share information across state-action pairs. In linear function approximation, we can design features that do share information.

---

## 6. Linear Function Approximation

In linear function approximation, we choose a feature map

$$
\phi:\mathcal S\times\mathcal A\to \mathbb R^d.
$$

For each state-action pair, the feature vector is

$$
\phi(s,a)
=
\begin{bmatrix}
\phi_1(s,a)\\
\phi_2(s,a)\\
\vdots\\
\phi_d(s,a)
\end{bmatrix}.
$$

The approximate action-value function is

$$
Q_\theta(s,a)=\theta^T \phi(s,a),
$$

where

$$
\theta\in\mathbb R^d.
$$

Equivalently,

$$
Q_\theta(s,a)
=
\sum_{j=1}^d \theta_j\phi_j(s,a).
$$

The value is a weighted combination of features. The features describe the state-action pair, and the parameters decide how important each feature is.

For example, in a robot navigation problem, the features may include distance to the goal, distance to obstacles, current speed, angle to the target, or energy consumption. In a gridworld, features may include row position, column position, whether the state is near a wall, or whether the goal is nearby.

The same parameter vector $$\theta$$ is used for all state-action pairs. Therefore, if two state-action pairs have similar features, then their predicted values will also be related.

This is the mechanism of generalization.

---

## 7. Linear TD Learning for Policy Evaluation

Let us first consider policy evaluation. The policy $$\pi$$ is fixed, and the goal is to approximate $$Q^\pi$$.

Given a transition

$$
(s_t,a_t,r_t,s_{t+1},a_{t+1}),
$$

the TD target is

$$
r_t+
\gamma Q_\theta(s_{t+1},a_{t+1}).
$$

The TD error is

$$
\delta_t(\theta)
=
r_t+
\gamma Q_\theta(s_{t+1},a_{t+1})
-
Q_\theta(s_t,a_t).
$$

Using the linear form,

$$
Q_\theta(s_t,a_t)=\theta^T\phi(s_t,a_t),
$$

and

$$
Q_\theta(s_{t+1},a_{t+1})=\theta^T\phi(s_{t+1},a_{t+1}).
$$

Hence

$$
\delta_t(\theta)
=
r_t
+
\gamma\theta^T\phi(s_{t+1},a_{t+1})
-
\theta^T\phi(s_t,a_t).
$$

The semi-gradient TD update is

$$
\theta_{t+1}
=
\theta_t
+
\eta_t\delta_t(\theta_t)\phi(s_t,a_t).
$$

This looks almost identical to the tabular TD update, but there is a crucial difference. In tabular RL, one update changes one entry of the table. In linear function approximation, one update changes the entire parameter vector $$\theta$$. Therefore, it can change the value estimate at many state-action pairs simultaneously.

Indeed, after the update, for any other pair $$(s,a)$$,

$$
Q_{\theta_{t+1}}(s,a)
=
\theta_{t+1}^\top\phi(s,a).
$$

So if $$\phi(s,a)$$ overlaps with $$\phi(s_t,a_t)$$, the value at $$(s,a)$$ also changes.

That is exactly what function approximation is supposed to do: it transfers information across related states.

---

## 8. Projection and Approximation Error

There is an important difference between the tabular and linear settings.

In the tabular case, the Bellman operator maps tables to tables. So the fixed-point equation

$$
Q^\pi=T^\pi Q^\pi
$$

has a solution inside the tabular space.

But with linear approximation, we restrict ourselves to

$$
\mathcal F
=
\{Q_\theta:Q_\theta(s,a)=\theta^T\phi(s,a)\}.
$$

Even if $$Q_\theta\in\mathcal F$$, the Bellman update $$T^\pi Q_\theta$$ may not lie in $$\mathcal F$$.

So the exact Bellman equation may not have a solution inside the approximation class.

Instead, linear TD methods are often understood through a projected Bellman equation:

$$
Q_\theta = \Pi T^\pi Q_\theta,
$$

where $$\Pi$$ is a projection operator onto the function class $$\mathcal F$$ under a suitable norm.

This means:

1. Apply the Bellman operator.
2. The result may leave the linear function class.
3. Project it back onto the linear function class.

This is one of the key mathematical differences introduced by function approximation.

It also creates approximation error. If $$Q^\pi$$ is not representable by the features, then no parameter vector can recover it exactly. The best possible error is

$$
\inf_\theta \|Q_\theta-Q^\pi\|.
$$

Thus, even with infinite data and perfect optimization, a poor feature representation may still give a poor approximation.

---

## 9. Linear Approximation for Control

For control, we want to approximate $$Q^\star$$. The approximate $$Q$$-Learning target is

$$
r_t+
\gamma\max_{a'\in\mathcal A}Q_\theta(s_{t+1},a').
$$

The TD error becomes

$$
\delta_t(\theta)
=
r_t
+
\gamma\max_{a'}Q_\theta(s_{t+1},a')
-
Q_\theta(s_t,a_t).
$$

With linear approximation,

$$
Q_\theta(s,a)=\theta^T\phi(s,a),
$$

so

$$
\delta_t(\theta)
=
r_t
+
\gamma\max_{a'}\theta^T\phi(s_{t+1},a')
-
\theta^T\phi(s_t,a_t).
$$

The semi-gradient update is

$$
\theta_{t+1}
=
\theta_t
+
\eta_t\delta_t(\theta_t)\phi(s_t,a_t).
$$

This is the natural linear-function-approximation version of tabular $$Q$$-Learning.

However, the theory becomes more delicate. In the tabular setting, the Bellman optimality operator is a contraction in the sup norm:

$$
\|T^\star Q_1-T^\star Q_2\|_\infty
\leq
\gamma\|Q_1-Q_2\|_\infty.
$$

With function approximation, the algorithm no longer simply applies this contraction inside a closed finite-dimensional table space. It performs bootstrapped stochastic updates through parameters, possibly with projection and approximation error. This is where much of the difficulty of RL with function approximation begins.

---

## 10. From Linear Approximation to Neural Networks

Linear approximation is powerful when good features are available. But in many problems, designing good features is hard.

For example, suppose the state is an image. It is not obvious how to manually construct features that capture the important objects, their locations, and their relevance to future reward.

Neural networks address this by learning the representation from data.

Instead of hand-designing a feature map $$\phi(s,a)$$, we use a neural network:

$$
Q_\theta(s,a)=\text{NN}_\theta(s,a).
$$

The parameter vector $$\theta$$ contains all weights and biases of the network.

The move from linear approximation to neural approximation can be thought of as replacing fixed features by learned nonlinear features.

In linear approximation, we write

$$
Q_\theta(s,a)=\theta^T\phi(s,a),
$$

where $$\phi$$ is chosen by us.

In neural approximation, the network internally constructs a feature representation. A two-layer neural network makes this especially clear.

---

## 11. A Two-Layer Neural Network for $$Q$$-Function Approximation

Let $$x(s,a)\in\mathbb R^d$$ be an input representation of the state-action pair. This could be a raw vector, a hand-crafted feature vector, or an embedding of the state and action.

A two-layer neural network with one hidden layer can be written as

$$
h_\theta(s,a)
=
\sigma(W_1x(s,a)+b_1),
$$

and

$$
Q_\theta(s,a)
=
w_2^\top h_\theta(s,a)+b_2.
$$

Here:

$$
W_1\in\mathbb R^{m\times d},
\qquad
b_1\in\mathbb R^m,
$$

$$
w_2\in\mathbb R^m,
\qquad
b_2\in\mathbb R,
$$

and $$\sigma$$ is a nonlinear activation function, such as ReLU:

$$
\sigma(z)=\max\{z,0\}.
$$

The hidden representation is

$$
h_\theta(s,a)
=
\begin{bmatrix}
\sigma(w_{1,1}^\top x(s,a)+b_{1,1})\\
\sigma(w_{1,2}^\top x(s,a)+b_{1,2})\\
\vdots\\
\sigma(w_{1,m}^\top x(s,a)+b_{1,m})
\end{bmatrix}.
$$

Then the output is

$$
Q_\theta(s,a)
=
\sum_{j=1}^m w_{2,j}h_{\theta,j}(s,a)+b_2.
$$

The important point is that the hidden layer produces learned features:

$$
h_\theta(s,a)=\sigma(W_1x(s,a)+b_1).
$$

So a two-layer network can be viewed as

$$
Q_\theta(s,a)
=
w_2^\top \underbrace{\sigma(W_1x(s,a)+b_1)}_{\text{learned features}}+b_2.
$$

This shows the connection with linear approximation. Linear approximation uses fixed features:

$$
Q_\theta(s,a)=\theta^T\phi(s,a).
$$

A two-layer neural network uses learned features:

$$
Q_\theta(s,a)=w_2^\top h_\theta(s,a)+b_2.
$$

The first layer learns the feature map, and the second layer linearly combines those learned features.

---

## 12. Two-Layer Network for Discrete Actions

For discrete action spaces, it is common to feed only the state into the network and output one value per action.

Let $$x(s)\in\mathbb R^d$$ be the state representation. A two-layer $$Q$$-network can be written as

$$
h_\theta(s)=\sigma(W_1x(s)+b_1),
$$

and

$$
Q_\theta(s,\cdot)=W_2h_\theta(s)+b_2.
$$

Here

$$
W_2\in\mathbb R^{|\mathcal A|\times m},
\qquad
b_2\in\mathbb R^{|\mathcal A|}.
$$

The output vector is

$$
Q_\theta(s,\cdot)
=
\begin{bmatrix}
Q_\theta(s,a_1)\\
Q_\theta(s,a_2)\\
\vdots\\
Q_\theta(s,a_{|\mathcal A|})
\end{bmatrix}.
$$

Then the greedy action is

$$
a^\star(s)
\in
\arg\max_{a\in\mathcal A} Q_\theta(s,a).
$$

This architecture is commonly used in Deep $$Q$$-Networks. The network receives the state and returns a vector of action-values.

This is efficient because one forward pass computes the values of all actions.

---

## 13. Neural-Network Bellman Loss

The Bellman optimality equation is still the guiding principle:

$$
Q^\star(s,a)
=
r(s,a)
+
\gamma
\mathbb E
\left[
\max_{a'}Q^\star(s',a')\mid s,a
\right].
$$

With a neural network, we try to make $$Q_\theta$$ approximately satisfy this equation.

Given a transition

$$
(s_t,a_t,r_t,s_{t+1}),
$$

the one-step target is

$$
y_t
=
r_t+
\gamma\max_{a'}Q_\theta(s_{t+1},a').
$$

The squared Bellman error loss is

$$
\mathcal L_t(\theta)
=
\left(
Q_\theta(s_t,a_t)-y_t
\right)^2.
$$

A semi-gradient update treats the target as fixed while differentiating. The TD error is

$$
\delta_t
=
y_t-Q_\theta(s_t,a_t).
$$

Then the update is

$$
\theta_{t+1}
=
\theta_t+
\eta_t\delta_t
\nabla_\theta Q_\theta(s_t,a_t).
$$

This is the neural-network analogue of the tabular update.

Compare the three cases:

Tabular update:

$$
Q_{t+1}(s_t,a_t)
=
Q_t(s_t,a_t)+\eta_t\delta_t.
$$

Linear update:

$$
\theta_{t+1}
=
\theta_t+
\eta_t\delta_t\phi(s_t,a_t).
$$

Neural-network update:

$$
\theta_{t+1}
=
\theta_t+
\eta_t\delta_t\nabla_\theta Q_\theta(s_t,a_t).
$$

For a linear model,

$$
\nabla_\theta Q_\theta(s_t,a_t)=\phi(s_t,a_t).
$$

For a neural network, this gradient is computed by backpropagation.

---

## 14. Writing the Two-Layer Network Update Explicitly

For the state-action input version, recall that

$$
Q_\theta(s,a)
=
w_2^\top \sigma(W_1x(s,a)+b_1)+b_2.
$$

Define

$$
z_t=W_1x(s_t,a_t)+b_1,
$$

and

$$
h_t=\sigma(z_t).
$$

Then

$$
Q_\theta(s_t,a_t)=w_2^\top h_t+b_2.
$$

The TD target is

$$
y_t
=
r_t+
\gamma\max_{a'}Q_\theta(s_{t+1},a').
$$

The TD error is

$$
\delta_t=y_t-Q_\theta(s_t,a_t).
$$

Using the squared loss

$$
\mathcal L_t(\theta)
=
\frac12\left(Q_\theta(s_t,a_t)-y_t\right)^2,
$$

the semi-gradient update is

$$
\theta_{t+1}=\theta_t-\eta_t\nabla_\theta \mathcal L_t(\theta_t).
$$

Because

$$
\nabla_\theta \mathcal L_t(\theta_t)
=
-\delta_t\nabla_\theta Q_\theta(s_t,a_t),
$$

we get

$$
\theta_{t+1}
=
\theta_t+
\eta_t\delta_t\nabla_\theta Q_\theta(s_t,a_t).
$$

For the final-layer weights,

$$
\nabla_{w_2}Q_\theta(s_t,a_t)=h_t,
$$

so

$$
w_{2,t+1}=w_{2,t}+\eta_t\delta_t h_t.
$$

For the final-layer bias,

$$
\nabla_{b_2}Q_\theta(s_t,a_t)=1,
$$

so

$$
b_{2,t+1}=b_{2,t}+\eta_t\delta_t.
$$

For the first-layer parameters, backpropagation gives

$$
\nabla_{W_1}Q_\theta(s_t,a_t)
=
\left(w_2\odot \sigma'(z_t)\right)x(s_t,a_t)^\top,
$$

and

$$
\nabla_{b_1}Q_\theta(s_t,a_t)
=
w_2\odot \sigma'(z_t),
$$

where $$\odot$$ denotes elementwise multiplication.

Therefore,

$$
W_{1,t+1}
=
W_{1,t}
+
\eta_t\delta_t
\left(w_{2,t}\odot \sigma'(z_t)\right)x(s_t,a_t)^\top,
$$

and

$$
b_{1,t+1}
=
b_{1,t}
+
\eta_t\delta_t
\left(w_{2,t}\odot \sigma'(z_t)\right).
$$

This makes the neural-network update concrete. The TD error tells us the direction in which the value prediction should move, and backpropagation distributes that correction across the network parameters.

---

## 15. DQN: Stabilizing Neural $$Q$$-Learning

Naively combining neural networks with $$Q$$-Learning can be unstable. The reason is that the target itself depends on the same network being updated.

If

$$
y_t=r_t+
\gamma\max_{a'}Q_\theta(s_{t+1},a'),
$$

then every change in $$\theta$$ also changes the target. The network is chasing a moving target.

Deep $$Q$$-Networks use two important stabilizing ideas.

### Experience Replay

The agent stores transitions in a replay buffer:

$$
\mathcal D
=
\{(s_i,a_i,r_i,s_i')\}.
$$

Instead of updating only from the most recent transition, the algorithm samples a mini-batch from $$\mathcal D$$ and minimizes

$$
\mathcal L(\theta)
=
\frac1B\sum_{i=1}^B
\left(
Q_\theta(s_i,a_i)-y_i
\right)^2.
$$

This reduces the correlation between consecutive updates and makes the learning process closer to supervised learning.

### Target Network

DQN also uses a separate target network with parameters $$\theta^-$$. The target is

$$
y_i
=
r_i+
\gamma\max_{a'}Q_{\theta^-}(s_i',a').
$$

The loss becomes

$$
\mathcal L(\theta)
=
\frac1B\sum_{i=1}^B
\left(
Q_\theta(s_i,a_i)
-
\left[
r_i+
\gamma\max_{a'}Q_{\theta^-}(s_i',a')
\right]
\right)^2.
$$

The online network $$Q_\theta$$ is updated frequently, while the target network $$Q_{\theta^-}$$ is updated more slowly, for example by periodically setting

$$
\theta^-\leftarrow \theta.
$$

This makes the target more stable and improves training.

---

## 16. Function Approximation for Policies

Function approximation is not only used for value functions. It is also used for policies.

Instead of learning $$Q_\theta$$, we may directly learn a policy

$$
\pi_\theta(a\mid s).
$$

For discrete action spaces, a neural network can output action probabilities through a softmax layer:

$$
\pi_\theta(a\mid s)
=
\frac{\exp(f_\theta(s,a))}{\sum_{b\in\mathcal A}\exp(f_\theta(s,b))}.
$$

The objective is to maximize the expected return:

$$
J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)
\right].
$$

Policy-gradient methods update $$\theta$$ using an estimate of

$$
\nabla_\theta J(\theta).
$$

A common form is

$$
\nabla_\theta J(\theta)
=
\mathbb E_{\pi_\theta}
\left[
\sum_{t=0}^{\infty}
\nabla_\theta\log\pi_\theta(a_t\mid s_t)G_t
\right],
$$

where

$$
G_t=\sum_{k=t}^{\infty}\gamma^{k-t}r(s_k,a_k).
$$

Actor-critic methods combine both ideas. The actor is the policy

$$
\pi_\theta(a\mid s),
$$

and the critic is a value function such as

$$
V_w(s)
\qquad\text{or}\qquad
Q_w(s,a).
$$

The critic estimates how good the current behavior is, and the actor uses that estimate to improve the policy.

---

## 17. The Deadly Triad

Function approximation gives generalization, but it also introduces instability. A famous warning in RL is the deadly triad:

1. Function approximation.
2. Bootstrapping.
3. Off-policy learning.

Bootstrapping means that the target uses the current value estimate, as in

$$
r_t+
\gamma Q_\theta(s_{t+1},a_{t+1})
$$

or

$$
r_t+
\gamma\max_{a'}Q_\theta(s_{t+1},a').
$$

Off-policy learning means the data is generated by one policy, while the algorithm evaluates or improves another policy.

Function approximation means that changing $$\theta$$ changes the values of many states at once.

Individually, each ingredient is useful. Together, they can cause instability or divergence.

The intuition is simple. Suppose we update $$Q_\theta(s_t,a_t)$$ using a target that depends on $$Q_\theta(s_{t+1},a')$$. Since both quantities depend on the same parameters, changing $$\theta$$ to fix one value may accidentally distort the other. Then future targets change as well. This creates a feedback loop.

In tabular RL, an update is local. In function approximation, an update is global.

That global coupling is the source of both power and difficulty.

---

## 18. A Unified View

The progression from tabular RL to linear approximation to neural networks can be summarized as follows.

In tabular RL, we store one value per state-action pair:

$$
Q(s,a)=\theta_{s,a}.
$$

Equivalently, this is linear approximation with one-hot features:

$$
Q_\theta(s,a)=\theta^T\phi(s,a).
$$

In linear function approximation, we use meaningful low-dimensional features:

$$
Q_\theta(s,a)=\theta^T\phi(s,a),
\qquad
\phi(s,a)\in\mathbb R^d.
$$

In neural function approximation, we learn nonlinear features:

$$
Q_\theta(s,a)
=
w_2^\top\sigma(W_1x(s,a)+b_1)+b_2.
$$

So the path is

$$
\text{tables}
\quad\longrightarrow\quad
\text{fixed features}
\quad\longrightarrow\quad
\text{learned features}.
$$

The Bellman principle remains the same at each stage. We want the estimate to be consistent with a bootstrapped target.

Tabular $$Q$$-Learning:

$$
Q(s_t,a_t)
\leftarrow
Q(s_t,a_t)+\eta_t\delta_t.
$$

Linear approximation:

$$
\theta
\leftarrow
\theta+\eta_t\delta_t\phi(s_t,a_t).
$$

Neural approximation:

$$
\theta
\leftarrow
\theta+\eta_t\delta_t\nabla_\theta Q_\theta(s_t,a_t).
$$

The TD error has the same conceptual form:

$$
\delta_t
=
r_t+
\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t).
$$

Only the representation changes.

---

## 19. Why Function Approximation Is Central to Modern RL

Function approximation is what allows RL to move beyond small finite MDPs.

Without function approximation, RL is mostly limited to tabular problems. With function approximation, RL can handle large state spaces, continuous states, image observations, robotic systems, games, recommender systems, and many other complex environments.

But this power comes with a cost.

In tabular RL, the mathematical structure is clean. The Bellman operator is a contraction, and many algorithms have strong convergence guarantees.

With function approximation, especially nonlinear approximation, the learning dynamics become much more complicated. The algorithm must simultaneously deal with sampling, bootstrapping, optimization, representation learning, exploration, and distribution shift.

This is why function approximation is one of the central themes of modern RL.

The Bellman equation still gives the target:

$$
Q^\star=T^\star Q^\star.
$$

But after introducing a parameterized approximation

$$
Q_\theta,
$$

the problem is no longer just a finite-dimensional fixed-point problem. It becomes a statistical optimization problem with self-generated targets.

In supervised learning, labels are usually fixed. In RL, the target often depends on the current model:

$$
y_t=r_t+
\gamma\max_{a'}Q_\theta(s_{t+1},a').
$$

This self-referential structure is one of the deepest and most beautiful aspects of RL.

---

## 20. Final Takeaway

Function approximation in RL means representing value functions, policies, or models using parameterized functions instead of tables.

The tabular case stores one number for every state-action pair:

$$
Q(s,a).
$$

Linear function approximation uses fixed features:

$$
Q_\theta(s,a)=\theta^T\phi(s,a).
$$

Neural function approximation uses learned nonlinear features. For example, a two-layer network gives

$$
Q_\theta(s,a)
=
w_2^\top\sigma(W_1x(s,a)+b_1)+b_2.
$$

The motivation is simple:

**We cannot store and visit every possible state-action pair, so we need representations that generalize.**

The tabular setting is clean but limited. Linear approximation gives a bridge from tables to generalization. Neural networks provide expressive learned representations powerful enough for high-dimensional problems.

So the story of function approximation in RL is the story of moving from exact storage to learned representations:

$$
\text{one value per state-action pair}
\quad\longrightarrow\quad
\text{linear features}
\quad\longrightarrow\quad
\text{neural representations}.
$$

At every stage, the Bellman equation remains the guiding idea. But the representation becomes richer, the generalization becomes stronger, and the mathematics becomes deeper.
