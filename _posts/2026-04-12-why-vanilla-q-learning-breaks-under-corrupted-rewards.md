---
title: Why Vanilla Q-Learning Breaks Under Corrupted Rewards
date: 2026-04-12 19:15:00 -0400
categories: [rl-blogs]
tags: [reinforcement-learning, robust-rl, q-learning, corruption, theory]
series: Adversarially Robust RL Series
math: true
---

Vanilla Q-learning is one of the most elegant ideas in reinforcement learning. It learns the optimal action-value function without knowing the model, and in the clean setting its behavior is governed by a beautiful contraction argument. But that clean theory rests on a fragile premise: the reward observations must be informative about the true environment. Once rewards are corrupted, the update no longer tracks the Bellman operator of the underlying MDP. It tracks a *perturbed* operator instead. That is the core reason vanilla Q-learning breaks.

In this post I want to make that statement mathematically precise. The real issue is not merely that corrupted rewards add noise. Ordinary noise can often be averaged out. The problem is that corruption introduces *bias* into the Bellman target, and a non-vanishing bias changes the fixed point toward which the recursion is driven. In a bootstrapped algorithm like Q-learning, this bias is then amplified by the geometry of the Bellman operator, typically by a factor of order $$1/(1-\gamma)$$. That is exactly why even seemingly local reward corruption can lead to globally wrong decisions.

## 1. Clean Q-learning as stochastic approximation to the Bellman operator

Consider a discounted finite Markov decision process with state space $$\mathcal S$$, action space $$\mathcal A$$, reward function $$r:\mathcal S \times \mathcal A \to \mathbb R$$, transition kernel $$P(\cdot\mid s,a)$$, and discount factor $$\gamma \in (0,1)$$.

The Bellman optimality operator is

$$
(\mathcal T Q)(s,a)
=
r(s,a)
+
\gamma \sum_{s' \in \mathcal S} P(s' \mid s,a)\max_{a' \in \mathcal A} Q(s',a').
$$

The optimal action-value function $$Q^\star$$ is the unique fixed point of $$\mathcal T$$, namely

$$
Q^\star = \mathcal T Q^\star.
$$

The reason this fixed point is unique is that $$\mathcal T$$ is a $$\gamma$$-contraction in the sup norm:

$$
\|\mathcal T Q - \mathcal T Q'\|_\infty
\leq
\gamma \|Q-Q'\|_\infty.
$$

This one inequality is the backbone of essentially all classical Q-learning theory.

Now suppose the learner observes a trajectory $$\{(s_t,a_t,R_t,s_{t+1})\}_{t\ge 0}$$ generated under some exploratory behavior policy. The vanilla Q-learning update is

$$
Q_{t+1}(s_t,a_t)
=
(1-\alpha_t)Q_t(s_t,a_t)
+
\alpha_t
\left(
R_t + \gamma \max_{a'} Q_t(s_{t+1},a')
\right),
$$

with all other coordinates unchanged.

A very useful way to rewrite this is coordinatewise as

$$
Q_{t+1}(s,a)
=
Q_t(s,a)
+
\alpha_t \mathbf 1\{(s_t,a_t)=(s,a)\}
\left(
Y_t - Q_t(s,a)
\right),
$$

where the one-step target is

$$
Y_t
=
R_t + \gamma \max_{a'} Q_t(s_{t+1},a').
$$

In the clean setting, if the observed reward satisfies

$$
\mathbb E[R_t \mid s_t=s,a_t=a] = r(s,a),
$$

then conditional on the current iterate $$Q_t$$ and on visiting $$ (s,a) $$, we have

$$
\mathbb E[Y_t \mid \mathcal F_t,\, s_t=s, a_t=a]
=
(\mathcal T Q_t)(s,a).
$$

So the update is an asynchronous stochastic approximation to the fixed-point equation

$$
Q = \mathcal T Q.
$$

That is the exact reason vanilla Q-learning works when the feedback is trustworthy: the drift points toward $$Q^\star$$.

## 2. What corruption does to the update target

Now let us change only one thing. Suppose the learner no longer observes a clean reward, but instead observes

$$
R_t = r(s_t,a_t) + \xi_t + c_t,
$$

where $$\xi_t$$ is ordinary centered noise and $$c_t$$ is a corruption term.

The crucial point is that $$c_t$$ is not assumed to have mean zero. It may depend on time, on the current state-action pair, on the history, or even on the current estimate $$Q_t$$. In the clean theory, the one-step target decomposes into

$$
Y_t
=
(\mathcal T Q_t)(s_t,a_t)
+
\text{martingale noise}.
$$

Under corruption, the target instead becomes

$$
Y_t
=
(\mathcal T Q_t)(s_t,a_t)
+
c_t
+
\text{martingale noise}.
$$

So the update contains a *persistent bias term*.

At the language level, one says “corruption changes the target.” At the mathematical level, the statement is sharper: the recursion is no longer centered around the Bellman optimality operator of the true MDP.

To see this clearly, suppose for a moment that corruption induces an effective mean shift

$$
\delta(s,a)
=
\mathbb E[c_t \mid s_t=s, a_t=a].
$$

Then the conditional mean target becomes

$$
\mathbb E[Y_t \mid \mathcal F_t,\, s_t=s,a_t=a]
=
(\widetilde{\mathcal T}Q_t)(s,a),
$$

where the perturbed Bellman operator is

$$
(\widetilde{\mathcal T}Q)(s,a)
=
\widetilde r(s,a)
+
\gamma \sum_{s' \in \mathcal S} P(s' \mid s,a)\max_{a'} Q(s',a'),
$$

with

$$
\widetilde r(s,a) = r(s,a) + \delta(s,a).
$$

This is the main mathematical failure mechanism. Vanilla Q-learning does not suddenly become “unstable” in some mysterious way. It simply starts tracking the *wrong operator*.

## 3. The wrong fixed point is still a fixed point

A subtle but very important point is that $$\widetilde{\mathcal T}$$ is also a $$\gamma$$-contraction. So it has its own unique fixed point, say $$\widetilde Q^\star$$, satisfying

$$
\widetilde Q^\star = \widetilde{\mathcal T}\widetilde Q^\star.
$$

Thus vanilla Q-learning under corrupted rewards may still converge nicely, smoothly, and stably. But what it converges to is no longer the true optimum $$Q^\star$$. It converges to the optimum of a *fictitious MDP* whose reward function has been altered by the corruption.

This is why empirical stability can be misleading. The iterates may look perfectly well-behaved while the learned policy is fundamentally wrong.

So the correct question is not “does the algorithm converge?” The correct question is

$$
\text{converge to what?}
$$

Under corruption, the answer can be: to the fixed point of the wrong Bellman operator.

## 4. A clean perturbation bound

The contraction structure lets us quantify the damage very cleanly.

Let $$Q^\star$$ be the fixed point of $$\mathcal T$$ and $$\widetilde Q^\star$$ the fixed point of $$\widetilde{\mathcal T}$$. Then

$$
\|\widetilde Q^\star - Q^\star\|_\infty
=
\|\widetilde{\mathcal T}\widetilde Q^\star - \mathcal T Q^\star\|_\infty.
$$

Add and subtract $$\mathcal T \widetilde Q^\star$$ to get

$$
\|\widetilde Q^\star - Q^\star\|_\infty
\le
\|\widetilde{\mathcal T}\widetilde Q^\star - \mathcal T \widetilde Q^\star\|_\infty
+
\|\mathcal T \widetilde Q^\star - \mathcal T Q^\star\|_\infty.
$$

By contraction,

$$
\|\mathcal T \widetilde Q^\star - \mathcal T Q^\star\|_\infty
\le
\gamma \|\widetilde Q^\star - Q^\star\|_\infty.
$$

And the first term is exactly the reward perturbation size:

$$
\|\widetilde{\mathcal T}\widetilde Q^\star - \mathcal T \widetilde Q^\star\|_\infty
=
\|\widetilde r - r\|_\infty.
$$

So we obtain

$$
\|\widetilde Q^\star - Q^\star\|_\infty
\le
\|\widetilde r - r\|_\infty
+
\gamma \|\widetilde Q^\star - Q^\star\|_\infty,
$$

hence

$$
\|\widetilde Q^\star - Q^\star\|_\infty
\le
\frac{\|\widetilde r - r\|_\infty}{1-\gamma}.
$$

This one bound already explains a lot.

A reward bias of size $$\|\widetilde r-r\|_\infty$$ gets amplified by a factor $$1/(1-\gamma)$$ in the value function. So when $$\gamma$$ is close to one, even a small reward distortion can produce a large action-value distortion.

That is the first place where the “hardcore math” and the intuition line up perfectly: bootstrapping plus discounting creates a geometric amplification mechanism.

## 5. Why the factor $$1/(1-\gamma)$$ is not an artifact

Some readers see the bound

$$
\frac{\|\widetilde r-r\|_\infty}{1-\gamma}
$$

and think it is just a loose proof artifact. In many cases it is not. It reflects a real accumulation phenomenon.

To see this, consider the simplest possible MDP: one state, one action, self-looping forever. Let the clean reward be $$r$$ and the corrupted reward be $$\widetilde r = r+\delta$$. Then

$$
Q^\star = r + \gamma Q^\star
\quad\Longrightarrow\quad
Q^\star = \frac{r}{1-\gamma},
$$

and similarly

$$
\widetilde Q^\star = \frac{r+\delta}{1-\gamma}.
$$

So the difference is exactly

$$
\widetilde Q^\star - Q^\star = \frac{\delta}{1-\gamma}.
$$

There is no looseness here. The reward corruption is literally accumulated over an infinite discounted future, and each step inherits the same bias. That is the geometric origin of the factor $$1/(1-\gamma)$$.

## 6. A two-action example where the optimal action flips

Now let us see how corruption can change the policy, not just the values.

Consider an MDP with one state $$s$$ and two actions, $$L$$ and $$R$$. Suppose both actions deterministically return to the same state. Let the clean rewards be

$$
r(s,L) = \Delta,
\qquad
r(s,R) = 0,
$$

for some $$\Delta > 0$$.

Then the clean optimal Q-values are

$$
Q^\star(s,L) = \frac{\Delta}{1-\gamma},
\qquad
Q^\star(s,R) = \frac{0}{1-\gamma} = 0.
$$

So action $$L$$ is optimal.

Now suppose the reward for action $$R$$ is corrupted upward by a constant amount $$b>0$$, so that the learner effectively sees

$$
\widetilde r(s,R) = b,
\qquad
\widetilde r(s,L)=\Delta.
$$

Then under the perturbed MDP,

$$
\widetilde Q^\star(s,L) = \frac{\Delta}{1-\gamma},
\qquad
\widetilde Q^\star(s,R) = \frac{b}{1-\gamma}.
$$

Therefore action $$R$$ becomes optimal whenever

$$
\frac{b}{1-\gamma} > \frac{\Delta}{1-\gamma},
$$

which is equivalent to

$$
b > \Delta.
$$

So a corruption of size larger than the reward gap flips the optimal action.

Notice something important: the factor $$1/(1-\gamma)$$ cancels when comparing the two actions in this toy example. The policy flip is driven by whether corruption overcomes the *advantage gap*. This is the right way to think about robustness in decision-making: values may distort by order $$1/(1-\gamma)$$, but a policy changes once the distortion is large relative to the local action gap.

## 7. Huber contamination viewpoint

Another mathematically useful way to describe corruption is through a Huber contamination model. For a fixed state-action pair $$ (s,a) $$, suppose the observed reward distribution is

$$
(1-\varepsilon)\mathbb P_{\text{clean}} + \varepsilon \mathbb P_{\text{adv}},
$$

where $$\mathbb P_{\text{clean}}$$ is the clean reward law and $$\mathbb P_{\text{adv}}$$ is an arbitrary adversarial distribution.

If the adversary places a point mass at a large value $$B$$, then the contaminated mean can shift by order

$$
\varepsilon B.
$$

Plugging that into the perturbation bound above yields a value distortion of order

$$
\frac{\varepsilon B}{1-\gamma}.
$$

This immediately shows two things.

First, if the corruption magnitude $$B$$ is unbounded, then the Q-error can be arbitrarily large even when $$\varepsilon$$ is small.

Second, even if $$B$$ is bounded, the corruption term does *not* vanish just because the sample size grows. It is a population-level bias, not a sampling fluctuation.

That is why simply collecting more data does not fix the problem. One needs an estimator that is itself robust to contamination.

## 8. The stochastic approximation decomposition

It is also useful to write the update in error form.

Let

$$
e_t = Q_t - Q^\star.
$$

In the clean setting, the update decomposes roughly as

$$
e_{t+1}
=
e_t
+
\alpha_t
\Bigl[
\text{Bellman contraction term}
+
\text{martingale noise}
\Bigr].
$$

The contraction part pushes the error toward zero, while the martingale noise averages out under standard step-size conditions.

Under corruption, the recursion becomes

$$
e_{t+1}
=
e_t
+
\alpha_t
\Bigl[
\text{Bellman contraction term}
+
\text{martingale noise}
+
\text{bias term}
\Bigr].
$$

The crucial difference is that the last term is not centered. It does not disappear by averaging. So the drift is no longer toward zero error relative to $$Q^\star$$. Instead, the recursion is pulled toward a shifted equilibrium.

This is the stochastic approximation version of the “wrong operator” story. Both perspectives say the same thing.

## 9. Why bootstrapping makes corruption worse than in supervised learning

In supervised learning, a corrupted label damages the fit at that sample. In Q-learning, a corrupted reward does more than that. It enters a target of the form

$$
R_t + \gamma \max_{a'}Q_t(s_{t+1},a'),
$$

and the second term already depends on previously learned values. Thus once corruption pushes part of the Q-function in the wrong direction, future updates are computed using that corrupted estimate. The error feeds into the next target, and then into the next one again.

So the harm of reward corruption is recursive.

This is why a single bad reward observation can have a longer shadow in RL than in classical prediction problems. The algorithm reuses its own past beliefs, and corruption can pollute those beliefs early.

## 10. Why sample averaging is the wrong tool

Vanilla Q-learning uses the raw reward observation itself. Implicitly, this is equivalent to trusting the sample mean as the right summary statistic for the reward channel.

But the sample mean is notoriously non-robust. A small fraction of arbitrarily large outliers can move it by an arbitrary amount.

That is mathematically the wrong estimator under adversarial contamination.

Once you see the update target as an estimator of a Bellman operator, the robust-RL prescription becomes almost forced: replace fragile empirical averages by robust estimators such as median-of-means, trimmed means, clipped estimates, or other contamination-resistant procedures. Then add thresholding or clipping to prevent rare estimator failures from destabilizing the bootstrap.

In other words, the correction is not cosmetic. The statistical core of the update must change.

## 11. The deepest lesson

The deepest lesson is that vanilla Q-learning is perfectly matched to *honest stochasticity* and poorly matched to *adversarial contamination*. The classical theory assumes that uncertainty enters as centered randomness around the truth. Under that model, contraction wins.

Under corruption, the uncertainty is not centered. The target itself is shifted. Contraction still wins, but it wins in the wrong direction: it drives the iterates toward the fixed point of a perturbed Bellman operator.

That is exactly why corruption is so dangerous. The same contraction mechanism that gives us convergence in the clean setting also gives us convergence to the wrong answer in the corrupted setting.

## 12. Final takeaway

Vanilla Q-learning breaks under corrupted rewards for a mathematically precise reason: the update no longer provides an unbiased stochastic approximation to the true Bellman optimality operator. Instead, corruption introduces a persistent bias into the Bellman target, which shifts the operator and hence shifts the fixed point. By contraction, the algorithm is then driven toward the optimum of a fictitious corrupted MDP rather than the true one. The induced value error is typically amplified by a factor of order $$1/(1-\gamma)$$, and once the distortion exceeds an action gap, the learned policy itself can flip.

That is why robustness in RL is not a minor refinement. It is a structural necessity whenever the reward channel cannot be blindly trusted.

---

## What comes next

The natural next step is to study the smallest possible example where one can *prove* an action flip and then connect that example to lower bounds in robust RL. That toy example is where the full force of the corruption problem becomes impossible to ignore.
