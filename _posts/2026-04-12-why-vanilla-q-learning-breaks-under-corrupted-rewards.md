---
title: Why Vanilla Q-Learning Breaks Under Corrupted Rewards
date: 2026-04-12 19:15:00 -0400
categories: [rl-blogs]
tags: [reinforcement-learning, robust-rl, q-learning, corruption, theory]
series: Adversarially Robust RL Series
---

Reinforcement learning is often advertised as a framework for learning to act optimally from interaction alone. Among all RL algorithms, **Q-learning** occupies a special place. It is simple, model-free, elegant, and backed by decades of theory. Under appropriate assumptions, it converges to the optimal action-value function even when the environment is unknown.

That story is compelling. But it hides a fragile assumption.

Vanilla Q-learning quietly assumes that the feedback it receives is essentially trustworthy. The rewards may be noisy, but they are still assumed to be informative about the underlying environment. In many practical settings, that assumption is unrealistic. Rewards may be corrupted by sensor failures, communication glitches, logging errors, distributional mismatch, adversarial manipulation, or various forms of model misspecification. Once that happens, the classical Q-learning update can become deeply unreliable.

This post explains why.

The punchline is simple: **vanilla Q-learning breaks under corrupted rewards because it has no mechanism to distinguish informative feedback from malicious or contaminated feedback.** It averages everything into its update target. As a result, it may converge to the wrong value function, and in the worst case, even learn the wrong policy.

## 1. The clean picture: why Q-learning works in the first place

Let us begin with the standard setup.

We consider a discounted Markov decision process with finite state space \(\mathcal S\), finite action space \(\mathcal A\), discount factor \(\gamma \in (0,1)\), reward function \(r(s,a)\), and transition kernel \(P(\cdot \mid s,a)\).

The object of interest is the optimal action-value function \(Q^\star\), which satisfies the **Bellman optimality equation**

\[
Q^\star(s,a)
=
r(s,a)
+
\gamma \sum_{s'} P(s' \mid s,a)\max_{a'} Q^\star(s',a').
\]

If the model is known, one can in principle compute \(Q^\star\) by repeatedly applying the Bellman optimality operator

\[
(\mathcal T Q)(s,a)
=
r(s,a)
+
\gamma \sum_{s'} P(s' \mid s,a)\max_{a'} Q(s',a').
\]

The reason this works is that \(\mathcal T\) is a \(\gamma\)-contraction in the sup norm. So repeated application drives us toward its unique fixed point \(Q^\star\).

Q-learning is a stochastic approximation version of this idea. When the agent visits \((s_t,a_t)\), observes reward \(R_t\), and transitions to \(s_{t+1}\), the update is

\[
Q_{t+1}(s_t,a_t)
=
(1-\alpha_t)Q_t(s_t,a_t)
+
\alpha_t
\Bigl(
R_t + \gamma \max_{a'} Q_t(s_{t+1},a')
\Bigr).
\]

Everything else remains unchanged.

At a high level, the term

\[
R_t + \gamma \max_{a'} Q_t(s_{t+1},a')
\]

is an empirical proxy for \((\mathcal T Q_t)(s_t,a_t)\). If the samples are generated properly and the reward observations are honest, then Q-learning tracks the Bellman operator and converges to \(Q^\star\).

That is the clean story.

## 2. Where the clean story begins to fail

The update above uses the observed reward \(R_t\) directly. This is the first point of vulnerability.

In the standard theory, the reward is modeled as something like

\[
R_t = r(s_t,a_t) + \xi_t,
\]

where \(\xi_t\) is a mean-zero noise term. The noise can slow learning, but as long as it is sufficiently well-behaved, repeated averaging eventually helps. The randomness washes out.

Now suppose the learner instead observes

\[
R_t = r(s_t,a_t) + \xi_t + c_t,
\]

where \(c_t\) is a corruption term.

This corruption is not required to be small. It is not required to be mean-zero. It is not required to be random. It could be chosen adversarially. It could depend on the time index, the current state-action pair, or even the current values of the learner. In short, it can be actively hostile.

Vanilla Q-learning does not know any of this. It simply inserts the observed \(R_t\) into the update as if it were reliable.

That is the entire problem.

## 3. Noise is not the same as corruption

It is worth slowing down here, because this distinction is easy to underappreciate.

Ordinary stochastic noise is not fundamentally incompatible with Q-learning. In fact, the algorithm was built to handle randomness. If the reward observations fluctuate but are still centered around the truth, repeated visits help stabilize the estimate.

Corruption is different.

A corruption term introduces **bias**, not merely variability. If the bias persists over time, then the update target itself becomes systematically wrong. The learner is no longer receiving noisy information about the correct Bellman operator. It is receiving misleading information about a different one.

So the issue is not that the estimates become more variable. The issue is that the algorithm begins tracking the wrong target altogether.

## 4. The real failure mechanism: Q-learning follows the wrong operator

This is the most important conceptual point.

In the clean setting, Q-learning tracks the Bellman optimality operator \(\mathcal T\), whose fixed point is the true optimal action-value function \(Q^\star\).

Under corrupted rewards, the learner effectively sees a modified reward function. Call it \(\widetilde r(s,a)\). Then the implicit operator being tracked becomes

\[
(\widetilde{\mathcal T}Q)(s,a)
=
\widetilde r(s,a)
+
\gamma \sum_{s'} P(s' \mid s,a)\max_{a'} Q(s',a').
\]

This is still a contraction. So the algorithm may still converge.

But it converges to the fixed point of \(\widetilde{\mathcal T}\), not the fixed point of \(\mathcal T\).

That distinction is fatal.

The learner may look stable. The iterates may settle down. The Q-values may stop oscillating. But all of that can happen while the algorithm is converging to the wrong answer.

In other words, corruption can destroy correctness without destroying apparent convergence.

That is one reason the problem is subtle. The failure is not always dramatic instability. Often, it is **stable convergence to a distorted value function**.

## 5. A toy picture: how a small manipulation can flip a decision

Let us make this more concrete.

Imagine a state \(s\) with two actions: Left and Right. In the true environment, Left is slightly better than Right. So under the true value function,

\[
Q^\star(s,\text{Left}) > Q^\star(s,\text{Right}).
\]

Now suppose the reward observations for Right are repeatedly corrupted upward. Perhaps not on every round, but frequently enough. Vanilla Q-learning takes those inflated rewards seriously. Over time, the estimate \(Q_t(s,\text{Right})\) begins to drift upward.

Eventually it may cross the estimate for Left. Once that happens, the learner believes the wrong action is optimal.

At that point, the damage does not remain local. Since Q-learning is bootstrapped, future targets involve \(\max_{a'}Q_t(s',a')\), which means the incorrect estimate propagates forward into future updates. The error starts to reinforce itself.

So a corruption at the reward level can alter the learned action-value function enough to **change the induced policy**. The learner does not merely estimate the wrong numbers; it starts making the wrong decisions.

That is the core danger in control settings.

## 6. Why bootstrapping amplifies the damage

Q-learning is not a simple averaging procedure. It is a recursive, bootstrapped method. The update target contains the term

\[
\gamma \max_{a'} Q_t(s_{t+1},a').
\]

That means today’s estimate depends partly on yesterday’s estimate. This is what makes Q-learning powerful. It can propagate information backward through time without solving the full model.

But the same mechanism becomes problematic under corruption.

If a corrupted reward pushes the Q-value in the wrong direction, then future updates inherit that distortion through the bootstrap term. The learner is no longer just using corrupted rewards. It is also using **its own previously corrupted beliefs**.

This creates a feedback loop.

In clean settings, bootstrapping accelerates learning. In hostile settings, it can amplify error.

That is one of the reasons robust RL is more delicate than robust supervised learning. In supervised learning, a bad label corrupts one training example. In Q-learning, a bad reward can influence an entire chain of downstream estimates.

## 7. Why simple averaging is too fragile

A natural question is: why not just collect enough data and average more carefully?

The problem is that vanilla Q-learning does not do any explicit outlier rejection or robust aggregation. It treats each new reward observation as legitimate. If a few observations are badly corrupted, they can have a disproportionate effect, especially when a state-action pair is visited infrequently or the corruption magnitude is large.

Even if the corruption fraction is small, the influence of those corrupted samples may not vanish. If the contamination is strategically placed, it can bias the long-run estimate away from the truth.

This is precisely where classical sample averaging fails: it is not robust to contamination. The sample mean is excellent in clean stochastic models, but fragile in the presence of adversarial outliers.

Since the Q-learning target directly uses observed rewards, vanilla Q-learning inherits that fragility.

## 8. The role of corruption magnitude

Another reason vanilla Q-learning is vulnerable is that it has no built-in control on extreme rewards.

Suppose an adversary occasionally injects a reward of enormous magnitude. Then even a single update can be significantly distorted. If such events occur repeatedly, the learned values can become arbitrarily wrong.

This means that the algorithm is sensitive not only to **how often** corruption occurs, but also to **how large** the corruption can be.

That is dangerous in practice. In a real system, corrupted feedback need not be subtle. It may arise from sensor spikes, numerical faults, overflow, communication tampering, or malicious interventions that produce highly abnormal reward values.

Vanilla Q-learning has no defense against that. It trusts the channel completely.

## 9. Why this matters in real systems

At first glance, corrupted rewards may sound like an artificial theoretical pathology. They are not.

Consider a few realistic scenarios:

- A self-driving system receives noisy or manipulated proxy rewards from a perception module.
- A networked control system logs incorrect performance measurements because of packet corruption or synchronization errors.
- A recommender system uses click-based rewards that can be gamed or spammed.
- A federated or distributed RL system aggregates feedback from multiple unreliable agents or devices.
- A robot learns from a sensor stack where some measurements are occasionally grossly incorrect.

In each of these cases, the learner is not living in a clean probabilistic universe. The reward channel itself may be compromised.

If the algorithm assumes that every observed reward is fundamentally reliable, it is exposed.

That is why reward corruption is not merely a technical nuisance. It is a trustworthiness issue.

## 10. Stability is not enough

One of the most misleading aspects of this problem is that a corrupted algorithm may still look numerically well-behaved.

The iterates may remain bounded. The curves may appear smooth. The training may “converge.”

But convergence by itself is not the right notion of success. The real question is: **converging to what?**

Under corruption, vanilla Q-learning may converge nicely to the fixed point of a perturbed Bellman operator. That is mathematically stable but decision-theoretically wrong.

This distinction matters especially in safety-critical applications. An algorithm that converges reliably to the wrong control law is not a robust learning system. It is simply a dependable way of making the wrong decision.

## 11. What robustness must change

Once one understands the failure mode, the need for robustification becomes obvious.

A robust RL algorithm must stop treating every reward sample as equally trustworthy. At a minimum, it should do two things.

### (a) Use a corruption-resistant reward estimator

Instead of directly plugging in the raw observed reward or its empirical mean, one should use a robust estimator such as:

- median-of-means,
- trimmed mean,
- clipped mean,
- or other contamination-resistant procedures.

These estimators are designed so that a small fraction of arbitrary outliers cannot dominate the estimate.

### (b) Control rare catastrophic deviations

Even robust estimators can occasionally fail on finite samples. So many robust RL procedures add another protective layer, such as:

- clipping,
- thresholding,
- rejection rules,
- or confidence-based safeguards.

The goal is to ensure that a single pathological estimate does not derail the entire recursion.

This is especially important in bootstrapped methods like Q-learning, where one bad update can contaminate future targets as well.

## 12. The deeper lesson

The deeper lesson is not merely that “Q-learning is imperfect.” Every algorithm is imperfect.

The real lesson is that **the classical Q-learning update is structurally aligned with stochastic uncertainty, not adversarial contamination**. It works beautifully when the randomness is honest. It is brittle when the randomness is hostile.

That is why robust reinforcement learning is not just about proving stronger theorems for the same old algorithm. It often requires changing the estimator, the update rule, the thresholds, and sometimes even the sampling architecture.

Once the feedback stream cannot be blindly trusted, learning itself must become suspicious.

## 13. Final takeaway

Vanilla Q-learning breaks under corrupted rewards because it has no mechanism to separate clean feedback from contaminated feedback. It plugs the observed reward directly into a bootstrapped update, thereby allowing corrupted observations to bias the Bellman target. As a result, the algorithm can end up tracking a perturbed operator rather than the true Bellman optimality operator. The outcome may be stable, but wrong.

And wrong in reinforcement learning does not merely mean an inaccurate estimate. It can mean the wrong action, the wrong policy, and the wrong long-term behavior.

That is exactly why robust Q-learning is necessary.

---

## What comes next

A natural next question is: can this failure be shown in a very small example?

Yes.

In the next post, I will discuss a toy MDP where reward corruption can actually flip the optimal action, and explain why that makes the vulnerability of vanilla Q-learning impossible to ignore.
