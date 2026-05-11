---
title: Why Vanilla Q-Learning Breaks Under Corrupted Rewards
date: 2026-04-12 19:15:00 -0400
categories: [rl-blogs]
tags: [reinforcement-learning, robust-rl, q-learning, corruption, theory]
series: Adversarially Robust RL Series
math: true
---

Vanilla Q-Learning is one of the most elegant algorithms in reinforcement learning. It is model-free, computationally simple, and in the clean setting its behavior is governed by a beautiful fixed-point argument: the algorithm tracks the Bellman optimality operator, and that operator is a contraction. This is exactly why Q-Learning converges to the optimal action-value function \(Q^\star\).

However, this clean picture changes dramatically when the observed rewards are corrupted. The issue is not the Bellman contraction itself; the issue is the empirical update used to estimate the Bellman target. If even a small fraction of observed rewards can be adversarially perturbed, then the ordinary sample average used inside vanilla Q-Learning can be driven arbitrarily far from the true mean. As a result, the Q-update may be pulled toward a completely wrong value, and the algorithm can suffer arbitrarily large estimation error.

The key vulnerability is therefore statistical. Vanilla Q-Learning treats all observed rewards as trustworthy and averages them directly. Under a strong-contamination model, an adversary can replace a small fraction of reward samples by arbitrary values. Since the sample mean is not robust to outliers, a few corrupted rewards can dominate the empirical Bellman update.

A natural robust alternative is to replace the fragile empirical mean with a robust estimator. Instead of constructing the Bellman target using the ordinary average of historical rewards, one can use trimmed means, median-of-means estimators, or other robust mean-estimation procedures. This leads to a robust empirical Bellman operator that remains stable even when a small fraction of feedback is adversarially corrupted.

At a high level, the resulting theory separates two sources of error. The first is the usual statistical error that appears even in the clean setting. The second is an unavoidable corruption-induced error that depends on the adversarial contamination level. A representative finite-time guarantee has the form
\[
\|Q_T - Q^\star\|_\infty
\;\lesssim\;
\text{clean statistical error}
+
\text{corruption penalty}.
\]
The important point is that the corruption penalty is not merely a weakness of the proof. Information-theoretic lower bounds show that some dependence on the corruption level is unavoidable.

This perspective suggests a simple but powerful message: robust reinforcement learning is not just about modifying the algorithm after the fact. It requires replacing the fragile statistical primitives inside Bellman updates with estimators that are designed to withstand corrupted feedback.
