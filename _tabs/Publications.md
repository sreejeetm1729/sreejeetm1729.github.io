---
title: Publications
icon: fas fa-file-alt
order: 3
math: true
---

<style>
.pub-page {
  font-size: 16px;
  line-height: 1.65;
}

.pub-section {
  margin-bottom: 2rem;
}

.pub-entry {
  margin-bottom: 1.7rem;
}

.pub-title {
  font-weight: 700;
  font-size: 16px;
  line-height: 1.45;
  white-space: normal;
  overflow: visible;
}
.pub-authors {
  margin-top: 0.2rem;
}

.pub-venue {
  margin-top: 0.2rem;
  color: #d8a7a7;
  font-style: italic;
}

.pub-venue strong {
  color: inherit;
  font-weight: 700;
}

.pub-links a {
  text-decoration: none;
  font-weight: 700;
  margin-right: 0px;
  font-size: 14px;
  background-color: transparent !important;
  padding: 0 !important;
  border-radius: 0 !important;
}

.pub-summary {
  margin-top: 8px;
  padding: 10px 12px;
  border-left: 3px solid #e8c4bd;
  background: rgba(232, 196, 189, 0.08);
  font-size: 14px;
  line-height: 1.5;
}

.link-summary {
  color: #e8c4bd;
  font-weight: 700;
  font-size: 14px;
  background: transparent;
  border: none;
  padding: 0;
  margin: 0;
  cursor: pointer;
  font-family: inherit;
}

.link-summary:hover {
  color: #f3d7d2;
  text-decoration: underline;
}

.pub-summary-card {
  max-height: 0;
  opacity: 0;
  overflow: hidden;
  transform: translateY(-4px);
  transition: max-height 0.35s ease, opacity 0.25s ease, transform 0.25s ease;
  margin-top: 2px;
}

.pub-summary-card p {
  margin-top: 0;
}

.pub-summary-card.open {
  max-height: 2000px;
  opacity: 1;
  transform: translateY(-8px) !important;
}

.pub-summary-inner {
  margin-top: 0px;
  padding: 13px 15px;
  border-radius: 12px;
  border-left: 4px solid #f5e6b8;
  background: rgba(232, 196, 189, 0.08);
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.06);
}

.pub-summary-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 6px;
  font-size: 13px;
  font-weight: 800;
  color: #e8c4bd;
  text-transform: uppercase;
  letter-spacing: 0.03em;
}

.pub-summary-close {
  border: none;
  background: transparent;
  color: #999;
  font-size: 18px;
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.pub-summary-close:hover {
  color: #e8c4bd;
}

.pub-summary-text {
  font-size: 14px;
  line-height: 1.55;
}

.pub-summary-text strong {
  color: #e8c4bd;
}
/* smooth matte dark-mode palette */

.link-summary {
  color: #f5e6b8;
  background-color: rgba(5, 92, 9, 0.96);
}

.link-paper {
  color: #b7d3f2;
  background-color: rgba(100, 125, 236, 0.13);
}

.link-proceedings {
  color: #cbbff0;
  background-color: rgba(203, 191, 240, 0.13);
}

.link-poster {
  color: #e8bfd2;
  background-color: rgba(192, 158, 174, 0.13);
}

.link-slides {
  color: #bfe3cf;
  background-color: rgba(230, 92, 12, 0.13);
}

.link-code {
  color: #b8e3e8;
  background-color: rgba(19, 199, 219, 0.13);
}

.link-project {
  color: #e8c4bd;
  background-color: rgba(232, 196, 189, 0.13);
}

.link-workshop {
  color: #a5f3fc;
  text-decoration: none;
  font-weight: 700;
  background-color: transparent;
  padding: 0;
  border-radius: 0;
}

.link-workshop:hover {
  text-decoration: underline;
  filter: none;
}

.pub-links a:hover,
.link-workshop:hover {
  filter: brightness(1.15);
  text-decoration: none;
}

.workshop-entry {
  margin-bottom: 0.8rem;
}
</style>

<div class="pub-page">

<div class="pub-note">
  <strong>Comments on Publication Venues.</strong>
  The category “Selective ML/CS Conferences (M)” refers to papers published in highly selective machine learning and computer science venues, such as <em>NeurIPS</em>, <em>ICML</em>, <em>ICLR</em>, and <em>AISTATS</em>, typically with acceptance rates below 25%. These tend to be long papers of at least the same quality as typical journal papers in quality and rigor. In control theory, the premier journals (J) are <em>IEEE Transactions on Automatic Control</em> and <em>Automatica</em>, while the main flagship conferences (C) are the <em>IEEE Conference on Decision and Control</em> and the <em>American Control Conference</em>.
</div>

<div class="pub-section">

<h2><span style="font-size: 1rem;">♦️</span> Selected First-Authored Publications</h2>

<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(M_2\)]</span> Corruption-Tolerant Optimal Asynchronous Q-Learning</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Aritra Mitra</div>
  <div class="pub-venue">International Conference on Machine Learning, <strong>ICML 2026</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('icml2026-summary')">Summary</a>]
    [<a class="link-paper" href="https://arxiv.org/pdf/2509.08933">Paper</a>]
    [<a class="link-poster" href="https://icml.cc/media/PosterPDFs/ICML%202026/64666.png?t=1779063413.0709436">Poster</a>]
    [<a class="link-slides" href="https://icml.cc/media/icml-2026/Slides/64666.pdf">Slides</a>]
    [<a class="link-code" href="https://github.com/sreejeetm1729/Robust-Asynchronous-Q-Learning-with-Markovian-Data">Code</a>]
  </div>
</div>


<div id="icml2026-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('icml2026-summary')">×</button>
    </div>
    <div class="pub-summary-text">
      We study the problem of learning the optimal policy in a discounted, infinite-horizon reinforcement learning (RL) setting in the presence of adversarially corrupted rewards. To address this problem, we develop a novel robust variant of the Q-learning algorithm and analyze it under the challenging asynchronous sampling model with time-correlated data. Despite corruption, we prove that the finite-time guarantees of our approach match existing bounds, up to an additive term that scales with the fraction of corrupted samples. We also establish an information-theoretic lower bound, revealing that our guarantees are near-optimal. Notably, our algorithm is agnostic to the underlying reward distribution and provides the first finite-time robustness guarantees for asynchronous Q-learning. A key element of our analysis is a refined Azuma-Hoeffding inequality for almost-martingales, which may have broader applicability in the study of RL algorithms.
    </div>
  </div>
</div>

<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(M_1\)]</span> Adversarially-Robust TD Learning with Markovian Data: Finite-Time Rates and Fundamental Limits</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Aritra Mitra</div>
  <div class="pub-venue">International Conference on Artificial Intelligence and Statistics, <strong>AISTATS 2025</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('aistats2025-summary')">Summary</a>]
    [<a class="link-paper" href="https://arxiv.org/pdf/2502.04662">Paper</a>]
    [<a class="link-poster" href="https://virtual.aistats.org/media/PosterPDFs/AISTATS%202025/9390.png?t=1745618638.4746892">Poster</a>]
    [<a class="link-slides" href="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data/blob/main/AISTATS%20Slides.pdf">Slides</a>]
    [<a class="link-code" href="https://github.com/sreejeetm1729/Adversarially-Robust-TD-Learning-with-Markovian-Data">Code</a>]
  </div>
</div>
<div id="aistats2025-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('aistats2025-summary')">×</button>
    </div>
    <div class="pub-summary-text">
      One of the most basic problems in reinforcement learning (RL) is policy evaluation: estimating the long-term return, i.e., value function, corresponding to a given fixed policy. The celebrated Temporal Difference (TD) learning algorithm addresses this problem, and recent work has investigated finite-time convergence guarantees for this algorithm and variants thereof. However, these guarantees hinge on the reward observations being always generated from a well-behaved (e.g., sub-Gaussian) true reward distribution. Motivated by harsh, real-world environments where such an idealistic assumption may no longer hold, we revisit the policy evaluation problem from the perspective of adversarial robustness. In particular, we consider a Huber-contaminated reward model where an adversary can arbitrarily corrupt each reward sample with a small probability ϵ. Under this observation model, we first show that the adversary can cause the vanilla TD algorithm to converge to any arbitrary value function. We then develop a novel algorithm called Robust-TD and prove that its finite-time guarantees match that of vanilla TD with linear function approximation up to a small O(ϵ) term that captures the effect of corruption. We complement this result with a minimax lower bound, revealing that such an additive corruption-induced term is unavoidable. To our knowledge, these results are the first of their kind in the context of adversarial robustness of stochastic approximation schemes driven by Markov noise. The key new technical tool that enables our results is an analysis of the Median-of-Means estimator with corrupted, time-correlated data that might be of independent interest to the literature on robust statistics.
    </div>
  </div>
</div>
<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(C_4\)]</span> Robust Asynchronous Q-Learning under Reward and State Corruption via Batching</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Aritra Mitra</div>
  <div class="pub-venue">IEEE Conference on Decision and Control, <strong>CDC 2026</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('cdc2026-summary')">Summary</a>]
    [<a class="link-paper" href="www.example.com">Paper</a>]
    [<a class="link-poster" href="www.example.com">Poster</a>]
    [<a class="link-slides" href="www.example.com">Slides</a>]
    [<a class="link-code" href="www.example.com">Code</a>]
  </div>
</div>

<div id="cdc2026-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('cdc2024-summary')">×</button>
    </div>
    <div class="pub-summary-text">
       Motivated by reinforcement learning in harsh environments, we consider the problem of learning an optimal policy subject to adversarially corrupted feedback. Specifically, at each time-step, an adversary can perturb both the reward and state observations of the learner following the Huber contamination model. To defend against such data corruption, we propose BR-Async-Q: a novel, epoch-based, robust Q-learning algorithm built upon two key ideas: (i) partitioning the online data stream into batches to reduce variance, and (ii) constructing robust estimates of the Bellman optimality operator using such batched data. We prove a high-probability error bound for BR-Async-Q that matches that for vanilla Q-learning, up to a small additive term that scales with the fraction of corrupted samples. To our knowledge, this provides the first robustness guarantee for asynchronous Q-learning subject to both reward and state corruption. Furthermore, when only rewards are corrupted, the dependence of our algorithm’s bound on the corruption fraction is minimax optimal.
    </div>
  </div>
</div>

<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(C_3\)]</span> Robust Q-Learning under Corrupted Rewards</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Aritra Mitra</div>
  <div class="pub-venue">IEEE Conference on Decision and Control, <strong>CDC 2024</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('cdc2024-summary')">Summary</a>]
    [<a class="link-paper" href="https://arxiv.org/pdf/2409.03237">Paper</a>]
    [<a class="link-poster" href="https://github.com/sreejeetm1729/Robust-Q-Learning-under-Corrupted-Rewards/blob/main/Sreejeet_Maity_AI_Symposium.pdf">Poster</a>]
    [<a class="link-slides" href="https://github.com/sreejeetm1729/Robust-Q-Learning-under-Corrupted-Rewards/blob/main/CDC_Presentation_Slides.pdf">Slides</a>]
    [<a class="link-code" href="https://github.com/sreejeetm1729/Robust-Q-Learning-under-Corrupted-Rewards">Code</a>]
  </div>
</div>
<div id="cdc2024-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('cdc2024-summary')">×</button>
    </div>
    <div class="pub-summary-text">
       Recently, there has been a surge of interest in analyzing the non-asymptotic behavior of model-free reinforcement learning algorithms. However, the performance of such algorithms in non-ideal environments - such as in the presence of corrupted rewards - is poorly understood. Motivated by this gap, we investigate the robustness of the celebrated Q-learning algorithm to a strong-contamination attack model, where an adversary can arbitrarily perturb a small fraction of the observed rewards. We start by proving that such an attack can cause the vanilla Q-learning algorithm to incur arbitrarily large errors. We then develop a novel robust synchronous Qlearning algorithm that uses historical reward data to construct robust empirical Bellman operators at each time step. Finally, we prove a finite-time convergence rate for our algorithm that matches known state-of-the-art bounds (in the absence of attacks) up to a small inevitable error term that scales with the adversarial corruption fraction.
    </div>
  </div>
</div>
<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(C_2\)]</span> Robust Federated Q-Learning with Almost No Communication</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Aritra Mitra</div>
  <div class="pub-venue">American Control Conference, <strong>ACC 2026</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('acc2026-1-summary')">Summary</a>]
    [<a class="link-paper" href="https://github.com/sreejeetm1729/Robust-Federated-Q-Learning-with-Almost-No-communication/blob/main/ACC26_DistRobustQ.pdf">Paper</a>]
    [<a class="link-poster" href="https://github.com/sreejeetm1729/Robust-Federated-Q-Learning-with-Almost-No-communication/blob/main/Robust-FedQ%20Poster.pdf">Poster</a>]
    [<a class="link-slides" href="https://github.com/sreejeetm1729/Robust-Federated-Q-Learning-with-Almost-No-communication/blob/main/ACC%202026%20DisRobQ-ppt.pdf">Slides</a>]
    [<a class="link-code" href="https://github.com/sreejeetm1729/Robust-Federated-Q-Learning-with-Almost-No-communication">Code</a>]
  </div>
</div>
<div id="acc2026-1-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('acc2026-1-summary')">×</button>
    </div>
    <div class="pub-summary-text">
      We consider a federated reinforcement learning setting involving M agents, all of whom interact with a common Markov Decision Process (MDP). The agents exchange information via a central server to learn the optimal value function. Our goal is to understand to what extent one can hope for collaborative sample-complexity speedups in such a setting, when a small fraction of the agents are adversarial and can act arbitrarily. To that end, we propose Robust Fed-Q, a federated Q-learning algorithm that blends ideas from both model-based and model-free RL, along with the median-of-means device from robust statistics. We prove that despite corruption, with high-probability, Robust Fed-Q (i) guarantees exact convergence to the optimal value function in the limit of infinite samples, and (ii) enjoys near-optimal finite-time rates that benefit from collaboration. In addition, our approach requires just constant rounds of communication to achieve each of the above guarantees, a feature of independent interest in FL where communication is the major bottleneck.
    </div>
  </div>
</div>
<div class="pub-entry">
  <div class="pub-title"><span style="color:#e8c4bd;">[\(C_1\)]</span> Variance-Reduced Q-Learning over Static and Time-Varying Networks</div>
  <div class="pub-authors"><strong>Sreejeet Maity</strong><sup>†</sup>, Feng Zhu, Robert Heath, Aritra Mitra</div>
  <div class="pub-venue">American Control Conference, <strong>ACC 2026</strong>.</div>
  <div class="pub-links">
    [<a class="link-summary" href="javascript:void(0);" onclick="togglePubSummary('acc2026-2-summary')">Summary</a>]
    [<a class="link-paper" href="https://github.com/sreejeetm1729/Q-Learning-over-Static-and-Time-Varying-Networks/blob/main/ACC26_GraphQ.pdf">Paper</a>]
    [<a class="link-poster" href="https://github.com/sreejeetm1729/Q-Learning-over-Static-and-Time-Varying-Networks/blob/main/Summary_VRDQ.pdf">Poster</a>]
    [<a class="link-slides" href="https://github.com/sreejeetm1729/Q-Learning-over-Static-and-Time-Varying-Networks/blob/main/ACC-VRDQ.pdf">Slides</a>]
    [<a class="link-code" href="https://github.com/sreejeetm1729/Q-Learning-over-Static-and-Time-Varying-Networks">Code</a>]
  </div>
</div>
<div id="acc2026-2-summary" class="pub-summary-card">
  <div class="pub-summary-inner">
    <div class="pub-summary-header">
      <span style="color: #f5e6b8;">Summary</span>
      <button class="pub-summary-close" onclick="togglePubSummary('acc2026-2-summary')">×</button>
    </div>
    <div class="pub-summary-text">
      We investigate a decentralized reinforcement learning problem involving multiple agents that interact with the same Markov Decision Process (MDP). The agents can exchange information over a network to collectively learn the optimal state-action value function. For this setting, we introduce a novel epoch-based distributed Q-learning algorithm called VRDQ, where within each epoch, agents locally estimaten the Bellman optimality operator and diffuse information using a consensus-based protocol. For both static and time-varying networks, we establish high-probability finite-time convergence rates for VRDQ that enjoy linear speedups from collaboration. Crucially, we prove that such speedups in sample-complexity require only \(\tilde{O}(1)\) communication, substantially improving upon the communication costs in prior work.
    </div>
  </div>
</div>
</div>

<div class="pub-section">

<h2><span style="font-size: 1rem;">♠️</span> Accepted Workshop Presentations</h2>

<ol class="workshop-list">

  <li class="workshop-entry">
    <strong>Multi-Agent Robust FRL with Sparse Communication</strong>,
    <a class="link-workshop" href="https://engr.ncsu.edu/robotics/#symposium">NCSU Robotics Symposium 2026</a>.
  </li>

  <li class="workshop-entry">
    <strong>Agnostic Q-Learning under corrupted Feedback</strong>,
    <a class="link-workshop" href="https://neurips.cc/virtual/2025/workshop/109580">NeurIPS 2025 Reliable ML Workshop</a>.
  </li>

  <li class="workshop-entry">
    <strong>Robust Federated RL with Byzantine Agents</strong>,
    <a class="link-workshop" href="https://engr.ncsu.edu/applied-ai/events/symposium/">Applied AI Symposium 2025</a>,
    <a class="link-workshop" href="https://nescw.org/">NESCW 2026</a>.
  </li>

  <li class="workshop-entry">
    <strong>Theoretical Limits of Robust TD Learning</strong>,
    <a class="link-workshop" href="https://ny-rl.com/">New York RL Workshop, NYRL 2025, Amazon</a>.
  </li>

  <li class="workshop-entry">
    <strong>Towards Finite-Time Theory for Adversarially-Robust RL</strong>,
    <a class="link-workshop" href="https://nescw.org/"> NESCW 2025</a>.
  </li>

  <li class="workshop-entry">
    <strong>Adversarially-Robust Deep Q-Network for Algorithmic Trading</strong>,
    <a class="link-workshop" href="https://sites.google.com/ncsu.edu/mlss2025">MLSS 2025, NCSU</a>.
  </li>

  <li class="workshop-entry">
    <strong>Robust Algorithms for Adversarial Reinforcement Learning</strong>,
    <a class="link-workshop" href="https://engr.ncsu.edu/applied-ai/events/symposium/">Applied AI Symposium 2024</a>.
  </li>

</ol>

</div>

</div>

<div class="pub-section">

<h2><span style="font-size: 1rem;">♣️</span> Invited Talks / Seminars</h2>

<ol class="workshop-list">

  <li class="workshop-entry">
    <strong>Finite-Time Rates for Adversarially-Robust Reinforcement Learning</strong>,
    <span style="color:#e6c98f; font-style:bold;">CORAL Seminar, North Carolina State University</span>.
  </li>

</ol>
<div class="pub-section">
<h2><span style="font-size: 1rem;">♥️</span> Thesis </h2>
</div>

<script>
  function togglePubSummary(id) {
    const card = document.getElementById(id);
    if (!card) return;
    card.classList.toggle("open");
  }
</script>
