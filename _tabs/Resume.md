---
title: Resume
icon: fas fa-user-tie
order: 7
---
 
<style>
.cv-page {
  font-size: 16px;
  line-height: 1.65;
}

.cv-hero {
  padding: 1.4rem 1.5rem;
  border: 1px solid rgba(216,167,167,0.35);
  border-radius: 0px;
  background: rgba(255,255,255,0.025);
  margin-bottom: 2rem;
}

.cv-name {
  font-size: 2.15rem;
  font-weight: 800;
  color: #0891b2;
  margin-bottom: 0.2rem;
}

.cv-title {
  color: #d8a7a7;
  font-size: 1.05rem;
  font-style: bold;
  margin-bottom: 0.8rem;
}

.cv-short {
  max-width: 900px;
  text-align: justify;
  margin-bottom: 0.9rem;
}

.cv-links {
  margin-top: 0.9rem;
}

.cv-links a {
  color: #0891b2;
  text-decoration: none;
  font-weight: 700;
  margin-right: 16px;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.cv-links a:hover {
  color: #d8a7a7;
  text-decoration: none;
}

.cv-links i {
  font-size: 1.05rem;
}

.cv-section {
  margin-top: 2.1rem;
  padding-top: 1.1rem;
  border-top: 3px solid rgba(216,167,167,0.55);
}

.cv-section h2 {
  color: #d8a7a7;
  font-size: 1.45rem;
  margin-bottom: 1rem;
}

.cv-entry {
  margin-bottom: 1.25rem;
}

.cv-entry-title {
  font-weight: 800;
  color: #f5f5f5;
  font-size: 1.02rem;
}

.cv-entry-meta {
  color: #e6c98f;
  font-style: italic;
  margin-top: 0.15rem;
}

.cv-entry-desc {
  margin-top: 0.35rem;
  text-align: justify;
}

.cv-list {
  margin-top: 0.45rem;
  padding-left: 1.2rem;
}

.cv-list li {
  margin-bottom: 0.45rem;
}

.cv-highlight {
  color: #208894;
  font-weight: 800;
}

.cv-muted {
  color: #d8a7a7;
  font-style: italic;
}

.cv-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 1rem;
}

.cv-card {
  padding: 1rem;
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 0px;
  background: rgba(255,255,255,0.025);
}

.cv-chip-row {
  margin-top: 0.75rem;
}

.cv-chip {
  display: inline-block;
  margin: 0.25rem 0.35rem 0.25rem 0;
  padding: 0.2rem 0.55rem;
  border-radius: 0px;
  border: 1px solid rgba(165,243,252,0.25);
  color: #0891b2;
  background: rgba(165,243,252,0.06);
  font-size: 0.9rem;
  font-weight: 700;
}

.cv-pub-title {
  color: #f5f5f5;
  font-weight: 800;
}

.cv-pub-venue {
  color: #d8a7a7;
  font-style: italic;
}

.cv-award-date {
  color: #e6c98f;
}

.cv-publications .publication-theme {
    margin: 1.75rem 0 0.9rem;
    font-size: 1.05rem;
    line-height: 1.5;
    color: #9a3f3f;
  }

  .cv-publications .cv-entry {
    display: grid;
    grid-template-columns: 3.2em minmax(0, 1fr);
    gap: 0.5em;
    align-items: baseline;
    margin-bottom: 1rem;
  }

  .cv-publications .publication-label {
    white-space: nowrap;
  }

  .cv-publications .cv-entry-meta {
    display: inline;
    font-size: inherit;
  }
/* Scoped to Research Experience */
.research-experience {
  --re-text: #ebe9e5;
  --re-muted: #a0a4a5;
  --re-accent: #d8a7a7;
  --re-border: rgba(255, 255, 255, 0.12);
  --re-background: rgba(255, 255, 255, 0.025);
}

.research-experience .re-timeline {
  position: relative;
  margin-left: 0.35rem;
  padding-left: 1.75rem;
}

.research-experience .re-timeline::before {
  content: "";
  position: absolute;
  top: 1.8rem;
  bottom: 1.8rem;
  left: 0;
  width: 1px;
  background: rgba(216, 167, 167, 0.3);
}

.research-experience .re-card {
  position: relative;
  margin-bottom: 1.5rem;
  padding: 1.5rem;
  border: 1px solid var(--re-border);
  border-radius: 0;
  background: var(--re-background);
}

.research-experience .re-card:last-child {
  margin-bottom: 0;
}

.research-experience .re-card::before {
  content: "";
  position: absolute;
  top: 1.7rem;
  left: calc(-1.75rem - 5px);
  width: 9px;
  height: 9px;
  border: 2px solid var(--re-accent);
  border-radius: 0;
  background: var(--main-bg, #1b1b1e);
  box-sizing: border-box;
}

.research-experience .re-header {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  align-items: start;
  gap: 1rem;
}

.research-experience .re-institution {
  margin: 0;
  padding: 0;
  border: 0;
  color: var(--re-text);
  font-size: 1.18rem;
  font-weight: 700;
  line-height: 1.4;
}

.research-experience .re-institution a {
  color: inherit;
  text-decoration: none;
  border-bottom: 0;
}

.research-experience .re-institution a:hover {
  color: var(--re-accent);
}

.research-experience .re-role {
  margin-top: 0.25rem;
  color: var(--re-accent);
  font-size: 0.85rem;
  line-height: 1.6;
}

.research-experience .re-date {
  color: var(--re-muted);
  font-size: 0.78rem;
  line-height: 1.7;
  text-align: right;
  white-space: nowrap;
  font-variant-numeric: tabular-nums;
}

.research-experience .re-location {
  display: block;
  font-size: 0.73rem;
  opacity: 0.85;
}

.research-experience .re-person {
  margin: 0.75rem 0 0;
  color: var(--re-muted);
  font-size: 0.85rem;
  line-height: 1.7;
}

.research-experience .re-person a {
  color: var(--re-text);
  font-weight: 400;
  text-decoration: none;
  border-bottom: 1px solid rgba(216, 167, 167, 0.4);
}

.research-experience .re-person a:hover {
  color: var(--re-accent);
}

.research-experience .re-department {
  display: block;
  margin-top: 0.15rem;
  font-size: 0.78rem;
  color: var(--re-muted);
}

.research-experience .re-work {
  margin-top: 1.2rem;
  padding-top: 1.1rem;
  border-top: 1px solid var(--re-border);
}

.research-experience .re-topic {
  margin: 0 0 0.5rem;
  padding: 0;
  color: #0891b2; 
  font-size: 0.98rem;
  font-weight: 650;
  line-height: 1.5;
}
/*var(--re-accent);*/
.research-experience .re-description {
  margin: 0;
  color: #c9cccb;
  font-size: 0.92rem;
  line-height: 1.8;
  text-align: left;
}

.research-experience .re-description + .re-description {
  margin-top: 0.65rem;
}

.research-experience .re-tags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.45rem;
  margin-top: 0.8rem;
}

.research-experience .re-tag {
  display: inline-flex;
  align-items: center;
  padding: 0.22rem 0.6rem;
  border: 1px solid rgba(216, 167, 167, 0.3);
  border-radius: 0;
  background: rgba(216, 167, 167, 0.045);
  color: #ddc5c5;
  font-size: 0.72rem;
  font-weight: 400;
  line-height: 1.6;
  text-decoration: none;
}

.research-experience a.re-tag:hover {
  color: #f0dddd;
  border-color: var(--re-accent);
  background: rgba(216, 167, 167, 0.1);
}

.research-experience .re-tag-review {
  color: var(--re-muted);
  border-color: var(--re-border);
  background: transparent;
}

.research-experience a:focus-visible {
  outline: 2px solid var(--re-accent);
  outline-offset: 4px;
}
.education-refined {
    --edu-text: #ebe9e5;
    --edu-muted: #a5a5ac;
    --edu-accent: #d8a7a7;
    --edu-rule: rgba(255, 255, 255, 0.11);

    color: var(--edu-text);
    font-size: 15px;
    line-height: 1.65;
  }

  .education-refined .edu-entry {
    padding: 1.4rem 0;
    border-bottom: 1px solid var(--edu-rule);
  }

  .education-refined .edu-entry:last-child {
    padding-bottom: 0;
    border-bottom: 0;
  }

  .education-refined .edu-top {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    align-items: start;
    gap: 0.6rem 1.5rem;
  }

  .education-refined .edu-school {
    color: var(--edu-text);
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.5;
  }

  .education-refined .edu-degree {
    margin-top: 0.2rem;
    color: #0891b2;
    font-size: 0.9rem;
  }

  .education-refined .edu-time {
    padding-top: 0.15rem;
    color: var(--edu-muted);
    font-size: 0.78rem;
    text-align: right;
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }

  .education-refined .edu-place {
    display: block;
    margin-top: 0.2rem;
  }

  .education-refined .edu-advisor {
    margin: 0.65rem 0 0;
    color: var(--edu-muted);
    font-size: 0.85rem;
  }

  .education-refined .edu-advisor a {
    color: var(--edu-accent);
    text-decoration: none;
    border-bottom: 0;
  }

  .education-refined .edu-advisor a:hover {
    text-decoration: underline;
    text-underline-offset: 3px;
  }

  .education-refined .edu-thesis {
    margin-top: 1rem;
    padding-left: 1rem;
    border-left: 2px solid var(--edu-accent);
  }

  .education-refined .edu-label {
    color: var(--edu-muted);
    font-size: 0.7rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }

  .education-refined .edu-thesis-title {
    margin: 0.3rem 0 0;
    color: #e6c98f;
    font-size: 0.9rem;
    line-height: 1.65;
  }

  .education-refined .edu-graduation {
    margin: 0.6rem 0 0;
    color: var(--edu-muted);
    font-size: 0.8rem;
  }

  @media (max-width: 640px) {
    .education-refined .edu-top {
      grid-template-columns: 1fr;
      gap: 0.3rem;
    }

    .education-refined .edu-time {
      text-align: left;
      white-space: normal;
    }

    .education-refined .edu-place {
      display: inline;
    }

    .education-refined .edu-place::before {
      content: " · ";
    }
  }
@media (max-width: 640px) {
  .research-experience .re-timeline {
    margin-left: 0.2rem;
    padding-left: 1.15rem;
  }

  .research-experience .re-card {
    padding: 1.1rem;
  }

  .research-experience .re-card::before {
    left: calc(-1.15rem - 5px);
    top: 1.4rem;
  }

  .research-experience .re-header {
    grid-template-columns: 1fr;
    gap: 0.45rem;
  }

  .research-experience .re-date {
    text-align: left;
    white-space: normal;
  }

  .research-experience .re-location {
    display: inline;
    margin-left: 0.4rem;
  }

  .research-experience .re-institution {
    font-size: 1.05rem;
  }

  .research-experience .re-description {
    font-size: 0.88rem;
  }
}

@media screen and (max-width: 768px) {
  .cv-grid {
    grid-template-columns: 1fr;
  }

  .cv-name {
    font-size: 1.8rem;
  }
}
.education-refined .edu-entry {
  position: relative;
  margin-left: 0.35rem;
  padding-left: 1.75rem;
  border-left: 1px solid rgba(216, 167, 167, 0.3);
}

.education-refined .edu-entry::before {
  content: "";
  position: absolute;
  top: 1.9rem;
  left: -5px;
  width: 9px;
  height: 9px;
  border: 2px solid var(--edu-accent, #d8a7a7);
  border-radius: 0;
  background: var(--main-bg, #1b1b1e);
  box-sizing: border-box;
}

@media (max-width: 640px) {
  .education-refined .edu-entry {
    margin-left: 0.2rem;
    padding-left: 1.15rem;
  }
}
</style>

<div class="cv-page">

<div class="cv-hero">

<div class="cv-name">Sreejeet Maity</div>

<div class="cv-title">
  Ph.D. Student in Electrical Engineering · North Carolina State University · Raleigh, NC, U.S.A
</div>

<div class="cv-short">
I develop provably robust finite-sample guarantees for reinforcement learning (RL) under uncertainty and adversarial corruption. My current research interests include corruption-tolerant reinforcement learning, robust policy evaluation, distributed and federated reinforcement learning, and minimax lower bounds that characterize the fundamental limits of robust learning.
</div>

<div class="cv-links">

  <a href="https://scholar.google.com/citations?hl=en&user=tpNoQ3AAAAAJ" target="_blank">
    <i class="fas fa-graduation-cap"></i> Google Scholar
  </a>

  <a href="https://github.com/sreejeetm1729" target="_blank">
    <i class="fab fa-github"></i> GitHub
  </a>

  <a href="https://www.linkedin.com/in/sreejeet-maity-a3729b158/" target="_blank">
    <i class="fab fa-linkedin"></i> LinkedIn
  </a>

  <a href="{{ '/assets/pdf/Target_Academia_Resume-v1.pdf' | relative_url }}" target="_blank">
    <i class="fas fa-file-pdf"></i> Download CV
  </a>
</div>

<div class="cv-chip-row">
  <span class="cv-chip">Reinforcement Learning</span>
  <span class="cv-chip">Statistical Learning Theory</span>
  <span class="cv-chip">Control Theory</span>
  <span class="cv-chip">Applied Probability</span>
  <span class="cv-chip">Stochastic Approximation Algorithms</span>
  <span class="cv-chip">Optimization</span>
  <span class="cv-chip">Robust Statistics</span>
</div>
<div style="margin-top:2rem; padding:1rem 1.2rem; border:1px solid rgba(165,243,252,0.28); border-radius:0px; background:rgba(165,243,252,0.04); font-size:16px; line-height:1.7; text-align:justify;">

  <strong style="color:#d8a7a7;">Collaboration Note.</strong>
  I am always happy to engage with researchers working on broad areas of reinforcement learning, control theory, federated learning, and trustworthy machine learning. If you are interested in discussing possible collaborations or exchanging research ideas, please reach out to
 <a href="https://www.linkedin.com/in/sreejeet-maity-a3729b158/"
   target="_blank"
   title="LinkedIn"
   style="color:#0891b2; font-size:1.25rem; text-decoration:none; margin-left:4px;">
  <i class="fab fa-linkedin"></i>.
</a>
</div>
<div style="margin-top:0.8rem; padding:1rem 1.2rem; border:1px solid rgba(165,243,252,0.28); border-radius:0px; background:rgba(165,243,252,0.04); font-size:16px; line-height:1.7; text-align:justify;">

  <strong style="color:#d8a7a7;">Prospective Opportunities.</strong>
  I will be entering the academic and industry research job market next year (2027), and I would be grateful to hear about opportunities aligned with my interests in broad areas of <strong style="color:#e6c98f;">robust and safe RL, and reliable decision-making and control</strong>. I am also open to postdoctoral opportunities beginning in <strong style="color:#e6c98f;">December 2027</strong>, as well as opportunities in subsequent academic cycles. Prospective recruiters, search committees, and researchers are warmly welcome to reach out.
</div> 
</div>

<div class="cv-section education-refined">

  <h2>Education</h2>

  <!-- North Carolina State University -->
  <article class="edu-entry">

    <div class="edu-top">
      <div>
        <div class="edu-school">
          North Carolina State University
        </div>

        <div class="edu-degree">
          Ph.D. in Electrical Engineering
        </div>
      </div>

      <div class="edu-time">
        Aug 2023–Present
        <span class="edu-place">Raleigh, NC</span>
      </div>
    </div>

    <p class="edu-advisor">
      Advisor:
      <a href="https://amitra2.wordpress.ncsu.edu/"
         target="_blank"
         rel="noopener noreferrer">
        Dr. Aritra Mitra
      </a>
    </p>

    <div class="edu-thesis">
      <div class="edu-label">Thesis</div>

      <p class="edu-thesis-title">
        <em>
          Towards a Finite-Time Theory of Adversarially
          Robust RL: Algorithms, Guarantees, and
          Fundamental Limits
        </em>
      </p>

      <p class="edu-graduation">
        Expected graduation · December 2027–March 2028
      </p>
    </div>

  </article>

  <!-- Indian Institute of Science -->
  <article class="edu-entry">

    <div class="edu-top">
      <div>
        <div class="edu-school">
          Indian Institute of Science, Bangalore
        </div>

        <div class="edu-degree">
          M.Tech. in Robotics and Autonomous Systems
        </div>
      </div>

      <div class="edu-time">
        Aug 2021–June 2023
        <span class="edu-place">Bangalore, India</span>
      </div>
    </div>

  </article>

  <!-- Jadavpur University -->
  <article class="edu-entry">

    <div class="edu-top">
      <div>
        <div class="edu-school">
          Jadavpur University
        </div>

        <div class="edu-degree">
          B.E. in Electrical Engineering
        </div>
      </div>

      <div class="edu-time">
        Aug 2017–July 2021
        <span class="edu-place">Kolkata, India</span>
      </div>
    </div>

  </article>

</div>

<div class="cv-section research-experience">

  <h2>Research Experience</h2>

  <div class="re-timeline">

    <!-- NORTH CAROLINA STATE UNIVERSITY -->
    <article class="re-card">

      <div class="re-header">
        <div>
          <h3 class="re-institution">
            <a href="https://www.ncsu.edu/"
               target="_blank"
               rel="noopener noreferrer">
              North Carolina State University
            </a>
          </h3>

          <div class="re-role">
            Graduate Research / Teaching Assistant
          </div>
        </div>

        <div class="re-date">
          Aug 2023–Present
          <span class="re-location">Raleigh, NC</span>
        </div>
      </div>

      <p class="re-person">
        Ph.D. Advisor:
        <a href="https://amitra2.wordpress.ncsu.edu/"
           target="_blank"
           rel="noopener noreferrer">
          Dr. Aritra Mitra
        </a>

        <span class="re-department">
          Department of Electrical and Computer Engineering
        </span>
      </p>

      <div class="re-work">
        <h4 class="re-topic">
          Robust optimal policy learning from corrupted
          and correlated observations
        </h4>

        <p class="re-description">
          Showed that vanilla Q-learning is provably fragile
          under reward corruption, designed robust
          Bellman-update methods, and established finite-time
          convergence with matching minimax lower bounds.
        </p>

        <div class="re-tags">
          <a class="re-tag"
             href="https://proceedings.mlr.press/v306/maity26a.html"
             target="_blank"
             rel="noopener noreferrer">
            ICML 2026
          </a>

          <a class="re-tag"
             href="https://arxiv.org/abs/2409.03237"
             target="_blank"
             rel="noopener noreferrer">
            IEEE CDC 2024
          </a>

          <a class="re-tag"
             href="https://arxiv.org/pdf/2607.20822"
             target="_blank"
             rel="noopener noreferrer">
            IEEE CDC 2026
          </a>


          <span class="re-tag re-tag-review">
            Automatica · Under review
          </span>
        </div>
      </div>

      <div class="re-work">
        <h4 class="re-topic">
          Robust policy evaluation under adversarial
          influences and Markovian data
        </h4>

        <p class="re-description">
          Developed finite-time theory for robust
          temporal-difference learning with Markovian noise
          and function approximation, including upper bounds
          and near-tight lower bounds.
        </p>

        <p class="re-description">
          We also investigated communication-efficient policy
          evaluation under finite-bit communication constraints,
          developing adaptive quantization methods with provable
          finite-time convergence guarantees.
        </p>

        <div class="re-tags">
          <a class="re-tag"
             href="https://proceedings.mlr.press/v258/maity25a.html"
             target="_blank"
             rel="noopener noreferrer">
            AISTATS 2025 
          </a>

          <span class="re-tag re-tag-review">
            L4DC 2027 · Under review
          </span>

          <span class="re-tag re-tag-review">
            ECC 2027 · Under review
          </span>
        </div>
      </div>

      <div class="re-work">
        <h4 class="re-topic">
          Robust federated and multi-agent reinforcement learning
        </h4>

        <p class="re-description">
          Developed adversarially robust and communication-efficient
          reinforcement learning algorithms for federated multi-agent
          settings, including Byzantine-resilient methods with
          collaborative speedups.
        </p>

        <div class="re-tags">
          <a class="re-tag"
             href="https://arxiv.org/pdf/2609.20174"
             target="_blank"
             rel="noopener noreferrer">
            ACC 2026 
          </a>

          <a class="re-tag re-tag-review"
             href="https://arxiv.org/pdf/2610.06918"
             target="_blank"
             rel="noopener noreferrer">
            JMLR · Under review 
          </a>
        </div>
      </div>

    </article>

    <!-- UNIVERSITY OF CALIFORNIA, SAN DIEGO -->
    <article class="re-card">

      <div class="re-header">
        <div>
          <h3 class="re-institution">
            <a href="https://ucsd.edu/"
               target="_blank"
               rel="noopener noreferrer">
              University of California, San Diego
            </a>
          </h3>

          <div class="re-role">
            Research Collaboration · Decentralized Reinforcement Learning
          </div>
        </div>

        <div class="re-date">
          September 2025–Present
          <span class="re-location">San Diego, CA</span>
        </div>
      </div>

      <p class="re-person">
        Research Collaborator:
        <a href="https://profheath.org/"
           target="_blank"
           rel="noopener noreferrer">
          Prof. Robert W. Heath Jr.
        </a>

        <span class="re-department">
          Department of Electrical and Computer Engineering
        </span>
      </p>

      <div class="re-work">
        <h4 class="re-topic">
          Decentralized Q-learning with asynchronous sampling
          and partial state-action coverage
        </h4>

        <p class="re-description">
          Developing decentralized Q-learning algorithms that
          enable agents with partial state-action coverage to
          learn collaboratively from asynchronous observations.
          Establishing finite-time convergence guarantees and
          quantifying the benefits of variance reduction over
          static and time-varying communication networks.
        </p>

        <div class="re-tags">
          <a class="re-tag"
             href="https://arxiv.org/pdf/2607.21876"
             target="_blank"
             rel="noopener noreferrer">
            ACC 2026 
          </a>

          <span class="re-tag re-tag-review">
            IEEE TSP · In preparation
          </span>
        </div>
      </div>

    </article>

    <!-- COLUMBIA UNIVERSITY -->
    <article class="re-card">

      <div class="re-header">
        <div>
          <h3 class="re-institution">
            <a href="https://www.columbia.edu/"
               target="_blank"
               rel="noopener noreferrer">
              Columbia University
            </a>
          </h3>

          <div class="re-role">
            Research Collaboration · Byzantine-Robust Representation Learning
          </div>
        </div>

        <div class="re-date">
          May 2026–Present
          <span class="re-location">New York, NY</span>
        </div>
      </div>

      <p class="re-person">
        Research Collaborator:
        <a href="https://www.columbia.edu/~ja3451/"
           target="_blank"
           rel="noopener noreferrer">
          Prof. James Anderson
        </a>

        <span class="re-department">
          Department of Electrical Engineering
        </span>
      </p>

      <div class="re-work">
        <h4 class="re-topic">
          Byzantine-robust personalized federated policy
          evaluation with shared representations
        </h4>

        <p class="re-description">
          Developing Byzantine-robust representation learning
          methods for heterogeneous reinforcement learning
          agents operating in distinct MDPs with temporally
          dependent data. The approach decomposes each agent's
          value function into a shared low-dimensional
          representation and a personalized local parameter,
          enabling collaborative learning of common structure
          while accommodating differences in transitions,
          rewards, policies, and value functions without
          persistent heterogeneity bias.
        </p>

        <div class="re-tags">
          <span class="re-tag re-tag-review">
            TMLR · Under review
          </span>
        </div>
      </div>

    </article>

  </div>

</div>

<div class="cv-section">
<div class="cv-publications">
  <h2>Publications</h2>

  <section aria-labelledby="publication-theme-1">
    <h3 id="publication-theme-1" class="publication-theme">
      Theme 1. Provably Robust Reinforcement Learning:
      Algorithms and Statistical Limits
    </h3>

    <div class="cv-entry">
      <span class="publication-label">(P1)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Corruption-Tolerant Asynchronous Q-Learning with
          Near-Optimal Rates.</strong>
        <span class="cv-entry-meta">
          <em>International Conference on Machine Learning (ICML)</em>, 2026.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P2)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Robust Asynchronous Q-Learning under Reward and State
          Corruption via Batching.</strong>
        <span class="cv-entry-meta">
          <em>IEEE Conference on Decision and Control (CDC)</em>, 2026.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P3)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Adversarially-Robust TD-Learning with Markovian Data:
          Finite-Time Rates and Fundamental Limits.</strong>
        <span class="cv-entry-meta">
          <em>International Conference on Artificial Intelligence
            and Statistics (AISTATS)</em>, 2025.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P4)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Robust Q-Learning under Corrupted Rewards.</strong>
        <span class="cv-entry-meta">
          <em>IEEE Conference on Decision and Control (CDC)</em>, 2024.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P5)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Towards a Near-Optimal Finite-Time Theory of
          Robust Policy Evaluation under Corruption.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>Learning for Dynamics and Control (L4DC)</em>, 2027.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P6)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Towards Minimax-Optimal Asynchronous Q-Learning
          from Corrupted Markovian Data.</strong>
        <span class="cv-entry-meta">
          Under review at <em>Automatica</em>.
        </span>
      </div>
    </div>
  </section>

  <section aria-labelledby="publication-theme-2">
    <h3 id="publication-theme-2" class="publication-theme">
      Theme 2. Adversarially Robust Federated Reinforcement Learning
      with Collaborative Speedups
    </h3>

    <div class="cv-entry">
      <span class="publication-label">(P7)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Robust Federated Q-Learning with Almost
          No Communication.</strong>
        <span class="cv-entry-meta">
          <em>American Control Conference (ACC)</em>, 2026.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P8)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Learning from Unreliable Trajectories:
          Adversarially-Robust Federated Q-Learning.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>Journal of Machine Learning Research (JMLR)</em>.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P9)</span>
      <div>
        <em>Sreejeet Maity</em>, Leonardo Felipe Toso,
        James Anderson, Aritra Mitra.
        <strong>Value Function Representation for Adversarially
          Robust Federated TD Learning.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>Transactions on Machine Learning Research (TMLR)</em>.
        </span>
      </div>
    </div>
  </section>

  <section aria-labelledby="publication-theme-3">
    <h3 id="publication-theme-3" class="publication-theme">
      Theme 3. Decentralized Reinforcement Learning over
      Static and Time-Varying Networks
    </h3>

    <div class="cv-entry">
      <span class="publication-label">(P10)</span>
      <div>
        <em>Sreejeet Maity</em>, Feng Zhu,
        Robert W. Heath Jr., Aritra Mitra.
        <strong>Variance-Reduced Q-Learning over Static and
          Time-Varying Networks.</strong>
        <span class="cv-entry-meta">
          <em>American Control Conference (ACC)</em>, 2026.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P11)</span>
      <div>
        <em>Sreejeet Maity</em>, Feng Zhu,
        Robert W. Heath Jr., Aritra Mitra.
        <strong>Decentralized Q-Learning with Asynchronous
          Sampling and Partial Coverage.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>IEEE Transactions on Signal Processing (TSP)</em>.
        </span>
      </div>
    </div>
  </section>

  <section aria-labelledby="publication-theme-4">
    <h3 id="publication-theme-4" class="publication-theme">
      Theme 4. Communication Constraints and Sparse Model Changes
      in Reinforcement Learning
    </h3>

    <div class="cv-entry">
      <span class="publication-label">(P12)</span>
      <div>
        <em>Sreejeet Maity</em>, Aritra Mitra.
        <strong>Adaptive Quantization Preserves Fast Rates in
          TD-Learning over Rate Limited Channels.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>European Control Conference (ECC)</em>, 2027.
        </span>
      </div>
    </div>

    <div class="cv-entry">
      <span class="publication-label">(P13)</span>
      <div>
        <em>Sreejeet Maity</em>, Leonardo Felipe Toso.
        <strong>Adaptive History Selection for Reinforcement
          Learning under Sparse Model Changes.</strong>
        <span class="cv-entry-meta">
          Under review at
          <em>European Control Conference (ECC)</em>, 2027.
        </span>
      </div>
    </div>
  </section>
</div>

<div class="cv-section">

<h2>Projects</h2>

<div class="cv-entry">
  <div class="cv-entry-title">Federated MARL-GYM <a href="https://github.com/sreejeetm1729/Federated-MARL-Gym-Environment" target="_blank">
    <i class="fab fa-github"></i> 
  </a></div>
  <div class="cv-entry-desc">
    We introduce a custom multi-agent reinforcement learning environment built with Gymnasium and Pygame, designed for evaluating federated RL (FRL) algorithms. The environment models a grid world where multiple agents navigate to accomplish spatially distributed tasks, like reaching delivery points.
  </div>
</div>

</div>

<div class="cv-section">

<h2>Academic and Professional Service</h2>

<ul class="cv-list">
  <li>Head Teaching Assistant for <span class="cv-highlight">ECE 516: Systems and Control Engineering</span> and <span class="cv-highlight">ECE 308: Elements of Control Systems</span>, Department of Electrical and Computer Engineering, NC State.</li>
  <li>Served as a reviewer for 60+ papers in multiple flagship control/ ML venues, including the American Control Conference (ACC), the IEEE Conference on Decision and Control (CDC), Learning for Dynamics and Control (L4DC), Annual Conference on Neural Information Processing Systems(NeuRIPS), Artifical Intelligence and Statistics Conference (AISTATS), Journal of Machine Learning Research (JMLR), Transactions in Machine Learning Reserach (TMLR), IEEE Transactions in Automatic Control (TAC), Transactions in Signal and Information Processing over Networks (TSIPN), and Transactions in Signal Processing (TSP).</li>
</ul>

</div>

<div class="cv-section">

<h2>Awards</h2>

<ul class="cv-list">
  <li><span class="cv-highlight">ACC 2026 Travel Award</span> <span class="cv-award-date">May 2026</span></li>
  <li><span class="cv-highlight">L4DC Student Support Grant</span> <span class="cv-award-date">May 2025</span></li>
  <li><span class="cv-highlight">NESCW 2025 Student Support Grant</span> <span class="cv-award-date">May 2025</span></li>
  <li><span class="cv-highlight">IEEE CDC 2024 Student Support Award</span> <span class="cv-award-date">August 2024</span></li>
  <li><span class="cv-highlight">NC State ECE Student Research Support Award</span> <span class="cv-award-date">August 2024</span></li>
  <li><span class="cv-highlight">College of Engineering Graduate Merit Award</span> <span class="cv-award-date">2023--24, 2024--25</span></li>
</ul>

</div>

<div class="cv-section">

<h2>Skills</h2>

<div class="cv-grid">

<div class="cv-card">
  <div class="cv-entry-title">Mathematics</div>
  <div class="cv-entry-desc">Linear Algebra, Matrix Theory, Probability Theory, Random Process, Randomized Algorithms, Graph Theory, Analysis, Causal Inference, Bayesian Analysis, High-Dimensional Statistics, Stochastic Optimization.</div>
</div>

<div class="cv-card">
  <div class="cv-entry-title">Research</div>
  <div class="cv-entry-desc">Reinforcement Learning, Statistical Learning Theory, Bandit Algorithms, Optimization, Information Theory, Control Theory, Stochastic Approximation, Robust Statistics, Distributed Networks.</div>
</div>

<div class="cv-card">
  <div class="cv-entry-title">Programming</div>
  <div class="cv-entry-desc">Python, MATLAB, Simulink, C/C++.</div>
</div>

<div class="cv-card">
  <div class="cv-entry-title">Software and Libraries</div>
  <div class="cv-entry-desc">PyTorch, TensorFlow, NumPy, Pandas, scikit-learn, Stable-Baselines3, CVXPY, CasADi, CleanRL, Gymnasium,  MuJoCo.</div>
</div>

</div>

</div>

<div class="cv-section">

<h2>Relevant Coursework</h2>

<ul class="cv-list">
  <li><span class="cv-highlight">Learning Theory:</span> Theoretical Foundations of Large-Scale Machine Learning, Reinforcement Learning, Machine Learning for Signal Processing, Bayesian Learning, Physics Modelling with Neural Networks, Deep Learning and Neural Networks.</li>
  <li><span class="cv-highlight">Mathematics:</span> Analysis I, Probability and Stochastic Processes, Stochastic Models and Applications, Convex Optimization for Data Science, Detection and Estimation Theory.</li>
  <li><span class="cv-highlight">Control Theory:</span> Dynamics of Linear Systems, Networked and Distributed Control, Safety-Critical Control for Robotic Systems, Non-Linear Control Theory, Formal Analysis for Control Theory.</li>
</ul>

</div>

