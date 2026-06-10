---
title: Learning Agents
icon: fas fa-robot
order: 8
math: true
---

<div class="learning-agents-intro">
  <p>
    This page hosts small interactive environments for building intuition about learning, control, rewards, disturbances, and task completion. Each playground is embedded in an isolated frame so the website theme does not interfere with the simulation code.
  </p>
</div>

<section class="learning-agent-block">
  <div class="learning-agent-heading">
    <h3>3D Drone RL Playground</h3>
    <p>
      A compact drone-control playground for visualizing inertia, wind, waypoint tracking,
      hovering, and cargo delivery.
    </p>
  </div>

  <iframe
    class="learning-agent-frame drone-frame"
    src="{{ '/assets/html/rl3d_drone_playground_compact.html' | relative_url }}"
    title="3D Drone RL Playground"
    loading="lazy"
    allow="fullscreen"
  ></iframe>
</section>

<section class="learning-agent-block">
  <div class="learning-agent-heading">
    <h3>Learning to Navigate a Living City</h3>
    <p>
      A city-scale reinforcement-learning playground where an autonomous agent learns
      daily chore routines under traffic, pedestrians, weather, time, and energy costs.
    </p>
  </div>

  <iframe
  class="learning-agent-frame city-frame"
  src="{{ '/assets/html/living_city_exact_dashboard_width_dashboard.html' | relative_url }}"
  title="Learning to Navigate a Living City"
  loading="lazy"
  allow="fullscreen"
></iframe>

  <div class="learning-agent-credit">
    Built with <a href="https://threejs.org/" target="_blank" rel="noopener">Three.js</a> (MIT). Procedural city assets.
  </div>
</section>

<style>
/* Hide Chirpy right panel only on this page:
   this suppresses Recently Updated + Trending Tags. */
#panel-wrapper,
aside#panel-wrapper,
aside[aria-label="Panel"] {
  display: none !important;
}

/* Let the main page use the freed horizontal space. */
#main-wrapper,
#main-wrapper > .container,
#main-wrapper > .container > .row {
  max-width: 100% !important;
}

main[aria-label="Main Content"],
#main-wrapper main {
  flex: 0 0 100% !important;
  max-width: 100% !important;
}

/* Page intro */
.learning-agents-intro {
  max-width: 980px;
  margin: 0.25rem auto 1.05rem;
  color: var(--text-color, #334155);
}

.learning-agents-intro p {
  line-height: 1.6;
  color: var(--text-color, #334155);
}

/* Wider simulator sections */
.learning-agent-block {
  width: min(1220px, calc(100vw - 24px));
  margin: 1.2rem auto 2rem;
  position: relative;
  left: 50%;
  transform: translateX(-50%);
}

.learning-agent-heading {
  max-width: 1040px;
  margin: 0 auto 0.75rem;
  padding: 0 0.25rem;
}

.learning-agent-heading h3 {
  margin: 0 0 0.3rem;
  color: var(--heading-color, #111827);
  font-size: 1.25rem;
  font-weight: 850;
}

.learning-agent-heading p {
  margin: 0;
  color: var(--text-muted-color, #64748b);
  line-height: 1.5;
  font-size: 0.95rem;
}

.learning-agent-frame {
  display: block;
  width: 100%;
  border: 0;
  border-radius: 18px;
  background: #f8fafc;
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.12);
  overflow: hidden;
}

/* Taller iframes so the simulator itself does not scroll internally */
.drone-frame {
  height: 860px;
}

.city-frame {
  height: 940px;
}

.learning-agent-credit {
  margin-top: 0.4rem;
  text-align: right;
  font-size: 0.76rem;
  opacity: 0.72;
  color: var(--text-muted-color, #64748b);
}

.learning-agent-credit a {
  font-weight: 700;
}

@media (max-width: 900px) {
  .learning-agent-block {
    width: calc(100vw - 16px);
  }

  .drone-frame {
    height: 820px;
  }

  .city-frame {
    height: 900px;
  }
}

@media (max-width: 600px) {
  .learning-agent-block {
    width: calc(100vw - 10px);
  }

  .drone-frame {
    height: 790px;
  }

  .city-frame {
    height: 860px;
  }

  .learning-agent-heading h3 {
    font-size: 1.12rem;
  }

  .learning-agent-heading p,
  .learning-agents-intro p {
    font-size: 0.9rem;
  }
}
</style>
