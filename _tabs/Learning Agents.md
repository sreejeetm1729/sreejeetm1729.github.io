---
title: Learning Agents
icon: fas fa-robot
order: 8
math: true
---

<div class="learning-agents-intro">
  <p>
    This page hosts small interactive environments for building intuition about learning,
    control, rewards, disturbances, and task completion. Each playground is isolated in
    its own frame so the website theme does not interfere with the simulation code.
  </p>
</div>

<section class="learning-agent-block">
  <div class="learning-agent-heading">
    <h3>3D Drone RL Playground</h3>
    <p>
      A drone-control playground for visualizing inertia, wind, waypoint tracking,
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
    src="{{ '/assets/html/living_city_learning_compact.html' | relative_url }}"
    title="Learning to Navigate a Living City"
    loading="lazy"
    allow="fullscreen"
  ></iframe>

  <div class="learning-agent-credit">
    Built with <a href="https://threejs.org/" target="_blank" rel="noopener">Three.js</a> (MIT). Procedural city assets.
  </div>
</section>

<style>
.learning-agents-intro {
  margin: 0.25rem 0 1.25rem;
  color: var(--text-color, #334155);
}

.learning-agents-intro h2 {
  margin-bottom: 0.55rem;
  color: var(--heading-color, #111827);
}

.learning-agents-intro p {
  max-width: 900px;
  line-height: 1.65;
  color: var(--text-color, #334155);
}

.learning-agent-block {
  width: min(1120px, calc(100vw - 28px));
  margin: 1.15rem 0 1.75rem;
  position: relative;
  left: 50%;
  transform: translateX(-50%);
}

.learning-agent-heading {
  max-width: 960px;
  margin: 0 auto 0.85rem;
  padding: 0 0.25rem;
}

.learning-agent-heading h3 {
  margin: 0 0 0.35rem;
  color: var(--heading-color, #111827);
  font-size: 1.28rem;
  font-weight: 850;
}

.learning-agent-heading p {
  margin: 0;
  color: var(--text-muted-color, #64748b);
  line-height: 1.6;
}

.learning-agent-frame {
  display: block;
  width: 100%;
  border: 0;
  border-radius: 18px;
  background: #f8fafc;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.12);
  overflow: hidden;
}

.drone-frame {
  height: 760px;
}

.city-frame {
  height: 820px;
}

.learning-agent-credit {
  margin-top: 0.45rem;
  text-align: right;
  font-size: 0.78rem;
  opacity: 0.75;
  color: var(--text-muted-color, #64748b);
}

.learning-agent-credit a {
  font-weight: 700;
}

@media (max-width: 900px) {
  .learning-agent-block {
    width: calc(100vw - 20px);
  }

  .drone-frame {
    height: 720px;
  }

  .city-frame {
    height: 780px;
  }
}
</style>
