---
title: "Stochastic Gradient Descent: Why Randomness Works"
date: 2026-05-13
categories: [rl-blogs]
rl_section: rl-fundamentals
tags: [optimization, gradient-descent, stochastic-gradient-descent, machine-learning]
math: true
---

Optimization is one of the quiet engines behind modern machine learning. Whenever we train a neural network, fit a regression model, learn a value function, or tune a policy, we are usually doing some form of optimization.

At the heart of this story lies a deceptively simple idea:

> move in the direction that makes the loss smaller.

This is the philosophy behind **gradient descent**. But in modern machine learning, we often do not use plain gradient descent. Instead, we use its noisy, randomized, and surprisingly powerful cousin:

$$
\textbf{Stochastic Gradient Descent, or SGD.}
$$

The goal of this post is to explain, mathematically and intuitively, why SGD works, how it differs from vanilla gradient descent, and why injecting randomness into optimization is not a bug, but often a feature.

---

## 1. The Optimization Problem

Suppose we are given a loss function

$$
F(w),
$$

where $$w \in \mathbb{R}^d$$ denotes the parameter vector of a model. In supervised learning, for example, $$w$$ could be the weights of a neural network, and $$F(w)$$ measures how badly the model performs on the data.

A standard finite-sum learning objective has the form

$$
F(w)
=
\frac{1}{n}
\sum_{i=1}^n f_i(w),
$$

where $$f_i(w)$$ is the loss on the $$i$$-th data point.

For example, if we have training samples $$(x_i,y_i)$$, then one may define

$$
f_i(w)
=
\ell(h_w(x_i),y_i),
$$

where $$h_w$$ is the model and $$\ell$$ is a loss function.

The optimization goal is

$$
w^\star
\in
\arg\min_{w \in \mathbb{R}^d} F(w).
$$

So the question is simple:

> How do we find a parameter vector $$w$$ that makes $$F(w)$$ small?

---

## 2. Vanilla Gradient Descent

Gradient descent uses the local geometry of $$F$$. If $$F$$ is differentiable, then its gradient

$$
\nabla F(w)
$$

points in the direction of steepest local increase. Therefore, the negative gradient

$$
-\nabla F(w)
$$

points in the direction of steepest local decrease.

The vanilla gradient descent update is

$$
w_{t+1}
=
w_t
-
\eta \nabla F(w_t),
$$

where $$\eta>0$$ is the step size, also called the learning rate.

For the finite-sum objective,

$$
F(w)
=
\frac{1}{n}
\sum_{i=1}^n f_i(w),
$$

the full gradient is

$$
\nabla F(w)
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w).
$$

Thus each gradient descent step requires computing gradients over the entire dataset.

This is perfectly reasonable when $$n$$ is small. But when $$n$$ is very large, computing

$$
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w)
$$

at every iteration becomes expensive.

This is where stochastic gradient descent enters the story.

---

## 3. The Main Idea of SGD

Instead of computing the full gradient over all $$n$$ samples, SGD randomly picks one data point $$i_t$$ and uses

$$
\nabla f_{i_t}(w_t)
$$

as a cheap estimate of the full gradient.

The SGD update is

$$
w_{t+1}
=
w_t
-
\eta_t \nabla f_{i_t}(w_t),
$$

where $$i_t$$ is sampled randomly, often uniformly from $$\{1,\dots,n\}$$.

<div id="sgd-dual-widget">
  <div class="sgdd-card">
    <h3>Stochastic Gradient Descent: 2D and 3D views</h3>

    <p>
      The left panel shows the contour map of the objective, while the right panel
      shows the same objective as a three-dimensional surface. Unlike full gradient
      descent, the trajectory now contains random fluctuations: each step uses a noisy
      stochastic gradient rather than the exact full gradient.
    </p>

    <div class="sgdd-actions">
      <button id="sgdd-playpause" type="button">Pause</button>
      <button id="sgdd-reset" type="button">Reset</button>
      <button id="sgdd-randomize" type="button">Generate new landscape</button>
    </div>

    <div class="sgdd-sliders">
      <label>
        Learning rate:
        <input id="sgdd-eta" type="range" min="0.03" max="0.22" step="0.01" value="0.09">
        <span id="sgdd-eta-value"></span>
      </label>

      <label>
        Noise level:
        <input id="sgdd-noise" type="range" min="0" max="1.6" step="0.05" value="0.75">
        <span id="sgdd-noise-value"></span>
      </label>
    </div>

    <div class="sgdd-grid">
      <div class="sgdd-panel">
        <div class="sgdd-panel-title">2D contour view</div>
        <div id="sgdd-plot-2d"></div>
      </div>

      <div class="sgdd-panel">
        <div class="sgdd-panel-title">3D surface view</div>
        <div id="sgdd-plot-3d"></div>
      </div>
    </div>

    <div class="sgdd-readout">
      <span><strong>Iteration:</strong> <span id="sgdd-iter"></span></span>
      <span><strong>Point:</strong> <span id="sgdd-point"></span></span>
      <span><strong>Loss:</strong> <span id="sgdd-loss"></span></span>
      <span><strong>True gradient norm:</strong> <span id="sgdd-gradnorm"></span></span>
      <span><strong>Stochastic gradient norm:</strong> <span id="sgdd-stochgradnorm"></span></span>
      <span><strong>Message:</strong> <span id="sgdd-message"></span></span>
    </div>
  </div>
</div>

<style>
  #sgd-dual-widget {
    margin: 2rem 0;
    font-family: inherit;
  }

  #sgd-dual-widget .sgdd-card {
    border: 1px solid rgba(150, 150, 150, 0.25);
    border-radius: 18px;
    padding: 1.2rem;
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
  }

  #sgd-dual-widget h3 {
    margin: 0;
    font-size: 1.3rem;
  }

  #sgd-dual-widget p {
    margin: 0.45rem 0 1rem;
    opacity: 0.86;
    font-size: 0.95rem;
  }

  #sgd-dual-widget .sgdd-actions {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1rem;
    flex-wrap: wrap;
  }

  #sgd-dual-widget .sgdd-actions button {
    border: 1px solid rgba(150,150,150,0.35);
    border-radius: 999px;
    padding: 0.5rem 0.95rem;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-weight: 700;
  }

  #sgd-dual-widget .sgdd-actions button:hover {
    background: rgba(150,150,150,0.12);
  }

  #sgd-dual-widget .sgdd-sliders {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }

  #sgd-dual-widget .sgdd-sliders label {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.6rem;
    align-items: center;
    padding: 0.65rem 0.8rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
    font-size: 0.92rem;
  }

  #sgd-dual-widget input[type="range"] {
    width: 100%;
  }

  #sgd-dual-widget .sgdd-grid {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    gap: 0.75rem;
    align-items: stretch;
  }

  #sgd-dual-widget .sgdd-panel {
    min-width: 0;
    border-radius: 14px;
    overflow: hidden;
  }

  #sgd-dual-widget .sgdd-panel-title {
    font-weight: 700;
    margin: 0 0 0.55rem 0.2rem;
    opacity: 0.92;
  }

  #sgdd-plot-2d,
  #sgdd-plot-3d {
    width: 100%;
    height: 400px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(150,150,150,0.25);
    background: #111111;
  }

  #sgd-dual-widget .sgdd-readout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 1rem;
    font-size: 0.92rem;
  }

  #sgd-dual-widget .sgdd-readout span {
    padding: 0.65rem 0.8rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
  }

  @media (max-width: 700px) {
    #sgd-dual-widget .sgdd-grid,
    #sgd-dual-widget .sgdd-sliders,
    #sgd-dual-widget .sgdd-readout {
      grid-template-columns: 1fr;
    }

    #sgdd-plot-2d,
    #sgdd-plot-3d {
      height: 380px;
    }
  }
</style>

<script>
  function loadPlotlyForSGDDual(callback) {
    if (window.Plotly) {
      callback();
      return;
    }

    var script = document.createElement("script");
    script.src = "https://cdn.plot.ly/plotly-2.35.2.min.js";
    script.onload = function () {
      callback();
    };
    script.onerror = function () {
      var plot2d = document.getElementById("sgdd-plot-2d");
      var plot3d = document.getElementById("sgdd-plot-3d");
      var msg = "<p style='padding:1rem;'>Plotly could not be loaded.</p>";
      if (plot2d) plot2d.innerHTML = msg;
      if (plot3d) plot3d.innerHTML = msg;
    };
    document.head.appendChild(script);
  }

  function initSGDDualWidget() {
    var plot2d = document.getElementById("sgdd-plot-2d");
    var plot3d = document.getElementById("sgdd-plot-3d");

    if (!plot2d || !plot3d) return;

    var playPauseButton = document.getElementById("sgdd-playpause");
    var resetButton = document.getElementById("sgdd-reset");
    var randomizeButton = document.getElementById("sgdd-randomize");

    var etaSlider = document.getElementById("sgdd-eta");
    var noiseSlider = document.getElementById("sgdd-noise");
    var etaValue = document.getElementById("sgdd-eta-value");
    var noiseValue = document.getElementById("sgdd-noise-value");

    var iterText = document.getElementById("sgdd-iter");
    var pointText = document.getElementById("sgdd-point");
    var lossText = document.getElementById("sgdd-loss");
    var gradNormText = document.getElementById("sgdd-gradnorm");
    var stochGradNormText = document.getElementById("sgdd-stochgradnorm");
    var messageText = document.getElementById("sgdd-message");

    function rand(min, max) {
      return min + Math.random() * (max - min);
    }

    function randn() {
      var u = 0;
      var v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function randomLandscapeParams() {
      return {
        quadX: rand(0.18, 0.55),
        quadY: rand(0.15, 0.50),
        cross: rand(-0.14, 0.14),

        amp1: rand(0.12, 0.42),
        freqX1: rand(0.8, 1.7),
        freqY1: rand(0.8, 1.7),
        phase1: rand(0, 2 * Math.PI),

        amp2: rand(0.06, 0.25),
        freqX2: rand(1.0, 2.0),
        freqY2: rand(1.0, 2.0),
        phase2: rand(0, 2 * Math.PI)
      };
    }

    var landscape = randomLandscapeParams();

    function f(x, y) {
      var p = landscape;

      return (
        p.quadX * x * x +
        p.quadY * y * y +
        p.cross * x * y +
        p.amp1 * Math.sin(p.freqX1 * x + p.phase1) * Math.cos(p.freqY1 * y) +
        p.amp2 * Math.cos(p.freqX2 * x - p.freqY2 * y + p.phase2)
      );
    }

    function grad(x, y) {
      var p = landscape;

      return {
        x:
          2 * p.quadX * x +
          p.cross * y +
          p.amp1 * p.freqX1 * Math.cos(p.freqX1 * x + p.phase1) * Math.cos(p.freqY1 * y) -
          p.amp2 * p.freqX2 * Math.sin(p.freqX2 * x - p.freqY2 * y + p.phase2),

        y:
          2 * p.quadY * y +
          p.cross * x -
          p.amp1 * p.freqY1 * Math.sin(p.freqX1 * x + p.phase1) * Math.sin(p.freqY1 * y) +
          p.amp2 * p.freqY2 * Math.sin(p.freqX2 * x - p.freqY2 * y + p.phase2)
      };
    }

    function stochasticGrad(x, y) {
      var g = grad(x, y);
      var sigma = parseFloat(noiseSlider.value);

      return {
        x: g.x + sigma * randn(),
        y: g.y + sigma * randn()
      };
    }

    function linspace(a, b, n) {
      var arr = [];
      var step = (b - a) / (n - 1);
      for (var i = 0; i < n; i++) {
        arr.push(a + i * step);
      }
      return arr;
    }

    var xs = linspace(-3, 3, 65);
    var ys = linspace(-3, 3, 65);

    function buildSurfaceZ() {
      return ys.map(function (y) {
        return xs.map(function (x) {
          return f(x, y);
        });
      });
    }

    var surfaceZ = buildSurfaceZ();

    var maxIter = 220;
    var intervalMs = 230;

    var initialState = {
      x: 2.15,
      y: 1.65,
      iter: 0
    };

    var state = {
      x: initialState.x,
      y: initialState.y,
      iter: initialState.iter
    };

    var trajectoryX = [state.x];
    var trajectoryY = [state.y];

    var lastStochGrad = { x: 0, y: 0 };

    var running = true;
    var timer = null;

    function currentEta() {
      return parseFloat(etaSlider.value);
    }

    function updateSliderLabels() {
      etaValue.textContent = currentEta().toFixed(2);
      noiseValue.textContent = parseFloat(noiseSlider.value).toFixed(2);
    }

    function updateReadout() {
      var g = grad(state.x, state.y);
      var trueNorm = Math.sqrt(g.x * g.x + g.y * g.y);
      var stochNorm = Math.sqrt(
        lastStochGrad.x * lastStochGrad.x +
        lastStochGrad.y * lastStochGrad.y
      );

      iterText.textContent = state.iter.toString();
      pointText.textContent =
        "(" + state.x.toFixed(3) + ", " + state.y.toFixed(3) + ")";
      lossText.textContent = f(state.x, state.y).toFixed(5);
      gradNormText.textContent = trueNorm.toFixed(5);
      stochGradNormText.textContent = stochNorm.toFixed(5);

      if (parseFloat(noiseSlider.value) < 0.05) {
        messageText.textContent = "Almost full gradient descent.";
      } else {
        messageText.textContent = "Noisy steps, but downhill on average.";
      }

      updateSliderLabels();
    }

    function make2DArrowTrace() {
      var eta = currentEta();
      var dx = -eta * lastStochGrad.x;
      var dy = -eta * lastStochGrad.y;

      var norm = Math.sqrt(dx * dx + dy * dy) || 1;
      var scale = 0.45;

      var x0 = state.x;
      var y0 = state.y;
      var x1 = clamp(x0 + scale * dx / norm, -3, 3);
      var y1 = clamp(y0 + scale * dy / norm, -3, 3);

      return {
        type: "scatter",
        mode: "lines+markers",
        x: [x0, x1],
        y: [y0, y1],
        line: {
          color: "#ffcc00",
          width: 3
        },
        marker: {
          size: [1, 7],
          color: "#ffcc00"
        },
        hoverinfo: "skip",
        showlegend: false
      };
    }

    function make2DData() {
      return [
        {
          type: "contour",
          x: xs,
          y: ys,
          z: surfaceZ,
          colorscale: "RdYlBu",
          reversescale: true,
          contours: {
            coloring: "heatmap",
            showlines: true
          },
          line: {
            width: 0.9
          },
          showscale: true,
          colorbar: {
            title: {
              text: "F(θ)",
              side: "top"
            },
            thickness: 12,
            outlinewidth: 0
          },
          hovertemplate:
            "θ₁: %{x:.2f}<br>" +
            "θ₂: %{y:.2f}<br>" +
            "F(θ): %{z:.3f}",
          showlegend: false
        },
        {
          type: "scatter",
          mode: "lines",
          x: trajectoryX,
          y: trajectoryY,
          line: {
            color: "rgba(255,255,255,0.92)",
            width: 2
          },
          hoverinfo: "skip",
          showlegend: false
        },
        {
          type: "scatter",
          mode: "markers",
          x: trajectoryX,
          y: trajectoryY,
          marker: {
            size: 4,
            color: "rgba(255,255,255,0.75)"
          },
          hoverinfo: "skip",
          showlegend: false
        },
        make2DArrowTrace(),
        {
          type: "scatter",
          mode: "markers",
          x: [state.x],
          y: [state.y],
          marker: {
            size: 12,
            color: "#ff3b30",
            line: {
              color: "#ffffff",
              width: 2
            }
          },
          hovertemplate:
            "current point<br>" +
            "θ₁: %{x:.2f}<br>" +
            "θ₂: %{y:.2f}",
          showlegend: false
        }
      ];
    }

    function make3DData() {
      var trajZ = trajectoryX.map(function (x, i) {
        return f(x, trajectoryY[i]);
      });

      var eta = currentEta();
      var dx = -eta * lastStochGrad.x;
      var dy = -eta * lastStochGrad.y;
      var norm = Math.sqrt(dx * dx + dy * dy) || 1;

      var arrowScale = 0.50;
      var arrowX = clamp(state.x + arrowScale * dx / norm, -3, 3);
      var arrowY = clamp(state.y + arrowScale * dy / norm, -3, 3);

      return [
        {
          type: "surface",
          x: xs,
          y: ys,
          z: surfaceZ,
          colorscale: "Viridis",
          opacity: 0.90,
          showscale: false,
          showlegend: false,
          contours: {
            z: {
              show: true,
              usecolormap: true,
              highlightcolor: "#ffffff",
              project: {
                z: true
              }
            }
          },
          hovertemplate:
            "θ₁: %{x:.2f}<br>" +
            "θ₂: %{y:.2f}<br>" +
            "F(θ): %{z:.3f}"
        },
        {
          type: "scatter3d",
          mode: "lines",
          x: trajectoryX,
          y: trajectoryY,
          z: trajZ,
          line: {
            color: "#ffffff",
            width: 3
          },
          hoverinfo: "skip",
          showlegend: false
        },
        {
          type: "scatter3d",
          mode: "markers",
          x: trajectoryX,
          y: trajectoryY,
          z: trajZ,
          marker: {
            size: 2.5,
            color: "rgba(255,255,255,0.65)"
          },
          hoverinfo: "skip",
          showlegend: false
        },
        {
          type: "scatter3d",
          mode: "lines+markers",
          x: [state.x, arrowX],
          y: [state.y, arrowY],
          z: [f(state.x, state.y), f(arrowX, arrowY)],
          line: {
            color: "#ffcc00",
            width: 6
          },
          marker: {
            size: [1, 5],
            color: "#ffcc00"
          },
          hoverinfo: "skip",
          showlegend: false
        },
        {
          type: "scatter3d",
          mode: "markers",
          x: [state.x],
          y: [state.y],
          z: [f(state.x, state.y)],
          marker: {
            size: 6,
            color: "#ff3b30",
            line: {
              color: "#ffffff",
              width: 2
            }
          },
          hovertemplate:
            "current point<br>" +
            "θ₁: %{x:.2f}<br>" +
            "θ₂: %{y:.2f}<br>" +
            "F(θ): %{z:.3f}",
          showlegend: false
        }
      ];
    }

    function make2DLayout() {
      return {
        margin: {
          l: 48,
          r: 20,
          b: 40,
          t: 8
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        xaxis: {
          title: "θ₁",
          range: [-3, 3],
          zeroline: false,
          scaleanchor: "y",
          scaleratio: 1,
          gridcolor: "rgba(255,255,255,0.10)"
        },
        yaxis: {
          title: "θ₂",
          range: [-3, 3],
          zeroline: false,
          gridcolor: "rgba(255,255,255,0.10)"
        },
        showlegend: false,
        uirevision: "sgdd-2d"
      };
    }

    function make3DLayout() {
      return {
        margin: {
          l: 0,
          r: 0,
          b: 0,
          t: 8
        },
        paper_bgcolor: "rgba(0,0,0,0)",
        plot_bgcolor: "rgba(0,0,0,0)",
        showlegend: false,
        uirevision: "sgdd-3d",
        scene: {
          xaxis: {
            title: "θ₁",
            backgroundcolor: "rgba(0,0,0,0)",
            gridcolor: "rgba(150,150,150,0.25)",
            zerolinecolor: "rgba(150,150,150,0.4)"
          },
          yaxis: {
            title: "θ₂",
            backgroundcolor: "rgba(0,0,0,0)",
            gridcolor: "rgba(150,150,150,0.25)",
            zerolinecolor: "rgba(150,150,150,0.4)"
          },
          zaxis: {
            title: "F(θ)",
            backgroundcolor: "rgba(0,0,0,0)",
            gridcolor: "rgba(150,150,150,0.25)",
            zerolinecolor: "rgba(150,150,150,0.4)"
          },
          camera: {
            eye: {
              x: 1.5,
              y: 1.45,
              z: 1.0
            }
          },
          aspectratio: {
            x: 1,
            y: 1,
            z: 0.72
          }
        }
      };
    }

    var config2D = {
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    };

    var config3D = {
      responsive: true,
      displaylogo: false,
      scrollZoom: true
    };

    function renderBoth() {
      updateReadout();
      Plotly.react(plot2d, make2DData(), make2DLayout(), config2D);
      Plotly.react(plot3d, make3DData(), make3DLayout(), config3D);
    }

    function stepSGD() {
      if (state.iter >= maxIter) {
        stopAnimation();
        return;
      }

      var eta = currentEta();
      var ghat = stochasticGrad(state.x, state.y);

      lastStochGrad = {
        x: ghat.x,
        y: ghat.y
      };

      state.x = state.x - eta * ghat.x;
      state.y = state.y - eta * ghat.y;

      state.x = clamp(state.x, -2.95, 2.95);
      state.y = clamp(state.y, -2.95, 2.95);

      state.iter += 1;

      trajectoryX.push(state.x);
      trajectoryY.push(state.y);

      renderBoth();
    }

    function startAnimation() {
      if (timer) return;

      running = true;
      playPauseButton.textContent = "Pause";
      timer = setInterval(stepSGD, intervalMs);
    }

    function stopAnimation() {
      running = false;
      playPauseButton.textContent = "Play";

      if (timer) {
        clearInterval(timer);
        timer = null;
      }
    }

    function resetState() {
      state.x = initialState.x;
      state.y = initialState.y;
      state.iter = initialState.iter;

      trajectoryX = [state.x];
      trajectoryY = [state.y];

      lastStochGrad = grad(state.x, state.y);
    }

    function resetAnimation() {
      stopAnimation();
      resetState();
      renderBoth();
      startAnimation();
    }

    function randomizeLandscape() {
      stopAnimation();
      landscape = randomLandscapeParams();
      surfaceZ = buildSurfaceZ();
      resetState();
      renderBoth();
      startAnimation();
    }

    Promise.all([
      Plotly.newPlot(plot2d, make2DData(), make2DLayout(), config2D),
      Plotly.newPlot(plot3d, make3DData(), make3DLayout(), config3D)
    ]).then(function () {
      resetState();
      updateReadout();

      playPauseButton.addEventListener("click", function () {
        if (running) {
          stopAnimation();
        } else {
          startAnimation();
        }
      });

      resetButton.addEventListener("click", function () {
        resetAnimation();
      });

      randomizeButton.addEventListener("click", function () {
        randomizeLandscape();
      });

      etaSlider.addEventListener("input", function () {
        updateSliderLabels();
      });

      noiseSlider.addEventListener("input", function () {
        updateSliderLabels();
      });

      startAnimation();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      loadPlotlyForSGDDual(initSGDDualWidget);
    });
  } else {
    loadPlotlyForSGDDual(initSGDDualWidget);
  }
</script>

At first glance, this looks reckless. Instead of using the true gradient $$\nabla F(w_t)$$, we use only one randomly chosen component gradient. But the key observation is that this stochastic gradient is unbiased:

$$
\mathbb{E}_{i_t}
\left[
\nabla f_{i_t}(w_t)
\mid w_t
\right]
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w_t)
=
\nabla F(w_t).
$$

So SGD does not follow the exact gradient. Rather, it follows a noisy estimate whose average direction is correct.

In other words,

$$
\nabla f_{i_t}(w_t)
=
\nabla F(w_t)
+
\xi_t,
$$

where

$$
\mathbb{E}[\xi_t \mid w_t] = 0.
$$

Therefore, SGD can be written as

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t.
$$

This makes the nature of SGD clear:

$$
\text{SGD}
=
\text{Gradient Descent}
+
\text{Random Noise}.
$$

---

## 4. Why Does This Randomness Not Destroy Learning?

The most important reason SGD works is that the noise is centered. The stochastic gradient may be wrong at any individual step, but it is correct on average.

Imagine walking down a hill in fog. Vanilla gradient descent has a perfect compass that always points downhill. SGD has a noisy compass: sometimes it points slightly left, sometimes slightly right, sometimes too steeply, sometimes not steeply enough. But on average, it points downhill.

So even though the SGD trajectory is more jagged, it still tends to move toward regions of lower loss.

Mathematically, suppose $$F$$ is smooth, meaning there exists $$L>0$$ such that

$$
F(y)
\le
F(x)
+
\langle \nabla F(x), y-x\rangle
+
\frac{L}{2}\|y-x\|^2
$$

for all $$x,y$$.

Using the SGD update

$$
w_{t+1}
=
w_t
-
\eta_t g_t,
$$

where

$$
g_t = \nabla f_{i_t}(w_t),
$$

smoothness gives

$$
F(w_{t+1})
\le
F(w_t)
-
\eta_t \langle \nabla F(w_t), g_t\rangle
+
\frac{L\eta_t^2}{2}\|g_t\|^2.
$$

Now take conditional expectation given $$w_t$$. Since

$$
\mathbb{E}[g_t \mid w_t]
=
\nabla F(w_t),
$$

we get

$$
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
\le
F(w_t)
-
\eta_t \|\nabla F(w_t)\|^2
+
\frac{L\eta_t^2}{2}
\mathbb{E}
[
\|g_t\|^2
\mid w_t
].
$$

This inequality captures the entire story.

The term

$$
-\eta_t \|\nabla F(w_t)\|^2
$$

is the descent term. It says SGD wants to reduce the loss.

The term

$$
\frac{L\eta_t^2}{2}
\mathbb{E}
[
\|g_t\|^2
\mid w_t
]
$$

is the price of stochasticity and curvature. It appears because the stochastic step is noisy and the function may be curved.

The first term scales like $$\eta_t$$, while the second term scales like $$\eta_t^2$$. Therefore, when $$\eta_t$$ is small enough, the descent term dominates.

This is the mathematical reason SGD can make progress.

---

## 5. Gradient Descent vs SGD

The difference between gradient descent and SGD is not just computational. They behave differently.

Vanilla gradient descent uses

$$
\nabla F(w_t)
=
\frac{1}{n}
\sum_{i=1}^n \nabla f_i(w_t),
$$

and updates as

$$
w_{t+1}
=
w_t
-
\eta \nabla F(w_t).
$$

SGD uses one randomly sampled gradient

$$
\nabla f_{i_t}(w_t),
$$

and updates as

$$
w_{t+1}
=
w_t
-
\eta_t \nabla f_{i_t}(w_t).
$$

So the main distinction is:

$$
\text{Gradient Descent uses the exact full gradient.}
$$

$$
\text{SGD uses a random unbiased estimate of the gradient.}
$$

The full gradient is accurate but expensive. The stochastic gradient is cheap but noisy.

This gives a fundamental tradeoff:

$$
\text{Gradient Descent: fewer but expensive iterations.}
$$

$$
\text{SGD: many cheap but noisy iterations.}
$$

For large-scale machine learning, the second option is often better.

---

## 6. A Simple Example

Consider the quadratic function

$$
F(w)
=
\frac{1}{2}w^2.
$$

Then

$$
\nabla F(w)=w.
$$

Gradient descent becomes

$$
w_{t+1}
=
w_t
-
\eta w_t
=
(1-\eta)w_t.
$$

If $$0<\eta<2$$, this converges to $$0$$, which is the minimizer.

Now imagine that instead of observing the exact gradient $$w_t$$, we observe a noisy version

$$
g_t
=
w_t + \xi_t,
$$

where

$$
\mathbb{E}[\xi_t \mid w_t]=0.
$$

Then SGD becomes

$$
w_{t+1}
=
w_t
-
\eta_t(w_t+\xi_t)
=
(1-\eta_t)w_t
-
\eta_t \xi_t.
$$

The first term pulls $$w_t$$ toward zero. The second term injects noise.

If $$\eta_t$$ is fixed, the algorithm may keep fluctuating around zero. But if $$\eta_t$$ decreases over time, the noise term becomes smaller and smaller. This is why classical SGD often uses decreasing step sizes such as

$$
\eta_t
=
\frac{c}{t+1}.
$$

Early in training, the algorithm explores aggressively. Later in training, the steps become smaller, allowing the iterates to stabilize.

---

## 7. Mini-Batch SGD

In practice, we often use a compromise between full gradient descent and one-sample SGD. This is called mini-batch SGD.

Instead of choosing one sample, we choose a random batch $$B_t\subseteq \{1,\dots,n\}$$ and compute

$$
g_t
=
\frac{1}{|B_t|}
\sum_{i\in B_t}
\nabla f_i(w_t).
$$

The update is

$$
w_{t+1}
=
w_t
-
\eta_t g_t.
$$

Again,

$$
\mathbb{E}[g_t \mid w_t]
=
\nabla F(w_t).
$$

But the variance decreases as the batch size increases. Roughly,

$$
\mathrm{Var}(g_t)
\approx
\frac{1}{|B_t|}
\mathrm{Var}(\nabla f_i(w_t)).
$$

So larger batches give more accurate gradients, but each iteration becomes more expensive.

This creates another tradeoff:

$$
\text{small batch}
=
\text{cheap but noisy};
$$

$$
\text{large batch}
=
\text{expensive but stable}.
$$

In deep learning, mini-batch SGD is the standard because it balances computational efficiency, memory constraints, and statistical stability.

---

## 8. Why SGD Can Be Better Than Gradient Descent

It may seem that full gradient descent should always be better because it uses the exact gradient. But this is not always true in practice.

There are several reasons SGD can be preferable.

### 8.1 One Full Gradient May Be Too Expensive

If $$n$$ is huge, one full gradient step costs $$n$$ gradient computations.

SGD can make $$n$$ small updates for roughly the same cost as one full gradient update.

So even if each SGD step is noisy, the algorithm may make much faster practical progress in terms of computation time.

### 8.2 Noise Can Help Escape Sharp Regions

In nonconvex optimization, such as neural network training, the loss landscape may contain saddle points, flat regions, and sharp minima.

The noise in SGD can help the algorithm avoid getting stuck in certain unstable regions. Informally, the randomness keeps the algorithm moving.

This does not mean noise is always good. Too much noise can prevent convergence. But moderate noise can act as an implicit exploratory force.

### 8.3 SGD Often Has an Implicit Regularization Effect

In modern machine learning, we do not only care about minimizing training loss. We care about generalization: performance on unseen data.

Empirically, SGD often finds solutions that generalize well. One intuitive explanation is that SGD noise biases the trajectory toward wider, flatter regions of the loss landscape, although the full theoretical picture is subtle.

A flat minimum is one where small perturbations of $$w$$ do not change the loss too much. Such solutions may be more stable and generalize better.

---

## 9. A Basic Convergence Intuition

Let us write the stochastic gradient as

$$
g_t
=
\nabla F(w_t)+\xi_t,
$$

where

$$
\mathbb{E}[\xi_t\mid w_t]=0.
$$

The SGD update is

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t.
$$

Suppose the noise has bounded second moment:

$$
\mathbb{E}
[
\|g_t\|^2
\mid w_t
]
\le
G^2.
$$

Then from the smoothness inequality,

$$
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
\le
F(w_t)
-
\eta_t \|\nabla F(w_t)\|^2
+
\frac{L\eta_t^2G^2}{2}.
$$

After rearranging,

$$
\eta_t \|\nabla F(w_t)\|^2
\le
F(w_t)
-
\mathbb{E}
[
F(w_{t+1})
\mid w_t
]
+
\frac{L\eta_t^2G^2}{2}.
$$

Summing from $$t=0$$ to $$T-1$$, taking expectations, and telescoping gives

$$
\sum_{t=0}^{T-1}
\eta_t
\mathbb{E}
[
\|\nabla F(w_t)\|^2
]
\le
F(w_0)-F^\star
+
\frac{LG^2}{2}
\sum_{t=0}^{T-1}
\eta_t^2.
$$

This inequality says that the average gradient norm becomes small if the step sizes are chosen properly.

For example, if we choose a constant step size $$\eta$$, then

$$
\frac{1}{T}
\sum_{t=0}^{T-1}
\mathbb{E}
[
\|\nabla F(w_t)\|^2
]
\lesssim
\frac{F(w_0)-F^\star}{\eta T}
+
L\eta G^2.
$$

The first term decreases with $$T$$. The second term is the noise floor caused by stochastic gradients.

If $$\eta$$ is small, the noise floor is small. But if $$\eta$$ is too small, progress is slow. This is the central step-size tradeoff in SGD.

---

## 10. The Role of the Learning Rate

The learning rate $$\eta_t$$ controls the balance between progress and noise.

If $$\eta_t$$ is too large, SGD becomes unstable:

$$
w_{t+1}
=
w_t
-
\eta_t g_t
$$

may jump wildly across the loss landscape.

If $$\eta_t$$ is too small, SGD becomes slow and may barely move.

A decreasing learning rate helps resolve this tension:

$$
\eta_t \downarrow 0.
$$

Early iterations use large steps to move quickly. Later iterations use small steps to reduce noise and stabilize near a minimizer.

Classical stochastic approximation often requires

$$
\sum_{t=0}^{\infty} \eta_t = \infty,
$$

and

$$
\sum_{t=0}^{\infty} \eta_t^2 < \infty.
$$

The first condition says the algorithm keeps moving enough to reach the solution. The second says the accumulated noise remains controlled.

A standard example is

$$
\eta_t
=
\frac{c}{t+1}.
$$

---

## 11. Convex Picture: Moving Toward the Optimum

When $$F$$ is convex, we can make the intuition even cleaner.

Convexity means

$$
F(w)-F(w^\star)
\le
\langle \nabla F(w), w-w^\star\rangle.
$$

Now consider the squared distance to the minimizer:

$$
\|w_{t+1}-w^\star\|^2.
$$

Using the SGD update,

$$
w_{t+1}
=
w_t-\eta_t g_t,
$$

we get

$$
\|w_{t+1}-w^\star\|^2
=
\|w_t-w^\star\|^2
-
2\eta_t \langle g_t, w_t-w^\star\rangle
+
\eta_t^2\|g_t\|^2.
$$

Taking conditional expectation,

$$
\mathbb{E}
[
\|w_{t+1}-w^\star\|^2
\mid w_t
]
=
\|w_t-w^\star\|^2
-
2\eta_t
\langle \nabla F(w_t), w_t-w^\star\rangle
+
\eta_t^2
\mathbb{E}
[
\|g_t\|^2
\mid w_t
].
$$

Using convexity,

$$
\langle \nabla F(w_t), w_t-w^\star\rangle
\ge
F(w_t)-F(w^\star).
$$

Therefore,

$$
\mathbb{E}
[
\|w_{t+1}-w^\star\|^2
\mid w_t
]
\le
\|w_t-w^\star\|^2
-
2\eta_t
(
F(w_t)-F(w^\star)
)
+
\eta_t^2G^2.
$$

Again, the same structure appears:

$$
\text{progress term}
-
2\eta_t(F(w_t)-F^\star),
$$

plus

$$
\text{noise term}
+
\eta_t^2G^2.
$$

The algorithm moves toward the optimum in expectation, while stochasticity introduces a second-order penalty.

---

## 12. A Geometric Intuition

Gradient descent follows a smooth path down the loss surface.

SGD follows a noisy path.

But the noisy path is not arbitrary. It is biased toward descent.

One can think of SGD as a particle moving under two forces:

$$
\text{deterministic force} = -\nabla F(w_t),
$$

and

$$
\text{random force} = -\xi_t.
$$

So the update

$$
w_{t+1}
=
w_t
-
\eta_t \nabla F(w_t)
-
\eta_t \xi_t
$$

resembles a discretized noisy dynamical system.

When the gradient is large, the deterministic descent dominates. When the gradient is small, the noise becomes more visible. This is why SGD often fluctuates near minimizers instead of stopping exactly.

The learning rate controls the temperature of this motion. A large learning rate creates a hotter, more exploratory process. A small learning rate creates a colder, more stable process.

---

## 13. Summary: What Makes SGD Work?

SGD works because it replaces the full gradient with a cheap random estimate:

$$
g_t
=
\nabla f_{i_t}(w_t).
$$

The crucial property is unbiasedness:

$$
\mathbb{E}[g_t\mid w_t]
=
\nabla F(w_t).
$$

Thus, SGD moves in the correct direction on average.

The price is variance:

$$
g_t
=
\nabla F(w_t)+\xi_t.
$$

This variance makes the trajectory noisy, but with appropriate learning rates, the noise can be controlled.

Vanilla gradient descent is accurate but expensive. SGD is noisy but cheap. In large-scale machine learning, cheap noisy progress often beats expensive exact progress.

The essence of SGD is therefore beautifully simple:

$$
\boxed{
\text{Do not compute the perfect direction. Compute a cheap direction that is correct on average.}
}
$$

That single idea powers much of modern machine learning.

---

## 14. Final Takeaway

Gradient descent asks:

> What is the exact direction of steepest descent?

SGD asks a more practical question:

> Can I cheaply estimate a useful descent direction?

The answer is yes.

And that is why stochastic gradient descent is so powerful. It turns optimization from a deterministic march into a noisy but efficient journey, where randomness is not merely tolerated, but exploited.
