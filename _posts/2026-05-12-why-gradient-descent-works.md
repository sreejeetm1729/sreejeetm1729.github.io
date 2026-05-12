---
title: "Why Gradient Descent Works: A Small Mathematical Story"
date: 2026-05-12 00:00:00 -0400
categories: [rl-blogs]
tags: [optimization, gradient-descent, machine-learning, theory]
series: Fundamentals of Reinforcement Learning
math: true
---

Gradient descent is one of the simplest and most influential algorithms in modern machine learning. At a high level, the idea is almost embarrassingly simple: if we want to minimize a function, we move in the direction in which the function decreases fastest.

Suppose we want to solve

$$
\min_{x \in \mathbb{R}^d} f(x),
$$

where $f:\mathbb{R}^d \to \mathbb{R}$ is differentiable. Gradient descent generates a sequence

$$
x_{t+1} = x_t - \eta \nabla f(x_t),
$$

where $\eta>0$ is called the step-size or learning rate.

The entire algorithm is contained in this single line. At first glance, this update looks almost too simple to matter. We begin at a point $$x_t$$, compute the gradient $$\nabla f(x_t)$$, and then take a small step in the opposite direction. The number $$\eta>0$$ is called the **learning rate** or **step size**, and it controls how far we move at each iteration.

The intuition is wonderfully geometric. The gradient points in the direction where the function increases the fastest. So, if our goal is to make the function smaller, the most natural thing to do is to walk in the opposite direction. Gradient descent is simply the act of repeatedly asking:

> Where is downhill from here?

and then taking a small step that way.

This is what makes the method so elegant. Gradient descent does not require us to understand the entire landscape of the function. It only uses local information: the slope at the current point. Yet, by repeating this local rule again and again, the iterates can move toward a minimizer.

Of course, the size of the step matters. If $$\eta$$ is too large, we may overshoot the minimum and oscillate. If $$\eta$$ is too small, we may move in the right direction but painfully slowly. But under suitable assumptions, such as smoothness and convexity, this one-line update can be proved to converge.

That is the beauty of gradient descent: a global optimization problem is attacked through a purely local rule. The convergence proof makes this intuition precise, and reveals why this deceptively simple algorithm sits at the heart of modern machine learning.

<div id="gd-3d-widget">
  <div class="gd3d-card">
    <div class="gd3d-header">
      <div>
        <h3>Gradient Descent on a 3D Loss Landscape</h3>
        <p>
          Rotate the surface and move the point. For visualization, we consider a simple
  two-parameter problem, where the loss is a function \(f:\mathbb{R}^2 \to \mathbb{R}\).
  This lets us draw the graph of the loss as a three-dimensional surface
  \(z=f(x,y)\). The red point lives on this loss surface, the white arrow points
  uphill in the direction of \(\nabla f(x,y)\), and the black arrow points downhill
  in the gradient descent direction \(-\nabla f(x,y)\).
        </p>
      </div>
    </div>

    <div id="gd-3d-plot"></div>

    <div class="gd3d-controls">
      <label>
        x:
        <input id="gd-x-slider" type="range" min="-3" max="3" step="0.05" value="1.2">
        <span id="gd-x-value"></span>
      </label>

      <label>
        y:
        <input id="gd-y-slider" type="range" min="-3" max="3" step="0.05" value="1.0">
        <span id="gd-y-value"></span>
      </label>
    </div>

    <div class="gd3d-readout">
      <span><strong>Loss:</strong> <span id="gd-loss-value"></span></span>
      <span><strong>Gradient:</strong> <span id="gd-grad-value"></span></span>
    </div>

    <p class="gd3d-caption">
      The gradient \(\nabla f(x,y)\) gives the direction of steepest increase on the surface.
      Gradient descent moves in the opposite direction, \(-\nabla f(x,y)\).
    </p>
  </div>
</div>

<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>

<style>
  #gd-3d-widget {
    margin: 2rem 0;
    font-family: inherit;
  }

  #gd-3d-widget .gd3d-card {
    border: 1px solid rgba(150, 150, 150, 0.25);
    border-radius: 18px;
    padding: 1.2rem;
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
  }

  #gd-3d-widget .gd3d-header h3 {
    margin: 0;
    font-size: 1.35rem;
  }

  #gd-3d-widget .gd3d-header p {
    margin: 0.35rem 0 1rem;
    opacity: 0.82;
    font-size: 0.95rem;
  }

  #gd-3d-plot {
    width: 100%;
    height: 520px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(150,150,150,0.25);
  }

  #gd-3d-widget .gd3d-controls {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1rem;
  }

  #gd-3d-widget .gd3d-controls label {
    display: grid;
    grid-template-columns: auto 1fr auto;
    gap: 0.6rem;
    align-items: center;
    padding: 0.7rem 0.85rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
  }

  #gd-3d-widget input[type="range"] {
    width: 100%;
  }

  #gd-3d-widget .gd3d-readout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin-top: 1rem;
    font-size: 0.92rem;
  }

  #gd-3d-widget .gd3d-readout span {
    padding: 0.65rem 0.8rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
  }

  #gd-3d-widget .gd3d-caption {
    margin: 1rem 0 0;
    opacity: 0.85;
    font-size: 0.95rem;
  }

  @media (max-width: 700px) {
    #gd-3d-widget .gd3d-controls,
    #gd-3d-widget .gd3d-readout {
      grid-template-columns: 1fr;
    }

    #gd-3d-plot {
      height: 430px;
    }
  }
</style>

<script>
document.addEventListener("DOMContentLoaded", function () {
  const plotDiv = document.getElementById("gd-3d-plot");

  const xSlider = document.getElementById("gd-x-slider");
  const ySlider = document.getElementById("gd-y-slider");

  const xValue = document.getElementById("gd-x-value");
  const yValue = document.getElementById("gd-y-value");
  const lossValue = document.getElementById("gd-loss-value");
  const gradValue = document.getElementById("gd-grad-value");

  function f(x, y) {
    return 0.35 * (x * x + 0.75 * y * y) + Math.sin(2 * x) * Math.cos(1.5 * y);
  }

  function grad(x, y) {
    return {
      x: 0.7 * x + 2 * Math.cos(2 * x) * Math.cos(1.5 * y),
      y: 0.525 * y - 1.5 * Math.sin(2 * x) * Math.sin(1.5 * y)
    };
  }

  function linspace(a, b, n) {
    const arr = [];
    const step = (b - a) / (n - 1);
    for (let i = 0; i < n; i++) {
      arr.push(a + i * step);
    }
    return arr;
  }

  const xs = linspace(-3, 3, 65);
  const ys = linspace(-3, 3, 65);

  const zSurface = ys.map(function (y) {
    return xs.map(function (x) {
      return f(x, y);
    });
  });

  const surfaceTrace = {
    type: "surface",
    x: xs,
    y: ys,
    z: zSurface,
    colorscale: "Viridis",
    opacity: 0.88,
    showscale: false,
    contours: {
      z: {
        show: true,
        usecolormap: true,
        highlightcolor: "#ffffff",
        project: { z: true }
      }
    },
    hovertemplate:
      "x: %{x:.2f}<br>" +
      "y: %{y:.2f}<br>" +
      "f(x,y): %{z:.3f}<extra></extra>"
  };

  function makePointTrace(x, y) {
    const z = f(x, y);

    return {
      type: "scatter3d",
      mode: "markers",
      x: [x],
      y: [y],
      z: [z],
      marker: {
        size: 7,
        color: "#ff3b30",
        line: {
          color: "#ffffff",
          width: 2
        }
      },
      name: "Current point",
      hovertemplate:
        "Current point<br>" +
        "x: %{x:.2f}<br>" +
        "y: %{y:.2f}<br>" +
        "f(x,y): %{z:.3f}<extra></extra>"
    };
  }

  function makeArrowTrace(x, y, directionSign, color, name) {
    const z = f(x, y);
    const g = grad(x, y);
    const norm = Math.sqrt(g.x * g.x + g.y * g.y) || 1;

    const scale = 0.65;
    const dx = directionSign * scale * g.x / norm;
    const dy = directionSign * scale * g.y / norm;

    const xEnd = x + dx;
    const yEnd = y + dy;

    /*
      The arrow is drawn as a tangent direction on the surface.

      If u = grad / ||grad||, then the height change in the uphill
      direction is approximately grad dot u = ||grad||.
    */
    const zEnd = z + directionSign * scale * norm;

    return {
      type: "scatter3d",
      mode: "lines+markers",
      x: [x, xEnd],
      y: [y, yEnd],
      z: [z, zEnd],
      line: {
        color: color,
        width: 8
      },
      marker: {
        size: [1, 5],
        color: color
      },
      name: name,
      hoverinfo: "skip"
    };
  }

  function makeProjectedDescentTrace(x, y) {
    const z = f(x, y);
    const g = grad(x, y);
    const norm = Math.sqrt(g.x * g.x + g.y * g.y) || 1;

    const scale = 0.65;
    const xEnd = x - scale * g.x / norm;
    const yEnd = y - scale * g.y / norm;
    const zEnd = f(xEnd, yEnd);

    return {
      type: "scatter3d",
      mode: "lines",
      x: [x, xEnd],
      y: [y, yEnd],
      z: [z, zEnd],
      line: {
        color: "#ff3b30",
        width: 4,
        dash: "dot"
      },
      name: "Next small descent step",
      hoverinfo: "skip"
    };
  }

  function currentX() {
    return parseFloat(xSlider.value);
  }

  function currentY() {
    return parseFloat(ySlider.value);
  }

  function updateReadout(x, y) {
    const g = grad(x, y);

    xValue.textContent = x.toFixed(2);
    yValue.textContent = y.toFixed(2);
    lossValue.textContent = f(x, y).toFixed(4);
    gradValue.textContent = "(" + g.x.toFixed(3) + ", " + g.y.toFixed(3) + ")";
  }

  function makeData(x, y) {
    return [
      surfaceTrace,
      makePointTrace(x, y),
      makeArrowTrace(x, y, 1, "#ffffff", "Uphill direction ∇f"),
      makeArrowTrace(x, y, -1, "#111111", "Descent direction -∇f"),
      makeProjectedDescentTrace(x, y)
    ];
  }

  const layout = {
    margin: { l: 0, r: 0, b: 0, t: 0 },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    scene: {
      xaxis: {
        title: "x",
        backgroundcolor: "rgba(0,0,0,0)",
        gridcolor: "rgba(150,150,150,0.25)",
        zerolinecolor: "rgba(150,150,150,0.4)"
      },
      yaxis: {
        title: "y",
        backgroundcolor: "rgba(0,0,0,0)",
        gridcolor: "rgba(150,150,150,0.25)",
        zerolinecolor: "rgba(150,150,150,0.4)"
      },
      zaxis: {
        title: "f(x,y)",
        backgroundcolor: "rgba(0,0,0,0)",
        gridcolor: "rgba(150,150,150,0.25)",
        zerolinecolor: "rgba(150,150,150,0.4)"
      },
      camera: {
        eye: { x: 1.55, y: 1.55, z: 1.05 }
      },
      aspectratio: {
        x: 1,
        y: 1,
        z: 0.65
      }
    },
    showlegend: true,
legend: {
  orientation: "h",
  x: 0.5,
  y: -0.08,
  xanchor: "center",
  yanchor: "top",
  bgcolor: "rgba(255,255,255,0)",
  bordercolor: "rgba(150,150,150,0)",
  font: {
    size: 12,
    color: "#ffffff",
  }
}
  };

  const config = {
    responsive: true,
    displaylogo: false,
    scrollZoom: true
  };

  function render() {
    const x = currentX();
    const y = currentY();

    updateReadout(x, y);

    Plotly.react(plotDiv, makeData(x, y), layout, config);
  }

  Plotly.newPlot(plotDiv, makeData(currentX(), currentY()), layout, config);
  updateReadout(currentX(), currentY());

  xSlider.addEventListener("input", render);
  ySlider.addEventListener("input", render);

  plotDiv.on("plotly_click", function (data) {
    if (!data.points || data.points.length === 0) return;

    const p = data.points[0];

    if (typeof p.x === "number" && typeof p.y === "number") {
      xSlider.value = Math.max(-3, Math.min(3, p.x));
      ySlider.value = Math.max(-3, Math.min(3, p.y));
      render();
    }
  });
});
</script>

To make this intuition more tangible, try dragging the point below. The landscape represents a loss function $$f(x,y)$$. At every location, the gradient $$\nabla f(x,y)$$ tells us which direction is uphill, while gradient descent moves in the opposite direction, $$-\nabla f(x,y)$$.

---

## The Basic Geometric Idea

The gradient $\nabla f(x)$ points in the direction of steepest local increase of $f$. Therefore, the direction

$$
-\nabla f(x)
$$

is the direction of steepest local decrease. If we take a small step in this direction, we should expect the objective value to go down.

To make this intuition precise, we need a smoothness assumption.

---

## Smoothness and the Descent Lemma

A differentiable function $f$ is called $L$-smooth if its gradient is Lipschitz:

$$
\|\nabla f(x)-\nabla f(y)\| \le L\|x-y\|,
\qquad \forall x,y \in \mathbb{R}^d.
$$

Equivalently, an $L$-smooth function satisfies the upper quadratic bound

$$
f(y)
\le
f(x)
+
\langle \nabla f(x), y-x\rangle
+
\frac{L}{2}\|y-x\|^2.
$$

This inequality says that, near $x$, the function can be upper bounded by a quadratic model.

Now set

$$
y = x - \eta \nabla f(x).
$$

Plugging this into the smoothness inequality gives

$$
f(x-\eta \nabla f(x))
\le
f(x)
-
\eta \|\nabla f(x)\|^2
+
\frac{L\eta^2}{2}\|\nabla f(x)\|^2.
$$

Therefore,

$$
f(x_{t+1})
\le
f(x_t)
-
\eta\left(1-\frac{L\eta}{2}\right)
\|\nabla f(x_t)\|^2.
$$

If we choose

$$
0 < \eta \le \frac{1}{L},
$$

then

$$
1-\frac{L\eta}{2} \ge \frac12,
$$

and hence

$$
f(x_{t+1})
\le
f(x_t)
-
\frac{\eta}{2}\|\nabla f(x_t)\|^2.
$$

This is the key inequality behind gradient descent.

It says: unless the gradient is small, the objective must decrease.

---

## What This Gives in the Nonconvex Case

Assume $f$ is bounded below by $f^\star$. Summing the descent inequality from $t=0$ to $T-1$, we get

$$
\sum_{t=0}^{T-1}
\frac{\eta}{2}\|\nabla f(x_t)\|^2
\le
f(x_0)-f(x_T)
\le
f(x_0)-f^\star.
$$

Therefore,

$$
\sum_{t=0}^{T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta}.
$$

Dividing by $T$,

$$
\frac1T
\sum_{t=0}^{T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta T}.
$$

Thus,

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|^2
\le
\frac{2(f(x_0)-f^\star)}{\eta T}.
$$

So gradient descent finds an approximate stationary point at rate

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|^2
=
O\left(\frac1T\right).
$$

Equivalently,

$$
\min_{0\le t\le T-1}
\|\nabla f(x_t)\|
=
O\left(\frac1{\sqrt{T}}\right).
$$

This is the standard nonconvex gradient descent guarantee.

---

## Convexity Gives More Structure

Now suppose $f$ is convex. Convexity means

$$
f(y) \ge f(x) + \langle \nabla f(x), y-x\rangle,
\qquad \forall x,y.
$$

Let $x^\star$ be a minimizer. Setting $y=x^\star$, we get

$$
f(x_t)-f(x^\star)
\le
\langle \nabla f(x_t), x_t-x^\star\rangle.
$$

Now look at the squared distance to the optimum:

$$
\|x_{t+1}-x^\star\|^2
=
\|x_t-\eta \nabla f(x_t)-x^\star\|^2.
$$

Expanding,

$$
\|x_{t+1}-x^\star\|^2
=
\|x_t-x^\star\|^2
-
2\eta \langle \nabla f(x_t),x_t-x^\star\rangle
+
\eta^2\|\nabla f(x_t)\|^2.
$$

Using convexity,

$$
\langle \nabla f(x_t),x_t-x^\star\rangle
\ge
f(x_t)-f(x^\star).
$$

Therefore,

$$
2\eta(f(x_t)-f(x^\star))
\le
\|x_t-x^\star\|^2
-
\|x_{t+1}-x^\star\|^2
+
\eta^2\|\nabla f(x_t)\|^2.
$$

For smooth convex functions, a sharper analysis with $\eta=1/L$ gives

$$
f(x_T)-f(x^\star)
\le
\frac{L\|x_0-x^\star\|^2}{2T}.
$$

Thus, in the convex smooth setting, gradient descent converges in function value at rate

$$
f(x_T)-f(x^\star)
=
O\left(\frac1T\right).
$$

---

## Strong Convexity Gives Linear Convergence

The most elegant result appears when $f$ is both smooth and strongly convex.

A function is $\mu$-strongly convex if

$$
f(y)
\ge
f(x)
+
\langle \nabla f(x),y-x\rangle
+
\frac{\mu}{2}\|y-x\|^2.
$$

Strong convexity says that the function has curvature everywhere. It rules out flat directions and ensures a unique minimizer.

For an $L$-smooth and $\mu$-strongly convex function, gradient descent with step-size $\eta=1/L$ satisfies

$$
\|x_{t+1}-x^\star\|
\le
\left(1-\frac{\mu}{L}\right)
\|x_t-x^\star\|.
$$

Iterating,

$$
\|x_t-x^\star\|
\le
\left(1-\frac{\mu}{L}\right)^t
\|x_0-x^\star\|.
$$

Thus, gradient descent converges geometrically:

$$
\|x_t-x^\star\|
=
O\left(
\left(1-\frac{\mu}{L}\right)^t
\right).
$$

The ratio

$$
\kappa = \frac{L}{\mu}
$$

is called the condition number. A smaller condition number means faster convergence; a larger condition number means the objective is poorly conditioned and gradient descent may move slowly.

---

## The Quadratic Case

A particularly transparent example is

$$
f(x)=\frac12 x^\top A x - b^\top x,
$$

where $A$ is symmetric positive definite. Then

$$
\nabla f(x)=Ax-b.
$$

The minimizer satisfies

$$
Ax^\star=b.
$$

Gradient descent becomes

$$
x_{t+1}
=
x_t-\eta(Ax_t-b).
$$

Subtracting $x^\star$, and using $Ax^\star=b$,

$$
x_{t+1}-x^\star
=
(I-\eta A)(x_t-x^\star).
$$

Therefore,

$$
x_t-x^\star
=
(I-\eta A)^t(x_0-x^\star).
$$

The convergence is controlled by the eigenvalues of $I-\eta A$. If the eigenvalues of $A$ lie in

$$
\mu \le \lambda_i(A) \le L,
$$

then choosing $0<\eta<2/L$ ensures convergence. The best fixed step-size is

$$
\eta^\star = \frac{2}{L+\mu},
$$

which gives contraction factor

$$
\frac{L-\mu}{L+\mu}
=
\frac{\kappa-1}{\kappa+1}.
$$

This explains why ill-conditioned problems are hard: when $\kappa$ is large,

$$
\frac{\kappa-1}{\kappa+1}
\approx 1,
$$

so the error shrinks very slowly.

---

## Takeaway

Gradient descent works because smoothness turns the first-order Taylor approximation into a reliable upper bound. Moving against the gradient decreases this upper bound, and therefore decreases the objective.

The basic story is:

$$
x_{t+1}=x_t-\eta\nabla f(x_t)
$$

combined with

$$
f(x_{t+1})
\le
f(x_t)
-
\frac{\eta}{2}\|\nabla f(x_t)\|^2.
$$

From this one inequality, we get:

- for nonconvex smooth functions, convergence to stationary points;
- for convex smooth functions, $O(1/T)$ convergence in function value;
- for strongly convex smooth functions, linear convergence;
- for quadratic functions, convergence determined exactly by eigenvalues.

This is why gradient descent remains one of the central algorithms in optimization and machine learning: it is simple enough to write in one line, yet rich enough to reveal the geometry of learning.
