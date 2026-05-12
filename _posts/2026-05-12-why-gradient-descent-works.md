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

The entire algorithm is contained in this single line. At first glance, this update looks almost too simple to matter. We begin at a point \(x_t\), compute the gradient \(\nabla f(x_t)\), and then take a small step in the opposite direction. The number \(\eta>0\) is called the **learning rate** or **step size**, and it controls how far we move at each iteration.

The intuition is wonderfully geometric. The gradient points in the direction where the function increases the fastest. So, if our goal is to make the function smaller, the most natural thing to do is to walk in the opposite direction. Gradient descent is simply the act of repeatedly asking:

> Where is downhill from here?

and then taking a small step that way.

This is what makes the method so elegant. Gradient descent does not require us to understand the entire landscape of the function. It only uses local information: the slope at the current point. Yet, by repeating this local rule again and again, the iterates can move toward a minimizer.

Of course, the size of the step matters. If $$\eta$$ is too large, we may overshoot the minimum and oscillate. If $$\eta$$ is too small, we may move in the right direction but painfully slowly. But under suitable assumptions, such as smoothness and convexity, this one-line update can be proved to converge.

That is the beauty of gradient descent: a global optimization problem is attacked through a purely local rule. The convergence proof makes this intuition precise, and reveals why this deceptively simple algorithm sits at the heart of modern machine learning.

<div class="gd-widget">
  <div class="gd-card">
    <div class="gd-header">
      <div>
        <h3>Explore the Gradient</h3>
        <p>Drag the red point on the landscape. The arrow shows the local gradient direction.</p>
      </div>
      <button class="gd-reset">Reset</button>
    </div>

    <canvas class="gd-canvas"></canvas>

    <div class="gd-readout">
      <span><strong>Point:</strong> <span class="gd-point"></span></span>
      <span><strong>Loss:</strong> <span class="gd-loss"></span></span>
      <span><strong>Gradient:</strong> <span class="gd-gradient"></span></span>
    </div>

    <p class="gd-caption">
      The gradient points in the direction of steepest increase. Gradient descent moves in the opposite direction.
    </p>
  </div>
</div>

<style>
  .gd-widget {
    margin: 2rem 0;
    font-family: inherit;
  }

  .gd-card {
    border: 1px solid rgba(150, 150, 150, 0.25);
    border-radius: 18px;
    padding: 1.2rem;
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
  }

  .gd-header {
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    align-items: flex-start;
    margin-bottom: 1rem;
  }

  .gd-header h3 {
    margin: 0;
    font-size: 1.35rem;
  }

  .gd-header p {
    margin: 0.35rem 0 0;
    opacity: 0.78;
    font-size: 0.95rem;
  }

  .gd-reset {
    border: 1px solid rgba(150,150,150,0.35);
    border-radius: 999px;
    padding: 0.45rem 0.8rem;
    background: transparent;
    color: inherit;
    cursor: pointer;
    font-size: 0.9rem;
  }

  .gd-reset:hover {
    background: rgba(150,150,150,0.12);
  }

  .gd-canvas {
    width: 100%;
    height: 420px;
    display: block;
    border-radius: 14px;
    border: 1px solid rgba(150,150,150,0.25);
    cursor: grab;
    touch-action: none;
  }

  .gd-canvas:active {
    cursor: grabbing;
  }

  .gd-readout {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
    font-size: 0.92rem;
  }

  .gd-readout span {
    padding: 0.6rem 0.75rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
  }

  .gd-caption {
    margin: 1rem 0 0;
    font-size: 0.95rem;
    opacity: 0.82;
  }

  @media (max-width: 700px) {
    .gd-readout {
      grid-template-columns: 1fr;
    }

    .gd-canvas {
      height: 340px;
    }
  }
</style>

<script>
(function () {
  const root = document.currentScript.closest(".gd-widget");
  const canvas = root.querySelector(".gd-canvas");
  const ctx = canvas.getContext("2d");

  const pointText = root.querySelector(".gd-point");
  const lossText = root.querySelector(".gd-loss");
  const gradientText = root.querySelector(".gd-gradient");
  const resetButton = root.querySelector(".gd-reset");

  const xMin = -3;
  const xMax = 3;
  const yMin = -3;
  const yMax = 3;

  let width = 0;
  let height = 0;
  let dpr = window.devicePixelRatio || 1;
  let dragging = false;

  let point = { x: 1.25, y: 1.0 };

  function f(x, y) {
    return 0.35 * (x * x + 0.75 * y * y) + Math.sin(2 * x) * Math.cos(1.5 * y);
  }

  function grad(x, y) {
    return {
      x: 0.7 * x + 2 * Math.cos(2 * x) * Math.cos(1.5 * y),
      y: 0.525 * y - 1.5 * Math.sin(2 * x) * Math.sin(1.5 * y)
    };
  }

  function toScreen(x, y) {
    return {
      x: ((x - xMin) / (xMax - xMin)) * width,
      y: height - ((y - yMin) / (yMax - yMin)) * height
    };
  }

  function toWorld(px, py) {
    return {
      x: xMin + (px / width) * (xMax - xMin),
      y: yMin + ((height - py) / height) * (yMax - yMin)
    };
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function colorMap(t) {
    t = clamp(t, 0, 1);

    const r = Math.round(35 + 190 * t);
    const g = Math.round(70 + 110 * (1 - Math.abs(t - 0.5) * 2));
    const b = Math.round(150 + 80 * (1 - t));

    return `rgb(${r}, ${g}, ${b})`;
  }

  function drawArrow(start, end, color, label) {
    const headLength = 11;
    const angle = Math.atan2(end.y - start.y, end.x - start.x);

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = 3;

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(
      end.x - headLength * Math.cos(angle - Math.PI / 6),
      end.y - headLength * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      end.x - headLength * Math.cos(angle + Math.PI / 6),
      end.y - headLength * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();

    ctx.font = "13px system-ui, sans-serif";
    ctx.fillText(label, end.x + 8, end.y - 8);

    ctx.restore();
  }

  function drawLandscape() {
    let minVal = Infinity;
    let maxVal = -Infinity;

    const sampleStep = 6;

    for (let py = 0; py < height; py += sampleStep) {
      for (let px = 0; px < width; px += sampleStep) {
        const w = toWorld(px, py);
        const z = f(w.x, w.y);
        minVal = Math.min(minVal, z);
        maxVal = Math.max(maxVal, z);
      }
    }

    for (let py = 0; py < height; py += sampleStep) {
      for (let px = 0; px < width; px += sampleStep) {
        const w = toWorld(px, py);
        const z = f(w.x, w.y);
        const t = (z - minVal) / (maxVal - minVal);
        ctx.fillStyle = colorMap(t);
        ctx.fillRect(px, py, sampleStep + 1, sampleStep + 1);
      }
    }

    drawContourLines(minVal, maxVal);
  }

  function drawContourLines(minVal, maxVal) {
    const nx = 70;
    const ny = 48;
    const values = [];

    for (let j = 0; j <= ny; j++) {
      values[j] = [];
      for (let i = 0; i <= nx; i++) {
        const x = xMin + (i / nx) * (xMax - xMin);
        const y = yMin + (j / ny) * (yMax - yMin);
        values[j][i] = f(x, y);
      }
    }

    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;

    const levels = 13;

    for (let levelIndex = 1; levelIndex < levels; levelIndex++) {
      const level = minVal + (levelIndex / levels) * (maxVal - minVal);

      for (let j = 0; j < ny; j++) {
        for (let i = 0; i < nx; i++) {
          const x0 = xMin + (i / nx) * (xMax - xMin);
          const x1 = xMin + ((i + 1) / nx) * (xMax - xMin);
          const y0 = yMin + (j / ny) * (yMax - yMin);
          const y1 = yMin + ((j + 1) / ny) * (yMax - yMin);

          const v00 = values[j][i];
          const v10 = values[j][i + 1];
          const v01 = values[j + 1][i];
          const v11 = values[j + 1][i + 1];

          const points = [];

          function cross(a, b) {
            return (a - level) * (b - level) <= 0 && a !== b;
          }

          function interp(a, b) {
            return (level - a) / (b - a);
          }

          if (cross(v00, v10)) {
            const t = interp(v00, v10);
            points.push(toScreen(x0 + t * (x1 - x0), y0));
          }

          if (cross(v10, v11)) {
            const t = interp(v10, v11);
            points.push(toScreen(x1, y0 + t * (y1 - y0)));
          }

          if (cross(v01, v11)) {
            const t = interp(v01, v11);
            points.push(toScreen(x0 + t * (x1 - x0), y1));
          }

          if (cross(v00, v01)) {
            const t = interp(v00, v01);
            points.push(toScreen(x0, y0 + t * (y1 - y0)));
          }

          if (points.length >= 2) {
            ctx.beginPath();
            ctx.moveTo(points[0].x, points[0].y);
            ctx.lineTo(points[1].x, points[1].y);
            ctx.stroke();
          }
        }
      }
    }

    ctx.restore();
  }

  function drawPointAndGradient() {
    const p = toScreen(point.x, point.y);
    const g = grad(point.x, point.y);
    const norm = Math.sqrt(g.x * g.x + g.y * g.y) || 1;

    const arrowScale = 0.55;

    const gradEnd = toScreen(
      point.x + arrowScale * g.x / norm,
      point.y + arrowScale * g.y / norm
    );

    const descentEnd = toScreen(
      point.x - arrowScale * g.x / norm,
      point.y - arrowScale * g.y / norm
    );

    drawArrow(p, gradEnd, "rgba(255,255,255,0.95)", "∇f");
    drawArrow(p, descentEnd, "rgba(20,20,20,0.85)", "-∇f");

    ctx.save();
    ctx.beginPath();
    ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI);
    ctx.fillStyle = "#ff3b30";
    ctx.fill();

    ctx.lineWidth = 3;
    ctx.strokeStyle = "white";
    ctx.stroke();
    ctx.restore();

    pointText.textContent = `(${point.x.toFixed(2)}, ${point.y.toFixed(2)})`;
    lossText.textContent = f(point.x, point.y).toFixed(3);
    gradientText.textContent = `(${g.x.toFixed(3)}, ${g.y.toFixed(3)})`;
  }

  function drawAxes() {
    ctx.save();
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.lineWidth = 1;

    const xAxisLeft = toScreen(xMin, 0);
    const xAxisRight = toScreen(xMax, 0);
    const yAxisBottom = toScreen(0, yMin);
    const yAxisTop = toScreen(0, yMax);

    ctx.beginPath();
    ctx.moveTo(xAxisLeft.x, xAxisLeft.y);
    ctx.lineTo(xAxisRight.x, xAxisRight.y);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(yAxisBottom.x, yAxisBottom.y);
    ctx.lineTo(yAxisTop.x, yAxisTop.y);
    ctx.stroke();

    ctx.restore();
  }

  function draw() {
    ctx.clearRect(0, 0, width, height);
    drawLandscape();
    drawAxes();
    drawPointAndGradient();
  }

  function resize() {
    const rect = canvas.getBoundingClientRect();
    width = Math.floor(rect.width);
    height = Math.floor(rect.height);
    dpr = window.devicePixelRatio || 1;

    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    draw();
  }

  function updatePointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    const w = toWorld(px, py);

    point.x = clamp(w.x, xMin, xMax);
    point.y = clamp(w.y, yMin, yMax);

    draw();
  }

  canvas.addEventListener("pointerdown", function (event) {
    dragging = true;
    canvas.setPointerCapture(event.pointerId);
    updatePointFromEvent(event);
  });

  canvas.addEventListener("pointermove", function (event) {
    if (dragging) {
      updatePointFromEvent(event);
    }
  });

  canvas.addEventListener("pointerup", function () {
    dragging = false;
  });

  canvas.addEventListener("pointercancel", function () {
    dragging = false;
  });

  resetButton.addEventListener("click", function () {
    point = { x: 1.25, y: 1.0 };
    draw();
  });

  window.addEventListener("resize", resize);

  resize();
})();
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
