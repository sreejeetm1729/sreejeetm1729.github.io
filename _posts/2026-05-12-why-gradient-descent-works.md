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

<div id="gd-three-widget">
  <div class="gd-three-card">
    <h3>Drag the Point on the Loss Landscape</h3>
    <p>
      Rotate the surface to understand the geometry. Then switch to move-point mode
      and drag the red point across the landscape.
    </p>

    <div class="gd-three-buttons">
      <button id="gd-rotate-mode">Rotate mode</button>
      <button id="gd-move-mode">Move point mode</button>
    </div>

    <div id="gd-three-scene"></div>

    <div class="gd-three-readout">
      <span><strong>Point:</strong> <span id="gd-three-point"></span></span>
      <span><strong>Loss:</strong> <span id="gd-three-loss"></span></span>
      <span><strong>Gradient:</strong> <span id="gd-three-grad"></span></span>
    </div>

    <p class="gd-three-caption">
      The white arrow points uphill in the direction of \(\nabla f(x,y)\).
      The black arrow points downhill in the gradient descent direction \(-\nabla f(x,y)\).
    </p>
  </div>
</div>

<style>
  #gd-three-widget {
    margin: 2rem 0;
    font-family: inherit;
  }

  #gd-three-widget .gd-three-card {
    border: 1px solid rgba(150, 150, 150, 0.25);
    border-radius: 18px;
    padding: 1.2rem;
    background: rgba(255, 255, 255, 0.04);
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
  }

  #gd-three-widget h3 {
    margin: 0;
    font-size: 1.35rem;
  }

  #gd-three-widget p {
    margin: 0.4rem 0 1rem;
    opacity: 0.84;
  }

  #gd-three-scene {
    width: 100%;
    height: 520px;
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(150,150,150,0.25);
    background: #111;
  }

  #gd-three-widget .gd-three-buttons {
    display: flex;
    gap: 0.7rem;
    margin: 1rem 0;
  }

  #gd-three-widget button {
    border: 1px solid rgba(150,150,150,0.35);
    border-radius: 999px;
    padding: 0.5rem 0.9rem;
    background: transparent;
    color: inherit;
    cursor: pointer;
  }

  #gd-three-widget button.active {
    background: rgba(150,150,150,0.22);
  }

  #gd-three-widget .gd-three-readout {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
    margin-top: 1rem;
    font-size: 0.92rem;
  }

  #gd-three-widget .gd-three-readout span {
    padding: 0.65rem 0.8rem;
    border-radius: 12px;
    background: rgba(150,150,150,0.10);
  }

  #gd-three-widget .gd-three-caption {
    margin-top: 1rem;
    font-size: 0.95rem;
  }

  @media (max-width: 700px) {
    #gd-three-scene {
      height: 420px;
    }

    #gd-three-widget .gd-three-readout {
      grid-template-columns: 1fr;
    }
  }
</style>

<script type="module">
  import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
  import { OrbitControls } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js";

  const container = document.getElementById("gd-three-scene");

  const pointText = document.getElementById("gd-three-point");
  const lossText = document.getElementById("gd-three-loss");
  const gradText = document.getElementById("gd-three-grad");

  const rotateButton = document.getElementById("gd-rotate-mode");
  const moveButton = document.getElementById("gd-move-mode");

  let moveMode = false;
  let draggingPoint = false;

  rotateButton.classList.add("active");

  function f(x, y) {
    return 0.35 * (x * x + 0.75 * y * y) + Math.sin(2 * x) * Math.cos(1.5 * y);
  }

  function grad(x, y) {
    return {
      x: 0.7 * x + 2 * Math.cos(2 * x) * Math.cos(1.5 * y),
      y: 0.525 * y - 1.5 * Math.sin(2 * x) * Math.sin(1.5 * y)
    };
  }

  const zScale = 0.65;
  const xMin = -3;
  const xMax = 3;
  const yMin = -3;
  const yMax = 3;

  let currentPoint = { x: 1.2, y: 1.0 };

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111111);

  const camera = new THREE.PerspectiveCamera(
    45,
    container.clientWidth / container.clientHeight,
    0.1,
    100
  );

  camera.position.set(5.5, -6.5, 4.5);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(container.clientWidth, container.clientHeight);
  renderer.setPixelRatio(window.devicePixelRatio);
  container.appendChild(renderer.domElement);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;

  const ambientLight = new THREE.AmbientLight(0xffffff, 0.75);
  scene.add(ambientLight);

  const directionalLight = new THREE.DirectionalLight(0xffffff, 1.0);
  directionalLight.position.set(5, -4, 8);
  scene.add(directionalLight);

  const grid = new THREE.GridHelper(7, 14, 0x666666, 0x333333);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = -1.1;
  scene.add(grid);

  const axes = new THREE.AxesHelper(3.5);
  scene.add(axes);

  function createSurface() {
    const n = 95;
    const geometry = new THREE.BufferGeometry();

    const vertices = [];
    const colors = [];
    const indices = [];

    let zMin = Infinity;
    let zMax = -Infinity;

    const values = [];

    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        const x = xMin + (i / (n - 1)) * (xMax - xMin);
        const y = yMin + (j / (n - 1)) * (yMax - yMin);
        const z = zScale * f(x, y);

        values.push(z);
        zMin = Math.min(zMin, z);
        zMax = Math.max(zMax, z);

        vertices.push(x, y, z);
      }
    }

    for (let k = 0; k < values.length; k++) {
      const t = (values[k] - zMin) / (zMax - zMin);

      const color = new THREE.Color();
      color.setHSL(0.67 - 0.55 * t, 0.85, 0.55);

      colors.push(color.r, color.g, color.b);
    }

    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        const a = j * n + i;
        const b = j * n + i + 1;
        const c = (j + 1) * n + i;
        const d = (j + 1) * n + i + 1;

        indices.push(a, b, c);
        indices.push(b, d, c);
      }
    }

    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3)
    );

    geometry.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(colors, 3)
    );

    geometry.setIndex(indices);
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      roughness: 0.65,
      metalness: 0.05,
      transparent: true,
      opacity: 0.95
    });

    const mesh = new THREE.Mesh(geometry, material);
    return mesh;
  }

  const surface = createSurface();
  scene.add(surface);

  const wireframe = new THREE.WireframeGeometry(surface.geometry);
  const wireMaterial = new THREE.LineBasicMaterial({
    color: 0xffffff,
    transparent: true,
    opacity: 0.15
  });
  const wire = new THREE.LineSegments(wireframe, wireMaterial);
  scene.add(wire);

  const pointGeometry = new THREE.SphereGeometry(0.11, 32, 32);
  const pointMaterial = new THREE.MeshStandardMaterial({
    color: 0xff3b30,
    emissive: 0x551000,
    roughness: 0.35
  });

  const redPoint = new THREE.Mesh(pointGeometry, pointMaterial);
  scene.add(redPoint);

  let uphillArrow = null;
  let downhillArrow = null;

  function removeOldArrows() {
    if (uphillArrow) scene.remove(uphillArrow);
    if (downhillArrow) scene.remove(downhillArrow);
  }

  function updatePointAndArrows() {
    const x = currentPoint.x;
    const y = currentPoint.y;
    const z = zScale * f(x, y);

    redPoint.position.set(x, y, z + 0.08);

    const g = grad(x, y);
    const norm = Math.sqrt(g.x * g.x + g.y * g.y) || 1;

    removeOldArrows();

    const origin = new THREE.Vector3(x, y, z + 0.16);

    const uphillDirection = new THREE.Vector3(
      g.x / norm,
      g.y / norm,
      zScale * norm
    ).normalize();

    const downhillDirection = new THREE.Vector3(
      -g.x / norm,
      -g.y / norm,
      -zScale * norm
    ).normalize();

    uphillArrow = new THREE.ArrowHelper(
      uphillDirection,
      origin,
      0.9,
      0xffffff,
      0.18,
      0.1
    );

    downhillArrow = new THREE.ArrowHelper(
      downhillDirection,
      origin,
      0.9,
      0x000000,
      0.18,
      0.1
    );

    scene.add(uphillArrow);
    scene.add(downhillArrow);

    pointText.textContent = `(${x.toFixed(2)}, ${y.toFixed(2)})`;
    lossText.textContent = f(x, y).toFixed(4);
    gradText.textContent = `(${g.x.toFixed(3)}, ${g.y.toFixed(3)})`;
  }

  updatePointAndArrows();

  const raycaster = new THREE.Raycaster();
  const mouse = new THREE.Vector2();

  function setMouseFromEvent(event) {
    const rect = renderer.domElement.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  }

  function movePointFromEvent(event) {
    setMouseFromEvent(event);

    raycaster.setFromCamera(mouse, camera);

    const hits = raycaster.intersectObject(surface);

    if (hits.length > 0) {
      const p = hits[0].point;

      currentPoint.x = Math.max(xMin, Math.min(xMax, p.x));
      currentPoint.y = Math.max(yMin, Math.min(yMax, p.y));

      updatePointAndArrows();
    }
  }

  renderer.domElement.addEventListener("pointerdown", function (event) {
    if (!moveMode) return;

    draggingPoint = true;
    controls.enabled = false;
    movePointFromEvent(event);
  });

  renderer.domElement.addEventListener("pointermove", function (event) {
    if (!moveMode || !draggingPoint) return;

    movePointFromEvent(event);
  });

  renderer.domElement.addEventListener("pointerup", function () {
    draggingPoint = false;

    if (!moveMode) {
      controls.enabled = true;
    }
  });

  renderer.domElement.addEventListener("pointerleave", function () {
    draggingPoint = false;
  });

  rotateButton.addEventListener("click", function () {
    moveMode = false;
    controls.enabled = true;

    rotateButton.classList.add("active");
    moveButton.classList.remove("active");
  });

  moveButton.addEventListener("click", function () {
    moveMode = true;
    controls.enabled = false;

    moveButton.classList.add("active");
    rotateButton.classList.remove("active");
  });

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  }

  animate();

  window.addEventListener("resize", function () {
    const width = container.clientWidth;
    const height = container.clientHeight;

    camera.aspect = width / height;
    camera.updateProjectionMatrix();

    renderer.setSize(width, height);
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
