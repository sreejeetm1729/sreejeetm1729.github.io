---
title: Learning Agents
icon: fas fa-robot
order: 8
---

This page hosts small interactive environments for building intuition about learning, control, rewards, disturbances, and task completion. 
<style>
.learning-agents-copy {
  margin: 0.25rem 0 1.25rem 0;
  color: var(--text-color, #334155);
}

.learning-agents-copy h2 {
  margin-bottom: 0.55rem;
  color: var(--heading-color, #111827);
}

.learning-agents-copy p {
  line-height: 1.65;
  color: var(--text-color, #334155);
}


#rl3d-drone-widget {
      max-width: 1120px;
      width: 100%;
      margin: 1.25rem auto 2rem auto;
      padding: 1.15rem;
      border-radius: 18px;
      background: var(--card-bg, var(--main-bg, #ffffff));
      color: var(--text-color, #334155);
      border: 1px solid var(--card-border-color, rgba(148, 163, 184, 0.25));
      box-shadow: var(--card-shadow, 0 12px 30px rgba(15, 23, 42, 0.10));
      font-family: inherit;
      overflow: visible;
    }

    #rl3d-drone-widget * {
      box-sizing: border-box;
    }

    .rl3d-header h2 {
      margin: 0 0 0.45rem 0;
      color: var(--heading-color, #111827);
      font-size: 1.55rem;
      font-weight: 850;
    }

    .rl3d-header p {
      margin: 0 0 1.1rem 0;
      color: var(--text-muted-color, #64748b);
      line-height: 1.6;
    }

    .rl3d-toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.8rem;
      margin-bottom: 1rem;
    }

    .rl3d-toolbar label {
      display: flex;
      flex-direction: column;
      gap: 0.35rem;
      color: var(--text-muted-color, #64748b);
      font-size: 0.85rem;
      font-weight: 800;
    }

    .rl3d-toolbar select {
      min-width: 220px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 999px;
      padding: 0.65rem 0.85rem;
      color: var(--text-color, #334155);
      background: var(--main-bg, #ffffff);
      font-weight: 800;
      outline: none;
    }

    #rl3d-stage-wrap {
      isolation: isolate;
      position: relative;
      width: 100%;
      height: 620px;
      overflow: hidden;
      border-radius: 18px;
      background: #020617;
      border: 1px solid rgba(255, 255, 255, 0.1);
    }

    #rl3d-stage {
      width: 100%;
      height: 100%;
    }

    #rl3d-stage canvas {
      display: block;
      width: 100%;
      height: 100%;
      cursor: crosshair;
    }

    #rl3d-top-hint {
      position: absolute;
      left: 18px;
      top: 16px;
      padding: 0.58rem 0.8rem;
      border-radius: 999px;
      background: rgba(15, 23, 42, 0.72);
      color: #e2e8f0;
      font-size: 0.9rem;
      font-weight: 800;
      backdrop-filter: blur(8px);
      pointer-events: none;
    }

    #rl3d-task-card {
      position: absolute;
      left: 18px;
      bottom: 18px;
      max-width: 330px;
      padding: 0.8rem 0.95rem;
      border-radius: 18px;
      background: rgba(15, 23, 42, 0.68);
      border: 1px solid rgba(255, 255, 255, 0.16);
      backdrop-filter: blur(10px);
      pointer-events: none;
    }

    .rl3d-task-title {
      color: var(--heading-color, #111827);
      font-weight: 900;
      font-size: 0.98rem;
      margin-bottom: 0.2rem;
    }

    .rl3d-task-desc {
      color: var(--text-muted-color, #64748b);
      font-size: 0.86rem;
      line-height: 1.4;
    }

    #rl3d-compass-card {
      position: absolute;
      right: 18px;
      bottom: 18px;
      width: 150px;
      height: 150px;
      border-radius: 22px;
      background: rgba(15, 23, 42, 0.56);
      border: 1px solid rgba(255, 255, 255, 0.18);
      backdrop-filter: blur(10px);
      box-shadow: 0 14px 30px rgba(15, 23, 42, 0.32);
      overflow: hidden;
      pointer-events: none;
    }

    #rl3d-compass {
      width: 150px;
      height: 128px;
    }

    .rl3d-compass-caption {
      position: absolute;
      left: 0;
      right: 0;
      bottom: 9px;
      text-align: center;
      color: #fed7aa;
      font-size: 0.76rem;
      font-weight: 900;
      letter-spacing: 0.08em;
    }

    .rl3d-controls {
      margin-top: 0.9rem;
      display: flex;
      flex-wrap: wrap;
      gap: 0.65rem;
    }

    .rl3d-controls button {
      border: none;
      border-radius: 999px;
      padding: 0.72rem 1.05rem;
      cursor: pointer;
      font-weight: 850;
      color: #0f172a;
      background: #93c5fd;
      transition: transform 0.12s ease, opacity 0.12s ease;
    }

    .rl3d-controls button:hover {
      transform: translateY(-1px);
      opacity: 0.94;
    }

    #rl3d-policy-btn {
      background: #86efac;
    }

    #rl3d-wind-btn {
      background: #facc15;
    }

    .rl3d-stats {
      margin-top: 0.9rem;
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 0.65rem;
    }

    .rl3d-stats div {
      padding: 0.78rem;
      border-radius: 15px;
      background: rgba(148, 163, 184, 0.08);
      border: 1px solid rgba(148, 163, 184, 0.18);
    }

    .rl3d-stats span {
      display: block;
      color: var(--text-muted-color, #64748b);
      font-size: 0.82rem;
      margin-bottom: 0.25rem;
    }

    .rl3d-stats strong {
      color: var(--heading-color, #111827);
      font-size: 1rem;
    }

    @media (max-width: 850px) {
      #rl3d-stage-wrap {
        height: 480px;
      }

      .rl3d-stats {
        grid-template-columns: repeat(2, minmax(120px, 1fr));
      }

      #rl3d-compass-card {
        width: 128px;
        height: 128px;
      }

      #rl3d-compass {
        width: 128px;
        height: 108px;
      }

      .rl3d-toolbar select {
        min-width: 180px;
      }
    }
</style>

<div class="learning-agents-copy">

</div>

<div id="rl3d-drone-widget">
    <div class="rl3d-header">
      <h2>3D Drone RL Playground</h2>
      <p>
         The demo below is a 3D drone playground where the drone must compensate for inertia and wind while solving simple reinforcement-learning-style tasks such as hovering, cargo delivery, and waypoint tracking. Move your mouse over the arena to control the drone. Try different environments and tasks:
        hover, deliver cargo, or visit waypoints. The drone receives reward for completing the task while compensating for wind and inertia.
      </p>
    </div>

    <div class="rl3d-toolbar">
      <label>
        Environment
        <select id="rl3d-env-select">
          <option value="lab">Robotics Lab</option>
          <option value="delivery">Delivery Arena</option>
          <option value="wind">Wind Tunnel</option>
          <option value="sky">Sky Park</option>
          <option value="warehouse">Warehouse</option>
          <option value="neon">Neon City</option>
        </select>
      </label>

      <label>
        Task
        <select id="rl3d-task-select">
          <option value="hover">Hover in Target Zone</option>
          <option value="cargo">Pick Up and Deliver Cargo</option>
          <option value="waypoints">Visit Waypoints</option>
        </select>
      </label>
    </div>

    <div id="rl3d-stage-wrap">
      <div id="rl3d-stage"></div>

      <div id="rl3d-top-hint">
        Move cursor on arena → drone accelerates toward that point
      </div>

      <div id="rl3d-task-card">
        <div class="rl3d-task-title" id="rl3d-task-title">Hover in Target Zone</div>
        <div class="rl3d-task-desc" id="rl3d-task-desc">
          Keep the drone inside the moving green target cylinder.
        </div>
      </div>

      <div id="rl3d-compass-card">
        <div id="rl3d-compass"></div>
        <div class="rl3d-compass-caption">WIND</div>
      </div>
    </div>

    <div class="rl3d-controls">
      <button id="rl3d-reset-btn">Reset</button>
      <button id="rl3d-policy-btn">Watch Demo Controller</button>
      <button id="rl3d-wind-btn">Toggle Wind</button>
    </div>

    <div class="rl3d-stats">
      <div>
        <span>Mode</span>
        <strong id="rl3d-mode">Human control</strong>
      </div>
      <div>
        <span>Total reward</span>
        <strong id="rl3d-total-reward">0.0</strong>
      </div>
      <div>
        <span>Instant reward</span>
        <strong id="rl3d-instant-reward">0.00</strong>
      </div>
      <div>
        <span>Progress</span>
        <strong id="rl3d-progress">0.0s</strong>
      </div>
      <div>
        <span>Wind</span>
        <strong id="rl3d-wind-text">On</strong>
      </div>
    </div>
  </div>

<script type="module">
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

    const stage = document.getElementById("rl3d-stage");
    const compassStage = document.getElementById("rl3d-compass");

    const envSelect = document.getElementById("rl3d-env-select");
    const taskSelect = document.getElementById("rl3d-task-select");

    const resetBtn = document.getElementById("rl3d-reset-btn");
    const policyBtn = document.getElementById("rl3d-policy-btn");
    const windBtn = document.getElementById("rl3d-wind-btn");

    const modeText = document.getElementById("rl3d-mode");
    const totalRewardText = document.getElementById("rl3d-total-reward");
    const instantRewardText = document.getElementById("rl3d-instant-reward");
    const progressText = document.getElementById("rl3d-progress");
    const windText = document.getElementById("rl3d-wind-text");

    const taskTitle = document.getElementById("rl3d-task-title");
    const taskDesc = document.getElementById("rl3d-task-desc");

    const ARENA_X = 8.0;
    const ARENA_Z = 5.2;
    const DRONE_ALTITUDE = 1.35;

    let currentEnv = "lab";
    let currentTask = "hover";

    let droneState;
    let controlPoint;
    let mouseActive = false;
    let policyMode = false;
    let windOn = true;

    let elapsed = 0;
    let totalReward = 0;
    let instantReward = 0;
    let progressLabel = "0.0s";

    let taskState = {};
    let obstacles = [];
    let animatedObjects = [];

    const clock = new THREE.Clock();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xdbeafe);
    scene.fog = new THREE.Fog(0xdbeafe, 14, 42);

    const camera = new THREE.PerspectiveCamera(52, 1, 0.1, 100);
    camera.position.set(0, 8.5, 11.8);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    stage.appendChild(renderer.domElement);

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);

    const ambient = new THREE.HemisphereLight(0xffffff, 0x64748b, 1.55);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xffffff, 2.4);
    sun.position.set(-5.2, 9.0, 6.0);
    sun.castShadow = true;
    sun.shadow.mapSize.width = 2048;
    sun.shadow.mapSize.height = 2048;
    sun.shadow.camera.left = -14;
    sun.shadow.camera.right = 14;
    sun.shadow.camera.top = 14;
    sun.shadow.camera.bottom = -14;
    scene.add(sun);

    const envGroup = new THREE.Group();
    scene.add(envGroup);

    const taskGroup = new THREE.Group();
    scene.add(taskGroup);

    const droneGroup = createDrone();
    scene.add(droneGroup);

    const controlMarker = createControlMarker();
    scene.add(controlMarker);

    const trail = createTrail();
    scene.add(trail.group);

    const windVectorGroup = createWindVector3D();
    scene.add(windVectorGroup);

    const compass = createCompassRenderer();

    buildEnvironment(currentEnv);
    resetGame();
    initTask(currentTask);
    resize();
    animate();

    function buildEnvironment(envName) {
      clearGroup(envGroup);
      obstacles = [];
      animatedObjects = [];

      if (envName === "lab") {
        scene.background = new THREE.Color(0xdbeafe);
        scene.fog = new THREE.Fog(0xdbeafe, 14, 42);
        ambient.intensity = 1.55;
        sun.intensity = 2.4;
        buildLabEnvironment();
      }

      if (envName === "delivery") {
        scene.background = new THREE.Color(0xe2e8f0);
        scene.fog = new THREE.Fog(0xe2e8f0, 14, 40);
        ambient.intensity = 1.45;
        sun.intensity = 2.2;
        buildDeliveryEnvironment();
      }

      if (envName === "wind") {
        scene.background = new THREE.Color(0xc7d2fe);
        scene.fog = new THREE.Fog(0xc7d2fe, 12, 36);
        ambient.intensity = 1.25;
        sun.intensity = 2.0;
        buildWindTunnelEnvironment();
      }

      if (envName === "sky") {
        scene.background = new THREE.Color(0xbfe4ff);
        scene.fog = new THREE.Fog(0xbfe4ff, 13, 44);
        ambient.intensity = 1.75;
        sun.intensity = 2.7;
        buildSkyParkEnvironment();
      }

      if (envName === "warehouse") {
        scene.background = new THREE.Color(0xcbd5e1);
        scene.fog = new THREE.Fog(0xcbd5e1, 13, 36);
        ambient.intensity = 1.35;
        sun.intensity = 2.1;
        buildWarehouseEnvironment();
      }

      if (envName === "neon") {
        scene.background = new THREE.Color(0x020617);
        scene.fog = new THREE.Fog(0x020617, 10, 34);
        ambient.intensity = 0.85;
        sun.intensity = 1.3;
        buildNeonCityEnvironment();
      }

      addArenaBorder();
    }

    function buildLabEnvironment() {
      addFloor(0xe5e7eb, 0x94a3b8, 0x64748b);
      addMotionCaptureRig();
      addLowObstacle(-3.8, -1.8, 0.55, 0x64748b);
      addLowObstacle(2.8, 1.7, 0.5, 0x64748b);
      addLowObstacle(5.2, -2.7, 0.45, 0x64748b);
      addLabConsole(-6.6, 3.7);
      addLabConsole(6.4, -3.6);
    }

    function buildDeliveryEnvironment() {
      addFloor(0xd1d5db, 0x64748b, 0x475569);
      addRoadStrip(0, 0, 16.0, 0.35, 0x334155);
      addRoadStrip(0, 0, 0.35, 10.0, 0x334155);
      addShelf(-7.0, 0.0);
      addShelf(7.0, 0.0);
      addCrate(-5.5, -3.4, 0.8, 0);
      addCrate(-4.5, -3.2, 0.7, 1);
      addCrate(4.7, 3.4, 0.9, 2);
      addCrate(5.9, 3.0, 0.65, 3);
      addCrate(-2.2, 3.8, 0.7, 4);
      addCrate(2.3, -3.7, 0.65, 5);
      addLandingPad(-5.7, 2.6, 0x2563eb);
      addLandingPad(5.6, -2.6, 0xf97316);
    }

    function buildWindTunnelEnvironment() {
      addFloor(0xcbd5e1, 0x64748b, 0x475569);

      const tunnelMat = new THREE.MeshStandardMaterial({
        color: 0x94a3b8,
        roughness: 0.55,
        metalness: 0.08,
        transparent: true,
        opacity: 0.45,
        side: THREE.DoubleSide
      });

      const leftWall = new THREE.Mesh(new THREE.BoxGeometry(0.12, 2.8, ARENA_Z * 2.05), tunnelMat);
      leftWall.position.set(-ARENA_X - 0.1, 1.4, 0);
      envGroup.add(leftWall);

      const rightWall = leftWall.clone();
      rightWall.position.x = ARENA_X + 0.1;
      envGroup.add(rightWall);

      const roof = new THREE.Mesh(new THREE.BoxGeometry(ARENA_X * 2.1, 0.08, ARENA_Z * 2.05), tunnelMat);
      roof.position.set(0, 2.8, 0);
      envGroup.add(roof);

      for (let i = 0; i < 5; i++) {
        addFan(-7.8, -4.0 + i * 2.0, 0.0);
        addFan(7.8, -4.0 + i * 2.0, Math.PI);
      }

      addLowObstacle(-2.8, 0.0, 0.42, 0x475569);
      addLowObstacle(2.8, 0.0, 0.42, 0x475569);
    }


    function buildSkyParkEnvironment() {
      addFloor(0x86efac, 0x15803d, 0x22c55e);

      addWaterPond(-4.9, 2.6, 1.35, 0.78);
      addWaterPond(4.7, -2.8, 1.05, 0.62);

      const treePositions = [
        [-6.4, -3.9], [-5.7, 4.2], [-3.2, -4.3],
        [3.3, 4.4], [6.2, 3.5], [6.5, -3.7],
        [-1.2, 4.5], [1.4, -4.5], [-6.8, 0.8], [6.7, -0.6]
      ];

      treePositions.forEach((p, idx) => {
        addTree(p[0], p[1], 0.78 + 0.16 * (idx % 3));
      });

      addCloud(-6.2, 4.4, -3.6, 1.0, 0.010);
      addCloud(5.8, 4.9, -3.2, 0.9, -0.008);
      addCloud(1.8, 5.2, 3.7, 0.78, 0.006);

      addBalloon(-6.1, -0.9, 2.8, 0xef4444);
      addBalloon(6.1, 1.1, 3.0, 0x60a5fa);
      addBalloon(0.1, 4.2, 3.15, 0xfacc15);

      addLowObstacle(-2.6, 0.7, 0.38, 0x15803d);
      addLowObstacle(3.1, -0.9, 0.38, 0x15803d);
    }

    function buildWarehouseEnvironment() {
      addFloor(0x94a3b8, 0x334155, 0x64748b);

      addWarehouseBackWall();
      addRoadStrip(0, -4.15, 13.2, 0.52, 0x334155);
      addRoadStrip(-5.6, 0.1, 0.52, 7.6, 0x334155);
      addRoadStrip(5.6, -0.1, 0.52, 7.6, 0x334155);

      addShelf(-7.0, 0.0);
      addShelf(7.0, 0.0);

      addCrate(-5.8, -3.4, 0.75, 0);
      addCrate(-4.6, -3.2, 0.85, 1);
      addCrate(-3.0, 3.6, 0.7, 2);
      addCrate(3.8, -3.6, 0.75, 3);
      addCrate(5.8, 3.1, 0.85, 4);
      addCrate(1.3, 4.0, 0.65, 5);

      addLowObstacle(0.0, 0.0, 0.45, 0xf97316);
      addLowObstacle(2.4, 1.4, 0.36, 0xf97316);
      addLowObstacle(-2.5, -1.2, 0.36, 0xf97316);

      addSafetyCone(-1.8, -3.7);
      addSafetyCone(-1.1, -3.7);
      addSafetyCone(-0.4, -3.7);
    }

    function buildNeonCityEnvironment() {
      addFloor(0x111827, 0x38bdf8, 0x7c3aed);

      addNeonRoad(0, 0, 16.4, 0.16, 0x38bdf8);
      addNeonRoad(0, 0, 0.16, 10.4, 0xa855f7);
      addNeonRoad(-4.2, 2.7, 5.8, 0.10, 0xec4899);
      addNeonRoad(4.2, -2.7, 5.8, 0.10, 0x22c55e);

      const towers = [
        [-6.4, -4.0, 0.9, 2.35, 0x38bdf8],
        [-5.2, 3.8, 0.75, 1.8, 0xa855f7],
        [-3.4, -4.2, 0.65, 1.55, 0xf97316],
        [4.1, -4.0, 0.7, 2.0, 0x22c55e],
        [6.3, 3.7, 0.85, 2.45, 0x38bdf8],
        [3.1, 4.2, 0.6, 1.65, 0xec4899],
        [-6.6, 0.7, 0.65, 1.7, 0xfacc15],
        [6.6, -0.7, 0.65, 1.75, 0xa855f7]
      ];

      towers.forEach((p, idx) => addNeonTower(p[0], p[1], p[2], p[3], p[4], idx));

      addNeonArch(-4.7, 0.0, 0x38bdf8);
      addNeonArch(4.7, 0.0, 0xa855f7);

      addHoloBillboard(-2.7, 4.8, 0x38bdf8, "RL");
      addHoloBillboard(2.8, -4.8, 0xec4899, "Q");

      addMovingLightStrip(-6.2, 0.0, 0x38bdf8, 0.0);
      addMovingLightStrip(6.2, 0.0, 0xec4899, 1.6);

      addLowObstacle(-2.2, 2.1, 0.38, 0x38bdf8);
      addLowObstacle(2.5, -1.9, 0.38, 0xa855f7);
    }

    function addWaterPond(x, z, sx, sz) {
      const pond = new THREE.Mesh(
        new THREE.CircleGeometry(1, 64),
        new THREE.MeshStandardMaterial({
          color: 0x38bdf8,
          transparent: true,
          opacity: 0.48,
          roughness: 0.18,
          metalness: 0.05,
          side: THREE.DoubleSide
        })
      );

      pond.scale.set(sx, sz, 1);
      pond.rotation.x = -Math.PI / 2;
      pond.position.set(x, 0.025, z);
      envGroup.add(pond);

      animatedObjects.push({
        type: "softPulse",
        obj: pond,
        baseOpacity: 0.42,
        phase: Math.random() * 5
      });
    }

    function addTree(x, z, scale) {
      const trunk = new THREE.Mesh(
        new THREE.CylinderGeometry(0.08 * scale, 0.11 * scale, 0.65 * scale, 16),
        new THREE.MeshStandardMaterial({ color: 0x7c2d12, roughness: 0.75 })
      );
      trunk.position.set(x, 0.32 * scale, z);
      trunk.castShadow = true;
      envGroup.add(trunk);

      const crown1 = new THREE.Mesh(
        new THREE.ConeGeometry(0.42 * scale, 0.82 * scale, 24),
        new THREE.MeshStandardMaterial({ color: 0x16a34a, roughness: 0.8 })
      );
      crown1.position.set(x, 0.93 * scale, z);
      crown1.castShadow = true;
      envGroup.add(crown1);

      const crown2 = new THREE.Mesh(
        new THREE.ConeGeometry(0.32 * scale, 0.62 * scale, 24),
        new THREE.MeshStandardMaterial({ color: 0x22c55e, roughness: 0.8 })
      );
      crown2.position.set(x, 1.25 * scale, z);
      crown2.castShadow = true;
      envGroup.add(crown2);
    }

    function addCloud(x, y, z, scale, speed) {
      const mat = new THREE.MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.68
      });

      const group = new THREE.Group();

      for (let i = 0; i < 6; i++) {
        const puff = new THREE.Mesh(
          new THREE.SphereGeometry((0.26 + 0.08 * (i % 2)) * scale, 18, 12),
          mat
        );
        puff.position.set((i - 2.5) * 0.25 * scale, 0.06 * Math.sin(i), 0.08 * Math.cos(i));
        group.add(puff);
      }

      group.position.set(x, y, z);
      envGroup.add(group);

      animatedObjects.push({
        type: "cloud",
        group,
        speed
      });
    }

    function addBalloon(x, z, y, color) {
      const group = new THREE.Group();

      const balloon = new THREE.Mesh(
        new THREE.SphereGeometry(0.22, 24, 16),
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.25,
          metalness: 0.05
        })
      );
      balloon.position.y = y;
      group.add(balloon);

      const string = createTubeBetween(
        new THREE.Vector3(0, 0.2, 0),
        new THREE.Vector3(0, y - 0.22, 0),
        0.01,
        new THREE.MeshStandardMaterial({ color: 0x475569 })
      );
      group.add(string);

      group.position.set(x, 0, z);
      envGroup.add(group);

      animatedObjects.push({
        type: "balloon",
        group,
        baseY: 0,
        phase: Math.random() * 10
      });
    }

    function addWarehouseBackWall() {
      const wallMat = new THREE.MeshStandardMaterial({
        color: 0xe2e8f0,
        roughness: 0.72,
        metalness: 0.04
      });

      const back = new THREE.Mesh(
        new THREE.BoxGeometry(18.4, 3.2, 0.16),
        wallMat
      );
      back.position.set(0, 1.6, -5.55);
      back.receiveShadow = true;
      envGroup.add(back);

      for (let i = 0; i < 5; i++) {
        const x = -5.6 + i * 2.8;

        const lightPanel = new THREE.Mesh(
          new THREE.BoxGeometry(1.1, 0.03, 0.22),
          new THREE.MeshBasicMaterial({
            color: 0xffffff,
            transparent: true,
            opacity: 0.85
          })
        );
        lightPanel.position.set(x, 3.05, -4.8);
        envGroup.add(lightPanel);

        const pointLight = new THREE.PointLight(0xffffff, 0.45, 5.0);
        pointLight.position.set(x, 2.8, -3.6);
        envGroup.add(pointLight);
      }
    }

    function addSafetyCone(x, z) {
      const cone = new THREE.Mesh(
        new THREE.ConeGeometry(0.18, 0.55, 24),
        new THREE.MeshStandardMaterial({
          color: 0xf97316,
          roughness: 0.45
        })
      );
      cone.position.set(x, 0.275, z);
      cone.castShadow = true;
      envGroup.add(cone);
    }

    function addNeonRoad(x, z, sx, sz, color) {
      const road = new THREE.Mesh(
        new THREE.BoxGeometry(sx, 0.025, sz),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.42
        })
      );
      road.position.set(x, 0.035, z);
      envGroup.add(road);
    }

    function addNeonTower(x, z, width, height, color, idx) {
      const tower = new THREE.Mesh(
        new THREE.BoxGeometry(width, height, width),
        new THREE.MeshStandardMaterial({
          color: 0x111827,
          emissive: color,
          emissiveIntensity: 0.20,
          roughness: 0.36,
          metalness: 0.35
        })
      );
      tower.position.set(x, height / 2, z);
      tower.castShadow = true;
      tower.receiveShadow = true;
      envGroup.add(tower);

      const glow = new THREE.Mesh(
        new THREE.BoxGeometry(width * 1.05, 0.045, width * 1.05),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.72
        })
      );
      glow.position.set(x, height + 0.04, z);
      envGroup.add(glow);

      const point = new THREE.PointLight(color, 0.45, 4.5);
      point.position.set(x, height + 0.45, z);
      envGroup.add(point);

      obstacles.push({ x, z, r: width * 0.62 });
    }

    function addNeonArch(x, z, color) {
      const mat = new THREE.MeshStandardMaterial({
        color,
        emissive: color,
        emissiveIntensity: 0.85,
        roughness: 0.2
      });

      const arch = new THREE.Mesh(
        new THREE.TorusGeometry(0.75, 0.04, 14, 72, Math.PI),
        mat
      );
      arch.position.set(x, 1.1, z);
      arch.rotation.z = Math.PI;
      arch.rotation.y = Math.PI / 2;
      envGroup.add(arch);
    }

    function addHoloBillboard(x, z, color, text) {
      const canvas = document.createElement("canvas");
      canvas.width = 256;
      canvas.height = 128;

      const c = canvas.getContext("2d");
      c.clearRect(0, 0, canvas.width, canvas.height);
      c.fillStyle = "rgba(15, 23, 42, 0.42)";
      c.fillRect(0, 0, 256, 128);
      c.strokeStyle = "#" + color.toString(16).padStart(6, "0");
      c.lineWidth = 8;
      c.strokeRect(8, 8, 240, 112);
      c.fillStyle = "#ffffff";
      c.font = "bold 54px system-ui, sans-serif";
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText(text, 128, 66);

      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;

      const material = new THREE.MeshBasicMaterial({
        map: texture,
        transparent: true,
        opacity: 0.78,
        side: THREE.DoubleSide
      });

      const mesh = new THREE.Mesh(
        new THREE.PlaneGeometry(1.6, 0.8),
        material
      );

      mesh.position.set(x, 2.25, z);
      mesh.rotation.y = x < 0 ? Math.PI / 4 : -Math.PI / 4;
      envGroup.add(mesh);

      animatedObjects.push({
        type: "billboard",
        mesh,
        phase: Math.random() * 5
      });
    }

    function addMovingLightStrip(x, z, color, phase) {
      const mesh = new THREE.Mesh(
        new THREE.BoxGeometry(0.16, 0.04, 0.8),
        new THREE.MeshBasicMaterial({
          color,
          transparent: true,
          opacity: 0.82
        })
      );
      mesh.position.set(x, 0.08, z);
      envGroup.add(mesh);

      animatedObjects.push({
        type: "lightStrip",
        mesh,
        baseZ: z,
        phase
      });
    }

    function addFloor(baseColor, gridMajor, gridMinor) {
      const floor = new THREE.Mesh(
        new THREE.PlaneGeometry(ARENA_X * 2.25, ARENA_Z * 2.25),
        new THREE.MeshStandardMaterial({
          color: baseColor,
          roughness: 0.72,
          metalness: 0.04
        })
      );
      floor.rotation.x = -Math.PI / 2;
      floor.receiveShadow = true;
      envGroup.add(floor);

      const grid = new THREE.GridHelper(18, 18, gridMajor, gridMinor);
      grid.material.transparent = true;
      grid.material.opacity = 0.35;
      grid.position.y = 0.012;
      envGroup.add(grid);
    }

    function addArenaBorder() {
      const borderMat = new THREE.MeshStandardMaterial({
        color: currentEnv === "wind" ? 0x334155 : 0x1e293b,
        roughness: 0.45,
        metalness: 0.1
      });

      const railHeight = 0.16;
      const railWidth = 0.08;

      const front = new THREE.Mesh(
        new THREE.BoxGeometry(ARENA_X * 2, railHeight, railWidth),
        borderMat
      );
      front.position.set(0, railHeight / 2, ARENA_Z);
      front.castShadow = true;
      envGroup.add(front);

      const back = front.clone();
      back.position.z = -ARENA_Z;
      envGroup.add(back);

      const left = new THREE.Mesh(
        new THREE.BoxGeometry(railWidth, railHeight, ARENA_Z * 2),
        borderMat
      );
      left.position.set(-ARENA_X, railHeight / 2, 0);
      left.castShadow = true;
      envGroup.add(left);

      const right = left.clone();
      right.position.x = ARENA_X;
      envGroup.add(right);
    }

    function addMotionCaptureRig() {
      const mat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.4, metalness: 0.2 });
      const lightMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x38bdf8, emissiveIntensity: 0.8 });
      const positions = [[-7.2, -4.6], [7.2, -4.6], [-7.2, 4.6], [7.2, 4.6]];

      positions.forEach((p, idx) => {
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 2.0, 16), mat);
        pole.position.set(p[0], 1.0, p[1]);
        pole.castShadow = true;
        envGroup.add(pole);

        const marker = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 16), lightMat);
        marker.position.set(p[0], 2.05, p[1]);
        envGroup.add(marker);

        animatedObjects.push({ type: "pulse", obj: marker, phase: idx });
      });
    }

    function addLabConsole(x, z) {
      const group = new THREE.Group();

      const base = new THREE.Mesh(
        new THREE.BoxGeometry(0.8, 0.55, 0.42),
        new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.42, metalness: 0.2 })
      );
      base.position.y = 0.275;
      base.castShadow = true;
      group.add(base);

      const screen = new THREE.Mesh(
        new THREE.BoxGeometry(0.64, 0.34, 0.035),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.82 })
      );
      screen.position.set(0, 0.68, -0.23);
      group.add(screen);

      group.position.set(x, 0, z);
      group.rotation.y = x < 0 ? Math.PI / 4 : -Math.PI / 4;
      envGroup.add(group);
    }

    function addRoadStrip(x, z, sx, sz, color) {
      const road = new THREE.Mesh(
        new THREE.BoxGeometry(sx, 0.025, sz),
        new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.05 })
      );
      road.position.set(x, 0.035, z);
      envGroup.add(road);
    }

    function addShelf(x, z) {
      const group = new THREE.Group();
      const mat = new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.45, metalness: 0.18 });

      for (let i = 0; i < 4; i++) {
        const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.35, 0.08, 4.8), mat);
        shelf.position.set(0, 0.35 + i * 0.55, 0);
        shelf.castShadow = true;
        group.add(shelf);
      }

      for (let zOffset of [-2.3, 2.3]) {
        const post = new THREE.Mesh(new THREE.BoxGeometry(0.08, 2.1, 0.08), mat);
        post.position.set(0, 1.05, zOffset);
        post.castShadow = true;
        group.add(post);
      }

      group.position.set(x, 0, z);
      envGroup.add(group);
    }

    function addCrate(x, z, size, idx) {
      const crate = new THREE.Mesh(
        new THREE.BoxGeometry(size, size, size),
        new THREE.MeshStandardMaterial({
          color: idx % 2 === 0 ? 0xb45309 : 0x92400e,
          roughness: 0.65,
          metalness: 0.05
        })
      );
      crate.position.set(x, size / 2, z);
      crate.rotation.y = 0.22 * idx;
      crate.castShadow = true;
      crate.receiveShadow = true;
      envGroup.add(crate);

      obstacles.push({ x, z, r: size * 0.58 });
    }

    function addLandingPad(x, z, color) {
      const pad = createPad(x, z, color, 0.85);
      envGroup.add(pad);
    }

    function addLowObstacle(x, z, r, color) {
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(r, r, 0.75, 32),
        new THREE.MeshStandardMaterial({ color, roughness: 0.45, metalness: 0.12 })
      );
      mesh.position.set(x, 0.375, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      envGroup.add(mesh);

      obstacles.push({ x, z, r });
    }

    function addFan(x, z, rotY) {
      const group = new THREE.Group();

      const frameMat = new THREE.MeshStandardMaterial({ color: 0x1e293b, roughness: 0.35, metalness: 0.25 });
      const bladeMat = new THREE.MeshStandardMaterial({
        color: 0x38bdf8,
        roughness: 0.25,
        metalness: 0.25,
        transparent: true,
        opacity: 0.7,
        side: THREE.DoubleSide
      });

      const ring = new THREE.Mesh(new THREE.TorusGeometry(0.48, 0.035, 12, 64), frameMat);
      ring.rotation.y = Math.PI / 2;
      ring.castShadow = true;
      group.add(ring);

      const hub = new THREE.Mesh(new THREE.SphereGeometry(0.1, 16, 16), frameMat);
      group.add(hub);

      const blades = new THREE.Group();
      for (let i = 0; i < 3; i++) {
        const blade = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.02, 0.55), bladeMat);
        blade.position.z = 0.25;
        blade.rotation.y = (i * 2 * Math.PI) / 3;
        blades.add(blade);
      }

      group.add(blades);
      group.position.set(x, 1.2, z);
      group.rotation.y = rotY;
      envGroup.add(group);

      animatedObjects.push({ type: "fan", blades });
    }

    function initTask(task) {
      clearGroup(taskGroup);

      taskState = {
        successTime: 0,
        deliveries: 0,
        collected: 0,
        waypointIndex: 0,
        hasCargo: false
      };

      if (task === "hover") {
        taskTitle.textContent = "Hover in Target Zone";
        taskDesc.textContent = "Keep the drone inside the moving green target cylinder.";

        taskState.target = createRewardZone(0x22c55e, 1.05);
        taskGroup.add(taskState.target.group);
      }

      if (task === "cargo") {
        taskTitle.textContent = "Pick Up and Deliver Cargo";
        taskDesc.textContent = "Pick up the blue cargo box and deliver it to the orange landing pad.";

        taskState.pickup = { x: -5.1, z: -2.8 };
        taskState.dropoff = { x: 5.2, z: 2.7 };
        taskState.pickupPad = createPad(taskState.pickup.x, taskState.pickup.z, 0x2563eb, 0.78);
        taskState.dropoffPad = createPad(taskState.dropoff.x, taskState.dropoff.z, 0xf97316, 0.94);
        taskState.cargo = createCargoBox();
        taskState.cargo.position.set(taskState.pickup.x, 0.35, taskState.pickup.z);

        taskGroup.add(taskState.pickupPad);
        taskGroup.add(taskState.dropoffPad);
        taskGroup.add(taskState.cargo);
      }

      if (task === "waypoints") {
        taskTitle.textContent = "Visit Waypoints";
        taskDesc.textContent = "Visit the glowing waypoints in order. The active waypoint is green.";

        taskState.waypoints = [];
        taskState.waypointIndex = 0;

        const points = [[-5.6, -3.2], [-2.0, 3.4], [2.6, -3.1], [5.7, 2.8]];

        points.forEach((p, idx) => {
          const wp = createWaypoint(idx);
          wp.position.set(p[0], 1.1, p[1]);

          taskState.waypoints.push({ x: p[0], z: p[1], mesh: wp, reached: false });
          taskGroup.add(wp);
        });

        updateWaypointColors();
      }
    }

    function resetGame() {
      droneState = { x: -3.5, z: 0.3, vx: 0, vz: 0 };
      controlPoint = { x: droneState.x, z: droneState.z };

      elapsed = 0;
      totalReward = 0;
      instantReward = 0;
      progressLabel = "0.0s";
      mouseActive = false;
      policyMode = false;

      policyBtn.textContent = "Watch Demo Controller";

      for (let i = 0; i < trail.points.length; i++) {
        trail.points[i].set(droneState.x, 0.18, droneState.z);
      }
      trail.geometry.setFromPoints(trail.points);
    }

    function step(dt) {
      elapsed += dt;
      animateEnvironment();

      const wind = getWind();
      const goal = getTaskGoal();
      const action = policyMode ? getPolicyAction(wind, goal) : getHumanAction();

      const drag = 1.18;
      droneState.vx += (action.ax + wind.x - drag * droneState.vx) * dt;
      droneState.vz += (action.az + wind.z - drag * droneState.vz) * dt;

      droneState.x += droneState.vx * dt;
      droneState.z += droneState.vz * dt;

      handleArenaBounds();
      handleObstacleCollisions();

      const rewardInfo = updateTask(dt);

      instantReward = rewardInfo.instantReward;
      totalReward += rewardInfo.continuousReward * dt * 10 + rewardInfo.eventReward;

      updateDroneVisual(action, rewardInfo.closeness);
      updateControlMarker();
      updateTrail();
      updateWindVector(wind);
      updateCompass(wind);
      updateCamera();
      updateStats(wind);
    }

    function updateTask(dt) {
      let continuousReward = 0;
      let eventReward = 0;
      let closeness = 0;

      if (currentTask === "hover") {
        const tx = 3.1 * Math.sin(0.36 * elapsed);
        const tz = 2.05 * Math.sin(0.25 * elapsed + 1.2);

        taskState.target.group.position.set(tx, 0, tz);
        taskState.target.ring.scale.setScalar(1.0 + 0.04 * Math.sin(elapsed * 3.0));
        taskState.target.column.material.opacity = 0.08 + 0.035 * Math.pow(Math.sin(elapsed * 2.2), 2);

        const dist = len2(droneState.x - tx, droneState.z - tz);
        closeness = clamp(1 - dist / 4.2, 0, 1);
        const inside = dist <= 1.05 ? 1 : 0;
        continuousReward = 0.25 * closeness + 0.75 * inside;

        if (inside) taskState.successTime += dt;
        progressLabel = taskState.successTime.toFixed(1) + "s";
      }

      if (currentTask === "cargo") {
        const activeGoal = taskState.hasCargo ? taskState.dropoff : taskState.pickup;
        const dist = len2(droneState.x - activeGoal.x, droneState.z - activeGoal.z);

        closeness = clamp(1 - dist / 5.5, 0, 1);
        continuousReward = 0.18 * closeness;

        if (!taskState.hasCargo) {
          taskState.cargo.position.set(taskState.pickup.x, 0.35, taskState.pickup.z);
          taskState.cargo.rotation.y += 0.015;

          const pickupDist = len2(droneState.x - taskState.pickup.x, droneState.z - taskState.pickup.z);
          if (pickupDist < 0.72) {
            taskState.hasCargo = true;
            eventReward += 4.0;
          }
        } else {
          taskState.cargo.position.set(droneState.x, DRONE_ALTITUDE - 0.6, droneState.z);
          taskState.cargo.rotation.y += 0.05;

          const dropDist = len2(droneState.x - taskState.dropoff.x, droneState.z - taskState.dropoff.z);
          if (dropDist < 0.86) {
            taskState.hasCargo = false;
            taskState.deliveries += 1;
            eventReward += 8.0;

            const newPickup = randomArenaPoint();
            const newDrop = randomArenaPointFarFrom(newPickup);

            taskState.pickup.x = newPickup.x;
            taskState.pickup.z = newPickup.z;
            taskState.dropoff.x = newDrop.x;
            taskState.dropoff.z = newDrop.z;

            taskState.pickupPad.position.set(taskState.pickup.x, 0.03, taskState.pickup.z);
            taskState.dropoffPad.position.set(taskState.dropoff.x, 0.03, taskState.dropoff.z);
            taskState.cargo.position.set(taskState.pickup.x, 0.35, taskState.pickup.z);
          }
        }

        taskState.pickupPad.scale.setScalar(1.0 + 0.04 * Math.sin(elapsed * 3));
        taskState.dropoffPad.scale.setScalar(1.0 + 0.04 * Math.sin(elapsed * 3 + 1.2));
        progressLabel = taskState.deliveries + " delivered";
      }

      if (currentTask === "waypoints") {
        const active = taskState.waypoints[taskState.waypointIndex];

        if (active) {
          const dist = len2(droneState.x - active.x, droneState.z - active.z);
          closeness = clamp(1 - dist / 5.5, 0, 1);
          continuousReward = 0.2 * closeness;

          if (dist < 0.65) {
            active.reached = true;
            taskState.waypointIndex += 1;
            eventReward += 3.5;
            updateWaypointColors();
          }
        }

        if (taskState.waypointIndex >= taskState.waypoints.length) {
          eventReward += 7.5;

          const newPoints = [randomArenaPoint(), randomArenaPoint(), randomArenaPoint(), randomArenaPoint()];
          taskState.waypoints.forEach((wp, idx) => {
            wp.x = newPoints[idx].x;
            wp.z = newPoints[idx].z;
            wp.reached = false;
            wp.mesh.position.set(wp.x, 1.1, wp.z);
          });

          taskState.waypointIndex = 0;
          updateWaypointColors();
        }

        taskState.waypoints.forEach((wp, idx) => {
          wp.mesh.rotation.y += 0.035;
          wp.mesh.position.y = 1.1 + 0.1 * Math.sin(elapsed * 2.6 + idx);
        });

        progressLabel = taskState.waypointIndex + " / " + taskState.waypoints.length;
      }

      return {
        continuousReward,
        eventReward,
        instantReward: continuousReward + eventReward,
        closeness
      };
    }

    function getTaskGoal() {
      if (currentTask === "hover" && taskState.target) {
        return { x: taskState.target.group.position.x, z: taskState.target.group.position.z };
      }
      if (currentTask === "cargo") return taskState.hasCargo ? taskState.dropoff : taskState.pickup;
      if (currentTask === "waypoints") return taskState.waypoints[taskState.waypointIndex] || { x: 0, z: 0 };
      return { x: 0, z: 0 };
    }

    function getHumanAction() {
      if (!mouseActive) return { ax: 0, az: 0 };

      const dx = controlPoint.x - droneState.x;
      const dz = controlPoint.z - droneState.z;
      const d = Math.max(0.001, len2(dx, dz));

      const maxAccel = 6.8;
      const strength = clamp(d / 2.6, 0, 1);

      return { ax: maxAccel * strength * dx / d, az: maxAccel * strength * dz / d };
    }

    function getPolicyAction(wind, goal) {
      const dx = goal.x - droneState.x;
      const dz = goal.z - droneState.z;

      const kp = 3.6;
      const kv = 2.35;

      let ax = kp * dx - kv * droneState.vx - wind.x;
      let az = kp * dz - kv * droneState.vz - wind.z;

      const mag = len2(ax, az);
      const maxAccel = 6.8;

      if (mag > maxAccel) {
        ax = maxAccel * ax / mag;
        az = maxAccel * az / mag;
      }

      return { ax, az };
    }

    function getWind() {
      if (!windOn) return { x: 0, z: 0 };

      if (currentEnv === "lab") {
        return { x: 0.35 * Math.sin(0.7 * elapsed + 0.4), z: 0.25 * Math.cos(0.55 * elapsed + 1.2) };
      }

      if (currentEnv === "delivery") {
        return { x: 0.65 * Math.sin(0.72 * elapsed + 0.55), z: 0.45 * Math.cos(0.57 * elapsed + 1.35) };
      }

      return { x: 1.15 + 0.28 * Math.sin(1.1 * elapsed), z: 0.62 * Math.cos(0.85 * elapsed + 1.4) };
    }

    function handleArenaBounds() {
      if (droneState.x < -ARENA_X) { droneState.x = -ARENA_X; droneState.vx *= -0.35; }
      if (droneState.x > ARENA_X) { droneState.x = ARENA_X; droneState.vx *= -0.35; }
      if (droneState.z < -ARENA_Z) { droneState.z = -ARENA_Z; droneState.vz *= -0.35; }
      if (droneState.z > ARENA_Z) { droneState.z = ARENA_Z; droneState.vz *= -0.35; }
    }

    function handleObstacleCollisions() {
      obstacles.forEach((obs) => {
        const dx = droneState.x - obs.x;
        const dz = droneState.z - obs.z;
        const dist = Math.max(0.001, len2(dx, dz));
        const minDist = obs.r + 0.34;

        if (dist < minDist) {
          const nx = dx / dist;
          const nz = dz / dist;

          droneState.x = obs.x + nx * minDist;
          droneState.z = obs.z + nz * minDist;

          const dot = droneState.vx * nx + droneState.vz * nz;

          if (dot < 0) {
            droneState.vx -= 1.45 * dot * nx;
            droneState.vz -= 1.45 * dot * nz;
          }

          totalReward -= 0.03;
        }
      });
    }

    function updateDroneVisual(action, closeness) {
      const altitude = DRONE_ALTITUDE + 0.06 * Math.sin(elapsed * 5.5);
      droneGroup.position.set(droneState.x, altitude, droneState.z);

      const roll = clamp(-droneState.vx * 0.18 - action.ax * 0.025, -0.45, 0.45);
      const pitch = clamp(droneState.vz * 0.18 + action.az * 0.025, -0.45, 0.45);
      droneGroup.rotation.set(pitch, 0.22 * Math.sin(elapsed * 0.8), roll);

      const goodColor = new THREE.Color(0x22c55e);
      const baseColor = new THREE.Color(0x2563eb);
      const bodyColor = baseColor.clone().lerp(goodColor, clamp((closeness - 0.55) / 0.45, 0, 1));
      droneGroup.userData.body.material.color.copy(bodyColor);

      droneGroup.userData.rotors.forEach((rotor, idx) => {
        const spinDirection = idx % 2 === 0 ? 1 : -1;
        rotor.rotation.y += spinDirection * (1.55 + idx * 0.08);
      });

      droneGroup.userData.navLights.forEach((light, idx) => {
        const blink = 0.7 + 0.3 * Math.sin(elapsed * 8 + idx);
        light.scale.setScalar(blink);
      });
    }

    function updateControlMarker() {
      controlMarker.visible = mouseActive && !policyMode;
      controlMarker.position.set(controlPoint.x, 0.075, controlPoint.z);
      controlMarker.rotation.y += 0.04;
    }

    function updateTrail() {
      const p = trail.points;
      trail.index = (trail.index + 1) % p.length;
      p[trail.index].set(droneState.x, 0.18, droneState.z);

      const ordered = [];
      for (let i = 0; i < p.length; i++) ordered.push(p[(trail.index + i) % p.length]);

      trail.geometry.setFromPoints(ordered);
      trail.geometry.attributes.position.needsUpdate = true;
    }

    function updateWindVector(wind) {
      // In-scene wind arrow removed intentionally.
      // The bottom-right 3D compass is the only wind visualization.
      windVectorGroup.visible = false;
    }

    function updateCompass(wind) {
      const speed = len2(wind.x, wind.z);
      const normalized = clamp(speed / 1.2, 0, 1);

      if (windOn && speed > 0.01) {
        const angle = Math.atan2(wind.x, wind.z);
        compass.needleGroup.visible = true;
        compass.needleGroup.rotation.y = angle;
      } else {
        compass.needleGroup.visible = false;
      }

      const ringScale = 0.65 + 0.45 * normalized + 0.04 * Math.sin(elapsed * 4);
      compass.strengthRing.scale.set(ringScale, ringScale, ringScale);
      compass.scene.rotation.y = 0.16 * Math.sin(elapsed * 0.7);
    }

    function updateCamera() {
      const desiredX = droneState.x * 0.12;
      const desiredZ = 11.8 + droneState.z * 0.08;

      camera.position.x += (desiredX - camera.position.x) * 0.018;
      camera.position.z += (desiredZ - camera.position.z) * 0.018;
      camera.lookAt(droneState.x * 0.08, 0.35, droneState.z * 0.08);
    }

    function updateStats(wind) {
      const windSpeed = len2(wind.x, wind.z);
      modeText.textContent = policyMode ? "Demo controller" : "Human control";
      totalRewardText.textContent = totalReward.toFixed(1);
      instantRewardText.textContent = instantReward.toFixed(2);
      progressText.textContent = progressLabel;
      windText.textContent = windOn ? "On, " + windSpeed.toFixed(2) : "Off";
    }

    function animateEnvironment() {
      animatedObjects.forEach((item) => {
        if (item.type === "pulse") {
          const s = 1.0 + 0.18 * Math.sin(elapsed * 3.5 + item.phase);
          item.obj.scale.setScalar(s);
        }

        if (item.type === "softPulse") {
          item.obj.material.opacity = item.baseOpacity + 0.08 * Math.sin(elapsed * 2.2 + item.phase);
        }

        if (item.type === "fan") {
          item.blades.rotation.z += 0.45;
        }

        if (item.type === "cloud") {
          item.group.position.x += item.speed;
          if (item.group.position.x > 10) item.group.position.x = -10;
          if (item.group.position.x < -10) item.group.position.x = 10;
        }

        if (item.type === "balloon") {
          item.group.position.y = item.baseY + 0.16 * Math.sin(elapsed * 1.3 + item.phase);
          item.group.rotation.y += 0.005;
        }

        if (item.type === "billboard") {
          item.mesh.material.opacity = 0.52 + 0.30 * Math.pow(Math.sin(elapsed * 2.2 + item.phase), 2);
        }

        if (item.type === "lightStrip") {
          item.mesh.position.z = item.baseZ + 2.7 * Math.sin(elapsed * 1.4 + item.phase);
        }
      });
    }

    function onPointerMove(event) {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

      raycaster.setFromCamera(pointer, camera);

      const intersection = new THREE.Vector3();
      raycaster.ray.intersectPlane(groundPlane, intersection);

      controlPoint.x = clamp(intersection.x, -ARENA_X, ARENA_X);
      controlPoint.z = clamp(intersection.z, -ARENA_Z, ARENA_Z);

      mouseActive = true;
      policyMode = false;
      policyBtn.textContent = "Watch Demo Controller";
    }

    function createDrone() {
      const group = new THREE.Group();

      const blueMat = new THREE.MeshStandardMaterial({ color: 0x2563eb, roughness: 0.28, metalness: 0.35 });
      const lightBlueMat = new THREE.MeshStandardMaterial({ color: 0x60a5fa, roughness: 0.25, metalness: 0.25 });
      const darkMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.42, metalness: 0.32 });
      const armMat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.45, metalness: 0.35 });
      const propMat = new THREE.MeshStandardMaterial({ color: 0x111827, roughness: 0.22, metalness: 0.28 });
      const glassMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, roughness: 0.05, metalness: 0.1, transparent: true, opacity: 0.65 });
      const greenLightMat = new THREE.MeshStandardMaterial({ color: 0x22c55e, emissive: 0x22c55e, emissiveIntensity: 1.2 });
      const redLightMat = new THREE.MeshStandardMaterial({ color: 0xef4444, emissive: 0xef4444, emissiveIntensity: 1.2 });
      const whiteLightMat = new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0xffffff, emissiveIntensity: 0.8 });

      const body = new THREE.Mesh(new THREE.SphereGeometry(0.42, 40, 24), blueMat);
      body.scale.set(1.25, 0.42, 0.82);
      body.castShadow = true;
      body.receiveShadow = true;
      group.add(body);
      group.userData.body = body;

      const topShell = new THREE.Mesh(new THREE.SphereGeometry(0.32, 32, 16), lightBlueMat);
      topShell.scale.set(1.05, 0.22, 0.65);
      topShell.position.set(0, 0.16, -0.03);
      topShell.castShadow = true;
      group.add(topShell);

      const gimbal = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 16), darkMat);
      gimbal.position.set(0, -0.08, -0.45);
      gimbal.scale.set(1.0, 0.8, 0.75);
      gimbal.castShadow = true;
      group.add(gimbal);

      const lens = new THREE.Mesh(new THREE.CylinderGeometry(0.075, 0.075, 0.045, 24), glassMat);
      lens.rotation.x = Math.PI / 2;
      lens.position.set(0, -0.08, -0.56);
      group.add(lens);

      const nose = new THREE.Mesh(new THREE.ConeGeometry(0.13, 0.28, 28), lightBlueMat);
      nose.rotation.x = -Math.PI / 2;
      nose.position.set(0, 0.03, -0.55);
      nose.castShadow = true;
      group.add(nose);

      const armPositions = [
        [-0.33, 0.02, -0.28, -0.95, 0.02, -0.88],
        [0.33, 0.02, -0.28, 0.95, 0.02, -0.88],
        [-0.33, 0.02, 0.28, -0.95, 0.02, 0.88],
        [0.33, 0.02, 0.28, 0.95, 0.02, 0.88]
      ];

      armPositions.forEach((p) => {
        const arm = createTubeBetween(
          new THREE.Vector3(p[0], p[1], p[2]),
          new THREE.Vector3(p[3], p[4], p[5]),
          0.045,
          armMat
        );
        arm.castShadow = true;
        group.add(arm);
      });

      const rotorPositions = [
        [-1.05, 0.08, -0.98],
        [1.05, 0.08, -0.98],
        [-1.05, 0.08, 0.98],
        [1.05, 0.08, 0.98]
      ];

      group.userData.rotors = [];
      group.userData.navLights = [];

      rotorPositions.forEach((p, idx) => {
        const rotorAssembly = new THREE.Group();
        rotorAssembly.position.set(p[0], p[1], p[2]);

        const guard = new THREE.Mesh(new THREE.TorusGeometry(0.32, 0.025, 12, 64), armMat);
        guard.rotation.x = Math.PI / 2;
        guard.castShadow = true;
        rotorAssembly.add(guard);

        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.105, 0.105, 0.11, 32), darkMat);
        hub.castShadow = true;
        rotorAssembly.add(hub);

        const propeller = new THREE.Group();

        const blade1 = new THREE.Mesh(new THREE.BoxGeometry(0.58, 0.018, 0.075), propMat);
        blade1.position.y = 0.075;
        blade1.castShadow = true;
        propeller.add(blade1);

        const blade2 = new THREE.Mesh(new THREE.BoxGeometry(0.075, 0.018, 0.58), propMat);
        blade2.position.y = 0.077;
        blade2.castShadow = true;
        propeller.add(blade2);

        const blur = new THREE.Mesh(
          new THREE.CircleGeometry(0.29, 48),
          new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.13, side: THREE.DoubleSide })
        );
        blur.rotation.x = -Math.PI / 2;
        blur.position.y = 0.083;
        propeller.add(blur);

        rotorAssembly.add(propeller);
        group.userData.rotors.push(propeller);

        const lightMat = idx < 2 ? redLightMat : greenLightMat;
        const navLight = new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 16), lightMat);
        navLight.position.set(0, 0.03, idx < 2 ? -0.29 : 0.29);
        rotorAssembly.add(navLight);
        group.userData.navLights.push(navLight);

        group.add(rotorAssembly);
      });

      group.add(createLandingSkid(-0.38, darkMat));
      group.add(createLandingSkid(0.38, darkMat));

      const rearLight = new THREE.Mesh(new THREE.SphereGeometry(0.04, 16, 16), whiteLightMat);
      rearLight.position.set(0, 0.03, 0.49);
      group.add(rearLight);

      group.scale.setScalar(0.95);
      return group;
    }

    function createTubeBetween(start, end, radius, material) {
      const direction = new THREE.Vector3().subVectors(end, start);
      const length = direction.length();
      const geometry = new THREE.CylinderGeometry(radius, radius, length, 18);
      const mesh = new THREE.Mesh(geometry, material);
      const midpoint = new THREE.Vector3().addVectors(start, end).multiplyScalar(0.5);
      mesh.position.copy(midpoint);
      const quat = new THREE.Quaternion();
      quat.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.clone().normalize());
      mesh.quaternion.copy(quat);
      return mesh;
    }

    function createLandingSkid(xOffset, material) {
      const skid = new THREE.Group();

      const rail = createTubeBetween(
        new THREE.Vector3(xOffset, -0.32, -0.52),
        new THREE.Vector3(xOffset, -0.32, 0.52),
        0.025,
        material
      );
      rail.castShadow = true;
      skid.add(rail);

      const frontLeg = createTubeBetween(
        new THREE.Vector3(xOffset, -0.06, -0.32),
        new THREE.Vector3(xOffset, -0.32, -0.42),
        0.023,
        material
      );
      frontLeg.castShadow = true;
      skid.add(frontLeg);

      const backLeg = createTubeBetween(
        new THREE.Vector3(xOffset, -0.06, 0.32),
        new THREE.Vector3(xOffset, -0.32, 0.42),
        0.023,
        material
      );
      backLeg.castShadow = true;
      skid.add(backLeg);

      return skid;
    }

    function createRewardZone(color, radius) {
      const group = new THREE.Group();

      const disk = new THREE.Mesh(
        new THREE.CircleGeometry(radius, 96),
        new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.22, side: THREE.DoubleSide, roughness: 0.35 })
      );
      disk.rotation.x = -Math.PI / 2;
      disk.position.y = 0.025;
      group.add(disk);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.035, 16, 128),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.38, roughness: 0.25, metalness: 0.08 })
      );
      ring.rotation.x = Math.PI / 2;
      ring.position.y = 0.055;
      group.add(ring);

      const column = new THREE.Mesh(
        new THREE.CylinderGeometry(radius, radius, 2.2, 96, 1, true),
        new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.09, side: THREE.DoubleSide, roughness: 0.2 })
      );
      column.position.y = 1.1;
      group.add(column);

      return { group, ring, column };
    }

    function createPad(x, z, color, radius) {
      const zone = createRewardZone(color, radius);
      zone.group.position.set(x, 0.03, z);
      return zone.group;
    }

    function createCargoBox() {
      const group = new THREE.Group();

      const box = new THREE.Mesh(
        new THREE.BoxGeometry(0.46, 0.38, 0.46),
        new THREE.MeshStandardMaterial({ color: 0xf59e0b, roughness: 0.55, metalness: 0.08 })
      );
      box.castShadow = true;
      box.receiveShadow = true;
      group.add(box);

      const strapMat = new THREE.MeshStandardMaterial({ color: 0x78350f, roughness: 0.5, metalness: 0.02 });

      const strap1 = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.04, 0.08), strapMat);
      strap1.position.y = 0.21;
      group.add(strap1);

      const strap2 = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.04, 0.5), strapMat);
      strap2.position.y = 0.215;
      group.add(strap2);

      return group;
    }

    function createWaypoint(idx) {
      const group = new THREE.Group();

      const orb = new THREE.Mesh(
        new THREE.SphereGeometry(0.22, 32, 32),
        new THREE.MeshStandardMaterial({ color: 0x38bdf8, emissive: 0x38bdf8, emissiveIntensity: 0.8, roughness: 0.12, metalness: 0.2 })
      );
      group.add(orb);

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.38, 0.018, 12, 64),
        new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.65 })
      );
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      group.userData.orb = orb;
      group.userData.ring = ring;
      return group;
    }

    function updateWaypointColors() {
      if (!taskState.waypoints) return;

      taskState.waypoints.forEach((wp, idx) => {
        let color = 0x38bdf8;
        let opacity = 0.45;
        let emissive = 0x0ea5e9;

        if (wp.reached) {
          color = 0x64748b;
          emissive = 0x334155;
          opacity = 0.18;
        }

        if (idx === taskState.waypointIndex) {
          color = 0x22c55e;
          emissive = 0x22c55e;
          opacity = 0.85;
        }

        wp.mesh.userData.orb.material.color.set(color);
        wp.mesh.userData.orb.material.emissive.set(emissive);
        wp.mesh.userData.ring.material.color.set(color);
        wp.mesh.userData.ring.material.opacity = opacity;
      });
    }

    function createControlMarker() {
      const group = new THREE.Group();

      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.22, 0.018, 10, 48),
        new THREE.MeshBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.82, side: THREE.DoubleSide })
      );
      ring.rotation.x = Math.PI / 2;
      group.add(ring);

      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.055, 16, 16), new THREE.MeshBasicMaterial({ color: 0x2563eb }));
      dot.position.y = 0.05;
      group.add(dot);

      group.visible = false;
      return group;
    }

    function createTrail() {
      const maxPoints = 90;
      const points = new Array(maxPoints).fill(0).map(() => new THREE.Vector3());
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({ color: 0x2563eb, transparent: true, opacity: 0.35 });
      const line = new THREE.Line(geometry, material);
      const group = new THREE.Group();
      group.add(line);
      return { group, points, geometry, index: 0 };
    }

    function createWindVector3D() {
      const group = new THREE.Group();

      const mat = new THREE.MeshStandardMaterial({ color: 0xf97316, emissive: 0x7c2d12, emissiveIntensity: 0.25, roughness: 0.4, metalness: 0.15 });

      const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 1.0, 18), mat);
      shaft.rotation.z = Math.PI / 2;
      shaft.position.x = 0.38;
      group.add(shaft);

      const head = new THREE.Mesh(new THREE.ConeGeometry(0.11, 0.28, 24), mat);
      head.rotation.z = -Math.PI / 2;
      head.position.x = 0.9;
      group.add(head);

      group.position.set(-6.8, 0.7, -4.1);
      group.visible = false;
      return group;
    }

    function createCompassRenderer() {
      const compassScene = new THREE.Scene();
      const compassCamera = new THREE.PerspectiveCamera(42, 1, 0.1, 20);
      compassCamera.position.set(0, 3.4, 4.6);
      compassCamera.lookAt(0, 0, 0);

      const compassRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      compassRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      compassRenderer.outputColorSpace = THREE.SRGBColorSpace;
      compassStage.appendChild(compassRenderer.domElement);

      compassScene.add(new THREE.HemisphereLight(0xffffff, 0x334155, 2.2));

      const dir = new THREE.DirectionalLight(0xffffff, 2.2);
      dir.position.set(2, 4, 3);
      compassScene.add(dir);

      const base = new THREE.Mesh(
        new THREE.CylinderGeometry(1.2, 1.2, 0.22, 96),
        new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.32, metalness: 0.22 })
      );
      base.position.y = -0.1;
      compassScene.add(base);

      const face = new THREE.Mesh(
        new THREE.CylinderGeometry(1.06, 1.06, 0.04, 96),
        new THREE.MeshStandardMaterial({ color: 0xf8fafc, roughness: 0.25, metalness: 0.08 })
      );
      face.position.y = 0.04;
      compassScene.add(face);

      const strengthRing = new THREE.Mesh(
        new THREE.TorusGeometry(0.68, 0.025, 12, 96),
        new THREE.MeshStandardMaterial({ color: 0x93c5fd, transparent: true, opacity: 0.58, roughness: 0.2, metalness: 0.1 })
      );
      strengthRing.rotation.x = Math.PI / 2;
      strengthRing.position.y = 0.12;
      compassScene.add(strengthRing);

      const tickMat = new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.4, metalness: 0.0 });

      for (let i = 0; i < 24; i++) {
        const major = i % 6 === 0;
        const tick = new THREE.Mesh(
          new THREE.BoxGeometry(major ? 0.035 : 0.018, 0.025, major ? 0.22 : 0.13),
          tickMat
        );

        const a = (i / 24) * Math.PI * 2;
        const radius = 0.87;
        tick.position.set(radius * Math.sin(a), 0.105, radius * Math.cos(a));
        tick.rotation.y = a;
        compassScene.add(tick);
      }

      const needleGroup = new THREE.Group();

      const needleMat = new THREE.MeshStandardMaterial({ color: 0xea580c, emissive: 0x7c2d12, emissiveIntensity: 0.25, roughness: 0.25, metalness: 0.18 });

      const shaft = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.055, 0.9), needleMat);
      shaft.position.z = 0.33;
      shaft.position.y = 0.18;
      needleGroup.add(shaft);

      const head = new THREE.Mesh(new THREE.ConeGeometry(0.18, 0.36, 32), needleMat);
      head.rotation.x = Math.PI / 2;
      head.position.z = 0.9;
      head.position.y = 0.18;
      needleGroup.add(head);

      const tail = new THREE.Mesh(
        new THREE.BoxGeometry(0.11, 0.045, 0.42),
        new THREE.MeshStandardMaterial({ color: 0xfed7aa, roughness: 0.3, metalness: 0.12 })
      );
      tail.position.z = -0.28;
      tail.position.y = 0.17;
      needleGroup.add(tail);

      compassScene.add(needleGroup);

      const center = new THREE.Mesh(
        new THREE.SphereGeometry(0.12, 32, 32),
        new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.25, metalness: 0.25 })
      );
      center.position.y = 0.23;
      compassScene.add(center);

      return { scene: compassScene, camera: compassCamera, renderer: compassRenderer, needleGroup, strengthRing };
    }

    function randomArenaPoint() {
      return { x: -6.2 + Math.random() * 12.4, z: -4.0 + Math.random() * 8.0 };
    }

    function randomArenaPointFarFrom(p) {
      let q = randomArenaPoint();
      let tries = 0;
      while (len2(q.x - p.x, q.z - p.z) < 5.0 && tries < 50) {
        q = randomArenaPoint();
        tries += 1;
      }
      return q;
    }

    function clearGroup(group) {
      while (group.children.length > 0) group.remove(group.children[0]);
    }

    function clamp(v, lo, hi) {
      return Math.max(lo, Math.min(hi, v));
    }

    function len2(x, z) {
      return Math.sqrt(x * x + z * z);
    }

    function resize() {
      const width = stage.clientWidth;
      const height = stage.clientHeight;
      renderer.setSize(width, height, false);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      const cWidth = compassStage.clientWidth;
      const cHeight = compassStage.clientHeight;
      compass.renderer.setSize(cWidth, cHeight, false);
      compass.camera.aspect = cWidth / cHeight;
      compass.camera.updateProjectionMatrix();
    }

    function animate() {
      requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.033);
      step(dt);
      renderer.render(scene, camera);
      compass.renderer.render(compass.scene, compass.camera);
    }

    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerleave", () => { mouseActive = false; });

    renderer.domElement.addEventListener("touchmove", (event) => {
      event.preventDefault();
      if (event.touches.length > 0) onPointerMove(event.touches[0]);
    }, { passive: false });

    renderer.domElement.addEventListener("touchend", () => { mouseActive = false; });

    resetBtn.addEventListener("click", () => {
      resetGame();
      initTask(currentTask);
    });

    policyBtn.addEventListener("click", () => {
      policyMode = !policyMode;
      mouseActive = false;
      policyBtn.textContent = policyMode ? "Back to Human Control" : "Watch Demo Controller";
    });

    windBtn.addEventListener("click", () => {
      windOn = !windOn;
    });

    envSelect.addEventListener("change", () => {
      currentEnv = envSelect.value;
      buildEnvironment(currentEnv);
      resetGame();
      initTask(currentTask);
    });

    taskSelect.addEventListener("change", () => {
      currentTask = taskSelect.value;
      resetGame();
      initTask(currentTask);
    });

    window.addEventListener("resize", resize);
</script>

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Learning to Navigate a Living City</title>
  <style>
    :root{
      --bg:#eef2f7;
      --panel:#ffffff;
      --panel2:#f8fafc;
      --text:#0f172a;
      --muted:#475569;
      --line:#d7dee7;
      --blue:#2563eb;
      --green:#16a34a;
      --amber:#d97706;
      --red:#dc2626;
      --slate:#334155;
    }
    body{
      margin:0;
      min-height:100vh;
      background:linear-gradient(180deg,#eef2f7 0%, #e2e8f0 100%);
      font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      color:var(--text);
    }
    #widget{
      width:min(1440px, calc(100% - 26px));
      margin:14px auto;
      padding:16px;
      border-radius:24px;
      background:rgba(255,255,255,.82);
      border:1px solid rgba(148,163,184,.28);
      box-shadow:0 20px 48px rgba(15,23,42,.10);
      backdrop-filter: blur(8px);
    }
    #widget *{ box-sizing:border-box; }
    .header h2{
      margin:0 0 6px 0;
      font-size:1.8rem;
      font-weight:900;
      letter-spacing:-.02em;
    }
    .header p{
      margin:0 0 14px 0;
      color:var(--muted);
      line-height:1.55;
      max-width:1020px;
    }

    .toolbar{
      display:flex;
      flex-wrap:wrap;
      align-items:end;
      gap:10px;
      margin-bottom:12px;
    }
    .toolbar label{
      display:flex;
      flex-direction:column;
      gap:5px;
      font-size:.82rem;
      color:var(--muted);
      font-weight:800;
    }
    .toolbar select,.toolbar button{
      padding:10px 14px;
      border-radius:999px;
      font-weight:850;
      border:1px solid var(--line);
      outline:none;
      background:#fff;
      color:var(--text);
    }
    .toolbar select{ min-width:180px; }
    .toolbar button{
      cursor:pointer;
      transition:transform .12s ease, box-shadow .12s ease;
      box-shadow:0 4px 10px rgba(15,23,42,.06);
    }
    .toolbar button:hover{ transform:translateY(-1px); }
    #taskBtn{ background:#fef3c7; }
    #autoBtn{ background:#dcfce7; }
    #timeBtn{ background:#dbeafe; }
    #cameraBtn{ background:#ede9fe; }
    #resetBtn{ background:#fee2e2; }
    #episodeBtn{ background:#e0f2fe; }

    #stageWrap{
      position:relative;
      width:100%;
      height:860px;
      overflow:hidden;
      border-radius:22px;
      border:1px solid var(--line);
      background:#d8e4f1;
    }
    #stage{
      width:100%;
      height:100%;
    }
    #stage canvas.webgl{
      display:block;
      width:100%;
      height:100%;
      cursor:crosshair;
    }

    .overlay{
      position:absolute;
      z-index:5;
      background:rgba(255,255,255,.82);
      border:1px solid rgba(148,163,184,.24);
      border-radius:18px;
      box-shadow:0 10px 24px rgba(15,23,42,.08);
      backdrop-filter: blur(8px);
    }
    .pill{
      left:16px; top:14px;
      padding:10px 14px;
      font-size:.88rem;
      font-weight:850;
      color:var(--slate);
      pointer-events:none;
    }
    .taskCard{
      left:16px; bottom:16px;
      width:430px;
      padding:14px 16px;
      pointer-events:none;
    }
    .taskCard strong{
      display:block; margin-bottom:4px; font-size:1rem;
    }
    .taskCard span{
      display:block; color:var(--muted); font-size:.86rem; line-height:1.45; margin-bottom:6px;
    }
    .taskCard .small{
      font-size:.78rem; color:#1d4ed8;
    }

    .rewardCard{
      right:16px; top:16px;
      width:280px;
      padding:14px 15px;
      pointer-events:none;
    }
    .miniCard{
      right:16px; bottom:16px;
      width:250px;
      padding:12px;
      pointer-events:none;
    }
    .overlay .label{
      color:#b45309;
      font-size:.74rem;
      font-weight:950;
      letter-spacing:.08em;
      margin-bottom:8px;
    }
    .row{
      display:flex; justify-content:space-between; gap:12px; margin:6px 0;
      font-size:.86rem; color:var(--slate);
    }
    .row strong{ color:var(--text); }
    #minimap{
      width:100%; height:210px; display:block; background:#eef4fb; border-radius:12px;
    }

    .bottom{
      margin-top:14px;
      display:grid;
      grid-template-columns:1.3fr 1fr;
      gap:12px;
    }
    .panel{
      background:rgba(255,255,255,.78);
      border:1px solid var(--line);
      border-radius:18px;
      box-shadow:0 10px 24px rgba(15,23,42,.05);
      padding:12px;
    }
    .panel h3{
      margin:0 0 8px 0; font-size:1rem; font-weight:900;
    }
    #rewardGraph{
      width:100%;
      height:220px;
      display:block;
      border-radius:12px;
      background:#f8fafc;
    }
    .stats{
      display:grid;
      grid-template-columns:repeat(10,minmax(80px,1fr));
      gap:10px;
    }
    .stats div{
      background:var(--panel2);
      border:1px solid var(--line);
      border-radius:14px;
      padding:12px;
    }
    .stats span{
      display:block; font-size:.78rem; color:var(--muted); margin-bottom:4px;
    }
    .stats strong{
      display:block; font-size:1rem;
    }


    .credits{
      margin-top:14px;
      padding:12px 14px;
      border-radius:16px;
      background:rgba(255,255,255,.72);
      border:1px solid var(--line);
      color:var(--muted);
      font-size:.86rem;
      line-height:1.55;
    }
    .credits strong{ color:var(--text); }
    .credits a{
      color:#1d4ed8;
      font-weight:800;
      text-decoration:none;
    }
    .credits a:hover{ text-decoration:underline; }

    @media (max-width: 1200px){
      .bottom{ grid-template-columns:1fr; }
      .stats{ grid-template-columns:repeat(5,minmax(80px,1fr)); }
    }
    @media (max-width: 860px){
      #stageWrap{ height:640px; }
      .rewardCard, .miniCard{ display:none; }
      .taskCard{ width:calc(100% - 32px); }
      .stats{ grid-template-columns:repeat(2,minmax(80px,1fr)); }
      .toolbar select{ min-width:160px; }
    }
  </style>
</head>
<body>
<div id="widget">
  <div class="header">
    <h2>Learning to Navigate a Living City</h2>
    <p>
      An interactive reinforcement-learning playground where a small autonomous agent learns daily chore routines in a dynamic city.
      The city has traffic, pedestrians, weather, energy constraints, pickup/dropoff objectives, and a clock-driven day/night cycle.
    </p>
  </div>

  <div class="toolbar">
    <label>
      Task
      <select id="taskSelect">
        <option value="home">Go Home</option>
        <option value="grocery">Grocery Run</option>
        <option value="delivery">Pickup & Dropoff</option>
        <option value="multidelivery">Multi-Delivery</option>
        <option value="icecream">Get Ice Cream</option>
      </select>
    </label>

    <label>
      Camera
      <select id="cameraSelect">
        <option value="follow">Follow Agent</option>
        <option value="orbit">Free Orbit</option>
        <option value="top">Top Down</option>
        <option value="skyline">City View</option>
      </select>
    </label>

    <label>
      Weather
      <select id="weatherSelect">
        <option value="clear">Clear</option>
        <option value="rain">Rain</option>
        <option value="fog">Fog</option>
        <option value="snow">Snow</option>
      </select>
    </label>

    <label>
      Clock speed
      <select id="clockSpeedSelect">
        <option value="180">1 day / 3 min</option>
        <option value="300" selected>1 day / 5 min</option>
        <option value="600">1 day / 10 min</option>
        <option value="900">1 day / 15 min</option>
      </select>
    </label>

    <button id="learnModeBtn">Pause Learning</button>
    <button id="taskBtn">Try Selected Chore</button>
    <button id="autoBtn">Autonomous Mode</button>
    <button id="timeBtn">Jump Day / Night</button>
    <button id="cameraBtn">Reset Camera</button>
    <button id="episodeBtn">Start New Day</button>
    <button id="resetBtn">Reset Agent</button>
  </div>

  <div id="stageWrap">
    <div id="stage"></div>

    <div class="overlay pill">
      Click a road or sidewalk node to move · Drag to rotate · Wheel to zoom · Shift+drag to pan
    </div>

    <div class="overlay taskCard">
      <strong id="taskTitle">Daily Objective: Discover Better Chore Routines</strong>
      <span id="taskDesc">Navigate back to a cozy home while avoiding collisions and managing energy.</span>
      <span class="small" id="taskRewardInfo">Reward: chore bonuses minus time cost, energy cost, traffic-rule violations, and safety penalties.</span>
    </div>

    <div class="overlay rewardCard">
      <div class="label">CITY STATE</div>
      <div class="row"><span>City clock</span><strong id="simClockText">06:00</strong></div>
      <div class="row"><span>Day</span><strong id="dayText">1</strong></div>
      <div class="row"><span>Exploration ε</span><strong id="epsilonText">0.45</strong></div>
      <div class="row"><span>Best routine so far</span><strong id="bestChoreText">—</strong></div>
      <div class="row"><span>Last reward</span><strong id="lastRewardText">0.00</strong></div>
      <div class="row"><span>Daily return</span><strong id="returnText">0.0</strong></div>
      <div class="row"><span>Current target</span><strong id="targetText">Home A</strong></div>
      <div class="row"><span>Signal phase</span><strong id="signalText">Vertical Green</strong></div>
      <div class="row"><span>Energy</span><strong id="batteryText">100%</strong></div>
      <div class="row"><span>Rest stop</span><strong id="stationText">Available</strong></div>
      <div class="row"><span>Weather</span><strong id="weatherText">Clear</strong></div>
      <div class="row"><span>Traffic violations</span><strong id="violationText">0</strong></div>
      <div class="row"><span>Pedestrian hits</span><strong id="pedHitText">0</strong></div>
      <div class="row"><span>Collisions</span><strong id="collisionText">0</strong></div>
      <div class="row"><span>Package</span><strong id="packageText">None</strong></div>
    </div>

    <div class="overlay miniCard">
      <div class="label">MINIMAP</div>
      <canvas id="minimap" width="250" height="210"></canvas>
    </div>
  </div>

  <div class="bottom">
    <div class="panel">
      <h3>Daily Reward Trace</h3>
      <canvas id="rewardGraph" width="900" height="220"></canvas>
    </div>

    <div class="panel">
      <h3>Learning Dashboard</h3>
      <div class="stats">
        <div><span>Mode</span><strong id="modeText">Manual</strong></div>
        <div><span>Status</span><strong id="statusText">Exploring</strong></div>
        <div><span>Current chore</span><strong id="taskStatText">Go Home</strong></div>
        <div><span>Distance</span><strong id="distanceText">0.0</strong></div>
        <div><span>Timer</span><strong id="timerText">0.0s</strong></div>
        <div><span>Day</span><strong id="episodeText">1</strong></div>
        <div><span>Completed today</span><strong id="scoreText">0</strong></div>
        <div><span>Best time</span><strong id="bestTimeText">—</strong></div>
        <div><span>Deliveries</span><strong id="deliveryText">0</strong></div>
        <div><span>Neighborhood</span><strong id="neighborhoodText">Residential</strong></div>
      </div>
    </div>
  </div>

<div class="tiny-credit">
    Built with <a href="https://threejs.org/" target="_blank" rel="noopener">Three.js</a> (MIT). Procedural city assets.
</div>
</div>

<script type="module">
/*
  Credits / license:
  - Simulator and procedural city visualization code by Sreejeet Maity.
  - 3D rendering uses Three.js from jsDelivr CDN.
  - Three.js is distributed under the MIT License.
    Copyright (c) 2010-2026 three.js authors.
  - This page uses procedural geometry/textures only; no external images, models, audio, or textures.
*/
import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";

const stage = document.getElementById("stage");
const taskSelect = document.getElementById("taskSelect");
const cameraSelect = document.getElementById("cameraSelect");
const weatherSelect = document.getElementById("weatherSelect");
const clockSpeedSelect = document.getElementById("clockSpeedSelect");
const learnModeBtn = document.getElementById("learnModeBtn");
const taskBtn = document.getElementById("taskBtn");
const autoBtn = document.getElementById("autoBtn");
const timeBtn = document.getElementById("timeBtn");
const cameraBtn = document.getElementById("cameraBtn");
const episodeBtn = document.getElementById("episodeBtn");
const resetBtn = document.getElementById("resetBtn");

const taskTitle = document.getElementById("taskTitle");
const taskDesc = document.getElementById("taskDesc");
const taskRewardInfo = document.getElementById("taskRewardInfo");

const simClockText = document.getElementById("simClockText");
const dayText = document.getElementById("dayText");
const epsilonText = document.getElementById("epsilonText");
const bestChoreText = document.getElementById("bestChoreText");
const lastRewardText = document.getElementById("lastRewardText");
const returnText = document.getElementById("returnText");
const targetText = document.getElementById("targetText");
const signalText = document.getElementById("signalText");
const batteryText = document.getElementById("batteryText");
const stationText = document.getElementById("stationText");
const weatherText = document.getElementById("weatherText");
const violationText = document.getElementById("violationText");
const pedHitText = document.getElementById("pedHitText");
const collisionText = document.getElementById("collisionText");
const packageText = document.getElementById("packageText");

const modeText = document.getElementById("modeText");
const statusText = document.getElementById("statusText");
const taskStatText = document.getElementById("taskStatText");
const distanceText = document.getElementById("distanceText");
const timerText = document.getElementById("timerText");
const episodeText = document.getElementById("episodeText");
const scoreText = document.getElementById("scoreText");
const bestTimeText = document.getElementById("bestTimeText");
const deliveryText = document.getElementById("deliveryText");
const neighborhoodText = document.getElementById("neighborhoodText");

const minimapCanvas = document.getElementById("minimap");
const minimapCtx = minimapCanvas.getContext("2d");
const rewardCanvas = document.getElementById("rewardGraph");
const rewardCtx = rewardCanvas.getContext("2d");

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 500);
const renderer = new THREE.WebGLRenderer({ antialias:true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.domElement.classList.add("webgl");
stage.appendChild(renderer.domElement);

const clock = new THREE.Clock();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0,1,0), 0);

const MAP = {
  worldHalf: 62,
  riverMinZ: -6,
  riverMaxZ: 6,
  verticalRoads: [
    {id:"v1", x:-42, width:3.1, cls:"local", bridge:false},
    {id:"v2", x:-24, width:4.8, cls:"major", bridge:true},
    {id:"v3", x:-8,  width:3.1, cls:"local", bridge:false},
    {id:"v4", x:10,  width:5.4, cls:"major", bridge:true},
    {id:"v5", x:28,  width:3.1, cls:"local", bridge:false},
    {id:"v6", x:44,  width:4.8, cls:"major", bridge:true}
  ],
  horizontalRoads: [
    {id:"h1", z:-48, width:3.1, cls:"local"},
    {id:"h2", z:-32, width:5.0, cls:"major"},
    {id:"h3", z:-18, width:3.1, cls:"local"},
    {id:"h4", z:18,  width:3.1, cls:"local"},
    {id:"h5", z:34,  width:5.0, cls:"major"},
    {id:"h6", z:50,  width:3.1, cls:"local"}
  ]
};

let isNight = false;
let cameraMode = "follow";
const cameraRig = {
  theta: Math.PI/4,
  distance: 70,
  height: 40,
  targetX: 0,
  targetZ: 0,
  dragging: false,
  dragMoved: false,
  lastX: 0,
  lastY: 0
};

const hemi = new THREE.HemisphereLight(0xffffff, 0xb8c8d8, 1.5);
scene.add(hemi);

const sun = new THREE.DirectionalLight(0xfff6d5, 1.25);
sun.position.set(-18, 62, 18);
sun.castShadow = true;
sun.shadow.mapSize.set(2048,2048);
sun.shadow.camera.left = -120;
sun.shadow.camera.right = 120;
sun.shadow.camera.top = 120;
sun.shadow.camera.bottom = -120;
scene.add(sun);

const cityGroup = new THREE.Group();
const skylineGroup = new THREE.Group();
const dynamicGroup = new THREE.Group();
const pathGroup = new THREE.Group();
const weatherGroup = new THREE.Group();
scene.add(cityGroup, skylineGroup, dynamicGroup, pathGroup, weatherGroup);

const clickMarker = createClickMarker();
scene.add(clickMarker);

const world = {
  obstacles: [],
  textures: [],
  graphNodes: [],
  graphEdges: new Map(),
  driveSegments: [],
  trafficSignals: [],
  cars: [],
  pedestrians: [],
  crossings: [],
  destinations: {},
  beacons: [],
  weatherParticles: null,
  weatherVelocities: [],
  lightableMaterials: [],
  streetLamps: [],
  vehicleLights: []
};

const agentColors = [0x2563eb, 0x16a34a, 0xd97706];
const agents = [];
let mainAgent = null;

let manualTarget = null;
let autoMode = true;
let learningMode = true;
let currentTask = null;
let currentTarget = null;

let score = 0;
let dailyChoresCompleted = 0;
let bestTime = null;
let deliveriesCompleted = 0;
let collisions = 0;
let trafficViolations = 0;
let pedestrianHits = 0;
let episode = 1;

let simulatedDay = 1;
let simulatedMinutes = 6 * 60;
let dayLengthSeconds = 300;
let lastClockNightState = null;

const choreKeys = ["home", "grocery", "icecream", "delivery", "multidelivery", "charge"];
const qValues = {
  home: 0,
  grocery: 0,
  icecream: 0,
  delivery: 0,
  multidelivery: 0,
  charge: 0
};
const qCounts = {
  home: 0,
  grocery: 0,
  icecream: 0,
  delivery: 0,
  multidelivery: 0,
  charge: 0
};
let epsilon = 0.45;
let currentChoreKey = "home";
let taskStartReturn = 0;
let taskTimer = 0;
let cumulativeReward = 0;
let lastReward = 0;
let rewardHistory = [];
let returnHistory = [];
let rewardTickAcc = 0;
let collisionCooldown = 0;
let trafficViolationCooldown = 0;
let pedestrianCollisionCooldown = 0;
let weatherMode = "clear";

init();

function emitSimEvent(name, detail = {}){
  const payload = {
    name,
    episode,
    task: currentTask?.kind || null,
    reward: lastReward,
    cumulativeReturn: cumulativeReward,
    energy: mainAgent?.energy ?? null,
    time: taskTimer,
    ...detail
  };

  window.dispatchEvent(new CustomEvent("city-sim-event", { detail: payload }));

  /*
    Sound-ready hook:
    window.addEventListener("city-sim-event", (event) => {
      if (event.detail.name === "collision") playCollisionSound();
      if (event.detail.name === "goal") playGoalSound();
      if (event.detail.name === "pickup") playPickupSound();
      if (event.detail.name === "traffic-violation") playWarningSound();
      if (event.detail.name === "weather-change") playWeatherAmbience(event.detail.weather);
    });
  */
}

function init(){
  applyEnvironmentTheme();
  buildWorld();
  createAgents();
  resetDailyExperience(false);
  resize();
  animate();
}


function registerNightMaterial(material, nightIntensity, dayIntensity = 0, profile = "default"){
  world.lightableMaterials.push({ material, nightIntensity, dayIntensity, profile });
  if(typeof material.emissiveIntensity === "number"){
    material.emissiveIntensity = currentNightIntensity(nightIntensity, dayIntensity, profile);
  }
}

function currentNightIntensity(nightIntensity, dayIntensity = 0, profile = "default"){
  if(!isNight) return dayIntensity;

  const minutes = typeof simulatedMinutes === "number" ? simulatedMinutes : 20 * 60;
  const hour = Math.floor(minutes / 60);

  // Office towers are brightest in the early evening and dimmer late night.
  if(profile === "office"){
    if(hour >= 19 && hour < 22) return nightIntensity * 1.18;
    if(hour >= 22 || hour < 5) return nightIntensity * 0.42;
    return nightIntensity * 0.72;
  }

  // Residential buildings stay warmly lit in the evening, but many windows go off later.
  if(profile === "residential"){
    if(hour >= 19 && hour < 23) return nightIntensity * 1.00;
    if(hour >= 23 || hour < 5) return nightIntensity * 0.58;
    return nightIntensity * 0.72;
  }

  // Shops glow in the evening, but become softer after closing hours.
  if(profile === "shop"){
    if(hour >= 18 && hour < 22) return nightIntensity * 1.10;
    if(hour >= 22 || hour < 6) return nightIntensity * 0.34;
    return nightIntensity * 0.75;
  }

  return nightIntensity;
}

function updateCityNightLights(){
  if(!world) return;

  for(const item of world.lightableMaterials){
    if(!item || !item.material) continue;
    if(typeof item.material.emissiveIntensity === "number"){
      item.material.emissiveIntensity = currentNightIntensity(item.nightIntensity, item.dayIntensity, item.profile);
    }
    if(item.material.transparent && item.material.userData?.nightOpacity !== undefined){
      item.material.opacity = isNight ? item.material.userData.nightOpacity : item.material.userData.dayOpacity;
    }
  }

  for(const lamp of world.streetLamps){
    if(!lamp) continue;
    if(lamp.bulbMaterial){
      lamp.bulbMaterial.emissiveIntensity = isNight ? 1.6 : 0.0;
      lamp.bulbMaterial.opacity = isNight ? 0.95 : 0.35;
      lamp.bulbMaterial.color.set(isNight ? 0xfff1bf : 0x888888);
    }
    if(lamp.glowMaterial){
      lamp.glowMaterial.opacity = isNight ? 0.22 : 0.0;
    }
    if(lamp.pointLight){
      lamp.pointLight.intensity = isNight ? lamp.baseIntensity : 0.0;
    }
  }

  for(const lights of world.vehicleLights){
    if(!lights) continue;
    for(const m of lights.headMaterials || []){
      m.emissiveIntensity = isNight ? 1.8 : 0.0;
    }
    for(const m of lights.tailMaterials || []){
      m.emissiveIntensity = isNight ? 1.1 : 0.0;
    }
    for(const l of lights.pointLights || []){
      l.intensity = isNight ? l.userData.baseIntensity : 0.0;
    }
  }
}

function createStreetLamp(x, z, height = 3.6){
  const group = new THREE.Group();

  const pole = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.07, height, 12),
    new THREE.MeshStandardMaterial({ color:0x64748b, roughness:0.52, metalness:0.22 })
  );
  pole.position.y = height / 2;
  group.add(pole);

  const arm = new THREE.Mesh(
    new THREE.BoxGeometry(0.48, 0.05, 0.05),
    new THREE.MeshStandardMaterial({ color:0x64748b, roughness:0.52, metalness:0.22 })
  );
  arm.position.set(0.18, height - 0.18, 0);
  group.add(arm);

  const bulbMat = new THREE.MeshStandardMaterial({
    color:0xfff1bf,
    emissive:0xffe08a,
    emissiveIntensity:isNight ? 1.6 : 0.0,
    roughness:0.18,
    metalness:0.0,
    transparent:true,
    opacity:isNight ? 0.95 : 0.35
  });
  const bulb = new THREE.Mesh(new THREE.SphereGeometry(0.11, 16, 16), bulbMat);
  bulb.position.set(0.39, height - 0.22, 0);
  group.add(bulb);

  const glowMat = new THREE.MeshBasicMaterial({
    color:0xffefad,
    transparent:true,
    opacity:isNight ? 0.22 : 0.0,
    depthWrite:false
  });
  const glow = new THREE.Mesh(new THREE.SphereGeometry(0.34, 16, 16), glowMat);
  glow.position.copy(bulb.position);
  group.add(glow);

  const light = new THREE.PointLight(0xffe8a3, isNight ? 1.6 : 0.0, 9, 2.0);
  light.position.copy(bulb.position);
  light.userData.baseIntensity = 1.6;
  group.add(light);

  group.position.set(x, 0, z);
  cityGroup.add(group);
  world.streetLamps.push({ bulbMaterial: bulbMat, glowMaterial: glowMat, pointLight: light, baseIntensity: 1.6 });
}

function addStreetLights(){
  // Along major vertical roads
  const zList = [-50, -34, -18, 18, 34, 50];
  for(const road of MAP.verticalRoads){
    if(road.cls !== "major") continue;
    for(const z of zList){
      if(!road.bridge && z > MAP.riverMinZ - 3 && z < MAP.riverMaxZ + 3) continue;
      createStreetLamp(road.x - road.width/2 - 1.35, z);
      createStreetLamp(road.x + road.width/2 + 1.35, z);
    }
  }

  // Along major horizontal roads
  const xList = [-50, -34, -18, -2, 16, 32, 48];
  for(const road of MAP.horizontalRoads){
    if(road.cls !== "major") continue;
    for(const x of xList){
      createStreetLamp(x, road.z - road.width/2 - 1.35);
      createStreetLamp(x, road.z + road.width/2 + 1.35);
    }
  }

  // Around landmarks / neighborhoods
  const landmarkLamps = [
    [36, -22], [50, -10], [32, -38], [-22, -26], [-26, 34], [34, 38], [-40, 20], [6, -38],
    [14, 14], [26, 14], [14, 30], [26, 30]
  ];
  for(const [x, z] of landmarkLamps){
    createStreetLamp(x, z, 3.9);
  }
}

function applyEnvironmentTheme(){
  if(isNight){
    scene.background = new THREE.Color(0x152130);
    scene.fog = new THREE.Fog(0x1c2b3d, 28, 155);
    hemi.intensity = 0.75;
    sun.intensity = 0.55;
    sun.color.set(0xdde7ff);
    sun.position.set(38, 46, 22);
  } else {
    scene.background = new THREE.Color(0xd8e4f1);
    scene.fog = new THREE.Fog(0xe6eef6, 48, 185);
    hemi.intensity = 1.55;
    sun.intensity = 1.25;
    sun.color.set(0xfff6d5);
    sun.position.set(-18, 62, 18);
  }
  applyWeatherVisuals();
  updateCityNightLights();
}

function clearGroup(group){
  while(group.children.length) group.remove(group.children[0]);
}

function buildWorld(){
  clearGroup(cityGroup);
  clearGroup(skylineGroup);
  clearGroup(dynamicGroup);
  clearGroup(pathGroup);
  clearGroup(weatherGroup);

  world.obstacles = [];
  world.textures = [];
  world.graphNodes = [];
  world.graphEdges = new Map();
  world.driveSegments = [];
  world.trafficSignals = [];
  world.cars = [];
  world.pedestrians = [];
  world.crossings = [];
  world.destinations = {};

  addSkyline();
  addGroundRiverRoads();
  addCentralPark();
  addBlocks();
  addLandmarks();
  addStreetLights();
  addTrafficSignalsAndCrossings();
  buildGraph();
  addVehicles();
  addPedestrians();
  buildWeatherSystem();
  updateCityNightLights();
  addPathNodesPreview(false);
}

function addSkyline(){
  const colors = isNight ? [0x3b4654,0x465466,0x546275] : [0xa5b2c0,0xb7c3cf,0xc7d2dd];
  for(let i=0;i<86;i++){
    const w = rand(3, 10);
    const d = rand(3, 10);
    const h = rand(10, 46);
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(w,h,d),
      new THREE.MeshStandardMaterial({
        color: colors[Math.floor(Math.random()*colors.length)],
        emissive: isNight ? 0xffd166 : 0xffffff,
        emissiveIntensity: isNight ? 0.05 : 0.0,
        roughness: 0.88,
        metalness: 0.04
      })
    );
    const r = rand(76, 108);
    const a = Math.random()*Math.PI*2;
    mesh.position.set(Math.cos(a)*r, h/2, Math.sin(a)*r);
    skylineGroup.add(mesh);
  }
}

function addGroundRiverRoads(){
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(MAP.worldHalf*2.6, MAP.worldHalf*2.6),
    new THREE.MeshStandardMaterial({ color:isNight?0x45576a:0xaabccc, roughness:0.95 })
  );
  ground.rotation.x = -Math.PI/2;
  ground.receiveShadow = true;
  cityGroup.add(ground);

  const river = new THREE.Mesh(
    new THREE.PlaneGeometry(MAP.worldHalf*2.4, MAP.riverMaxZ-MAP.riverMinZ),
    new THREE.MeshStandardMaterial({ color:0x6baed6, transparent:true, opacity:isNight?.88:.84, roughness:.3 })
  );
  river.rotation.x = -Math.PI/2;
  river.position.set(0,0.04,0);
  cityGroup.add(river);

  for(let i=0;i<14;i++){
    const wave = new THREE.Mesh(
      new THREE.TorusGeometry(rand(1.2,2.4),0.03,8,28,Math.PI*1.5),
      new THREE.MeshBasicMaterial({ color:0xffffff, transparent:true, opacity:isNight?0.16:0.10 })
    );
    wave.rotation.x = Math.PI/2;
    wave.position.set(rand(-58,58), 0.055, rand(MAP.riverMinZ+.6, MAP.riverMaxZ-.6));
    cityGroup.add(wave);
  }

  for(const road of MAP.verticalRoads){
    addVerticalRoad(road);
  }
  for(const road of MAP.horizontalRoads){
    addHorizontalRoad(road);
  }

  addCrosswalkMarkings();
}

function addVerticalRoad(road){
  const roadColor = isNight ? 0x344150 : 0x697887;
  const sideColor = isNight ? 0x728191 : 0xc7d0d8;
  const curbColor = isNight ? 0x243447 : 0x99a8b5;

  const roadMat = new THREE.MeshStandardMaterial({ color: road.cls==="major" ? darken(roadColor, .08) : roadColor, roughness:.86 });
  const sidewalkMat = new THREE.MeshStandardMaterial({ color: sideColor, roughness:.84 });
  const curbMat = new THREE.MeshStandardMaterial({ color: curbColor, roughness:.78 });

  const totalLen = MAP.worldHalf*2.05;
  const sideWidth = road.width + 1.7;

  if(road.bridge){
    addStrip(road.x, 0, sideWidth, totalLen, sidewalkMat, .026);
    addStrip(road.x, 0, road.width, totalLen, roadMat, .045);
    addBridge(road.x, road.width, MAP.riverMaxZ - MAP.riverMinZ + 2.0);
    world.driveSegments.push({orientation:"v", roadId:road.id, x:road.x, min:-MAP.worldHalf, max:MAP.worldHalf, width:road.width, bridge:true});
    addVerticalMarkings(road.x, road.width, -MAP.worldHalf, MAP.worldHalf, road.cls);
  } else {
    const northLen = MAP.worldHalf - MAP.riverMaxZ;
    const southLen = MAP.worldHalf + MAP.riverMinZ;
    addStrip(road.x, (MAP.worldHalf+MAP.riverMaxZ)/2, sideWidth, northLen, sidewalkMat, .026);
    addStrip(road.x, (MAP.worldHalf+MAP.riverMaxZ)/2, road.width, northLen, roadMat, .045);
    addStrip(road.x, (-MAP.worldHalf+MAP.riverMinZ)/2, sideWidth, southLen, sidewalkMat, .026);
    addStrip(road.x, (-MAP.worldHalf+MAP.riverMinZ)/2, road.width, southLen, roadMat, .045);
    world.driveSegments.push({orientation:"v", roadId:road.id, x:road.x, min:MAP.riverMaxZ, max:MAP.worldHalf, width:road.width, bridge:false});
    world.driveSegments.push({orientation:"v", roadId:road.id, x:road.x, min:-MAP.worldHalf, max:MAP.riverMinZ, width:road.width, bridge:false});
    addVerticalMarkings(road.x, road.width, MAP.riverMaxZ, MAP.worldHalf, road.cls);
    addVerticalMarkings(road.x, road.width, -MAP.worldHalf, MAP.riverMinZ, road.cls);
  }

  addStrip(road.x-road.width/2, 0, .08, totalLen, curbMat, .08);
  addStrip(road.x+road.width/2, 0, .08, totalLen, curbMat, .08);
}

function addHorizontalRoad(road){
  const roadColor = isNight ? 0x344150 : 0x697887;
  const sideColor = isNight ? 0x728191 : 0xc7d0d8;
  const curbColor = isNight ? 0x243447 : 0x99a8b5;
  const roadMat = new THREE.MeshStandardMaterial({ color: road.cls==="major" ? darken(roadColor, .08) : roadColor, roughness:.86 });
  const sidewalkMat = new THREE.MeshStandardMaterial({ color: sideColor, roughness:.84 });
  const curbMat = new THREE.MeshStandardMaterial({ color: curbColor, roughness:.78 });

  const totalLen = MAP.worldHalf*2.05;
  const sideWidth = road.width + 1.7;
  addStrip(0, road.z, totalLen, sideWidth, sidewalkMat, .026);
  addStrip(0, road.z, totalLen, road.width, roadMat, .045);
  world.driveSegments.push({orientation:"h", roadId:road.id, z:road.z, min:-MAP.worldHalf, max:MAP.worldHalf, width:road.width});
  addHorizontalMarkings(road.z, road.width, -MAP.worldHalf, MAP.worldHalf, road.cls);

  addStrip(0, road.z-road.width/2, totalLen, .08, curbMat, .08);
  addStrip(0, road.z+road.width/2, totalLen, .08, curbMat, .08);
}

function addStrip(x,z,sx,sz,mat,y){
  const m = new THREE.Mesh(new THREE.BoxGeometry(sx, .035, sz), mat);
  m.position.set(x,y,z);
  m.receiveShadow = true;
  cityGroup.add(m);
}

function addBridge(x,width,length){
  const deck = new THREE.Mesh(
    new THREE.BoxGeometry(width+.2,.22,length),
    new THREE.MeshStandardMaterial({ color:isNight?0x8793a0:0xaeb7bf, roughness:.75, metalness:.12 })
  );
  deck.position.set(x,.12,0);
  cityGroup.add(deck);

  const railMat = new THREE.MeshStandardMaterial({ color:isNight?0xc4d0db:0x8996a2, roughness:.55 });
  const r1 = new THREE.Mesh(new THREE.BoxGeometry(.08,.18,length), railMat);
  const r2 = r1.clone();
  r1.position.set(x-width/2-.03,.23,0);
  r2.position.set(x+width/2+.03,.23,0);
  cityGroup.add(r1,r2);
}

function addVerticalMarkings(x,width,zMin,zMax,cls){
  const yellow = new THREE.MeshBasicMaterial({ color:0xfacc15, transparent:true, opacity:.96 });
  const white = new THREE.MeshBasicMaterial({ color:0xf8fafc, transparent:true, opacity:.92 });
  if(cls === "major"){
    const c1 = new THREE.Mesh(new THREE.BoxGeometry(.06,.052,zMax-zMin), yellow);
    const c2 = new THREE.Mesh(new THREE.BoxGeometry(.06,.052,zMax-zMin), yellow);
    c1.position.set(x-.10,.082,(zMin+zMax)/2);
    c2.position.set(x+.10,.082,(zMin+zMax)/2);
    cityGroup.add(c1,c2);
    addDashedLine("v", x-width*.22, zMin, zMax, white);
    addDashedLine("v", x+width*.22, zMin, zMax, white);
  } else {
    addDashedLine("v", x, zMin, zMax, yellow);
  }
}
function addHorizontalMarkings(z,width,xMin,xMax,cls){
  const yellow = new THREE.MeshBasicMaterial({ color:0xfacc15, transparent:true, opacity:.96 });
  const white = new THREE.MeshBasicMaterial({ color:0xf8fafc, transparent:true, opacity:.92 });
  if(cls === "major"){
    const c1 = new THREE.Mesh(new THREE.BoxGeometry(xMax-xMin,.052,.06), yellow);
    const c2 = new THREE.Mesh(new THREE.BoxGeometry(xMax-xMin,.052,.06), yellow);
    c1.position.set((xMin+xMax)/2,.082,z-.10);
    c2.position.set((xMin+xMax)/2,.082,z+.10);
    cityGroup.add(c1,c2);
    addDashedLine("h", z-width*.22, xMin, xMax, white);
    addDashedLine("h", z+width*.22, xMin, xMax, white);
  } else {
    addDashedLine("h", z, xMin, xMax, yellow);
  }
}
function addDashedLine(orientation, fixed, min, max, mat){
  let p = min;
  while(p < max){
    const len = Math.min(1.2, max-p);
    const m = orientation === "v"
      ? new THREE.Mesh(new THREE.BoxGeometry(.06,.052,len), mat)
      : new THREE.Mesh(new THREE.BoxGeometry(len,.052,.06), mat);
    if(orientation === "v") m.position.set(fixed,.082,p+len/2);
    else m.position.set(p+len/2,.082,fixed);
    cityGroup.add(m);
    p += 2.1;
  }
}
function addCrosswalkMarkings(){
  const mat = new THREE.MeshBasicMaterial({ color:0xffffff, transparent:true, opacity:.70 });
  for(const vr of MAP.verticalRoads){
    for(const hr of MAP.horizontalRoads){
      if(vr.cls !== "major" && hr.cls !== "major") continue;
      const x = vr.x, z = hr.z;
      for(let k=-3;k<=3;k++){
        const a = new THREE.Mesh(new THREE.BoxGeometry(.18,.054,Math.min(vr.width,2.7)), mat);
        a.position.set(x+k*.34,.086,z+hr.width*.5);
        cityGroup.add(a);
        const b = new THREE.Mesh(new THREE.BoxGeometry(Math.min(hr.width,2.7),.054,.18), mat);
        b.position.set(x+vr.width*.5,.086,z+k*.34);
        cityGroup.add(b);
      }
    }
  }
}

function addCentralPark(){
  const park = new THREE.Mesh(
    new THREE.BoxGeometry(26,.03,18),
    new THREE.MeshStandardMaterial({ color:isNight?0x3d7257:0x5fa073, roughness:.92 })
  );
  park.position.set(20,.07,22);
  cityGroup.add(park);

  const pond = new THREE.Mesh(
    new THREE.CylinderGeometry(3.3,3.9,.05,44),
    new THREE.MeshBasicMaterial({ color:0x7ec8e3, transparent:true, opacity:.74 })
  );
  pond.position.set(16,.10,24);
  cityGroup.add(pond);

  const pathColor = 0xd0c1a1;
  const p1 = new THREE.Mesh(new THREE.BoxGeometry(22,.035,.30), new THREE.MeshStandardMaterial({ color:pathColor, roughness:.95 }));
  p1.position.set(20,.09,22); cityGroup.add(p1);
  const p2 = new THREE.Mesh(new THREE.BoxGeometry(.30,.035,15), new THREE.MeshStandardMaterial({ color:pathColor, roughness:.95 }));
  p2.position.set(20,.09,22); cityGroup.add(p2);

  for(let i=0;i<18;i++) addTree(8+rand(0,24), 14+rand(0,14), rand(.52,.85));
  addBench(25,18,.2); addBench(11,28,-.4); addBench(21,29,.8);
}

function addBlocks(){
  const vBounds = [-MAP.worldHalf, ...MAP.verticalRoads.map(r=>r.x), MAP.worldHalf];
  const hBounds = [-MAP.worldHalf, ...MAP.horizontalRoads.map(r=>r.z), MAP.worldHalf];

  for(let i=0;i<vBounds.length-1;i++){
    for(let j=0;j<hBounds.length-1;j++){
      const left = vBounds[i], right = vBounds[i+1], bottom = hBounds[j], top = hBounds[j+1];
      const cx = (left+right)/2, cz = (bottom+top)/2;
      const width = right-left, depth = top-bottom;
      if(width < 6 || depth < 6) continue;
      if(cz > MAP.riverMinZ-3 && cz < MAP.riverMaxZ+3) continue;
      if(cx > 7 && cx < 33 && cz > 13 && cz < 31) continue;
      if(isReserved(cx,cz)) continue;

      if(Math.random() < .16){
        addPocketPark(cx, cz, Math.min(width-1.8, 8.4), Math.min(depth-1.8, 7.6));
      } else {
        addBuildingBlock(cx, cz, width-1.6, depth-1.6, neighborhoodOf(cx,cz));
      }
    }
  }
}
function isReserved(cx,cz){
  const specials = [
    [36,-26,10,10],[50,-14,10,10],[32,-42,10,10],
    [-22,-30,10,10],[-26,30,10,10],[34,34,10,10],[-40,16,10,10],[6,-42,8,8]
  ];
  return specials.some(([x,z,w,d]) => Math.abs(cx-x)<w/2 && Math.abs(cz-z)<d/2);
}
function neighborhoodOf(x,z){
  if(z > MAP.riverMaxZ+2 && x > -4) return "downtown";
  if(x < -12 && z > MAP.riverMaxZ+2) return "riverside";
  if(z < -8 && x > 18) return "residential";
  if(z < -8 && x < 8) return "market";
  return "mixed";
}
function addPocketPark(cx,cz,w,d){
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(w,.03,d),
    new THREE.MeshStandardMaterial({ color:isNight?0x48775f:0x73ae82, roughness:.92 })
  );
  base.position.set(cx,.07,cz); cityGroup.add(base);
  const path = new THREE.Mesh(
    new THREE.BoxGeometry(w*.65,.035,.28),
    new THREE.MeshStandardMaterial({ color:0xd0c1a1, roughness:.95 })
  );
  path.position.set(cx,.09,cz); cityGroup.add(path);
  for(let i=0;i<4+Math.floor(Math.random()*3);i++){
    addTree(cx+rand(-w*.28,w*.28), cz+rand(-d*.28,d*.28), rand(.45,.72));
  }
  addFlowerPatch(cx-w*.10, cz+d*.12);
}
function addBuildingBlock(cx,cz,w,d,neighborhood){
  const groundColor = neighborhood === "residential" ? (isNight?0x527f62:0x8cb78e) : (isNight?0xaab2ba:0xd8dde3);
  const ground = new THREE.Mesh(
    new THREE.BoxGeometry(w,.03,d),
    new THREE.MeshStandardMaterial({ color:groundColor, roughness:.90 })
  );
  ground.position.set(cx,.07,cz);
  cityGroup.add(ground);

  const n = neighborhood === "downtown" ? 2 + Math.floor(Math.random()*3) : 1 + Math.floor(Math.random()*3);
  for(let k=0;k<n;k++){
    const bw = rand(Math.min(2.2,w*.22), Math.min(6.6,w*.42));
    const bd = rand(Math.min(2.2,d*.22), Math.min(6.6,d*.42));
    const bx = cx + rand(-w*.28,w*.28);
    const bz = cz + rand(-d*.28,d*.28);
    let hMin = 6, hMax = 18;
    if(neighborhood === "downtown"){ hMin=16; hMax=34; }
    if(neighborhood === "riverside"){ hMin=10; hMax=22; }
    if(neighborhood === "residential"){ hMin=5; hMax=13; }
    if(neighborhood === "market"){ hMin=7; hMax=17; }
    const bh = rand(hMin,hMax);
    addDetailedBuilding(bx,bz,bw,bd,bh,neighborhood);
  }
  if(Math.random() < .42) addTree(cx+rand(-w*.18,w*.18), cz+rand(-d*.18,d*.18), rand(.50,.72));
}
function addDetailedBuilding(x,z,w,d,h,neighborhood){
  const colors = [0xd8dfe6,0xcfd8e1,0xe4e9ef,0xc5d0db,0xd6dde4,0xced6df];
  const officeLike = neighborhood === "downtown" && h > 20;
  const texSet = createBuildingTextureSet(neighborhood, officeLike);

  const mat = new THREE.MeshStandardMaterial({
    color: colors[Math.floor(Math.random()*colors.length)],
    map: texSet.facadeMap,
    emissiveMap: texSet.windowLightMap,
    emissive: 0xffd166,
    emissiveIntensity: isNight ? texSet.nightIntensity : 0.0,
    roughness:.78,
    metalness: neighborhood==="downtown" ? .14 : .06
  });
  registerNightMaterial(mat, texSet.nightIntensity, 0.0, texSet.profile);

  const mesh = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), mat);
  mesh.position.set(x,h/2,z);
  mesh.castShadow = true;
  cityGroup.add(mesh);

  const roof = new THREE.Mesh(
    new THREE.BoxGeometry(w*.96,.18,d*.96),
    new THREE.MeshStandardMaterial({ color:0x96a3b0, roughness:.80 })
  );
  roof.position.set(x,h+.09,z);
  cityGroup.add(roof);

  if(Math.random() < .30){
    const unit = new THREE.Mesh(
      new THREE.BoxGeometry(w*rand(.15,.28), .3, d*rand(.15,.28)),
      new THREE.MeshStandardMaterial({ color:0xaab4bf, roughness:.65 })
    );
    unit.position.set(x+rand(-w*.2,w*.2), h+.32, z+rand(-d*.2,d*.2));
    cityGroup.add(unit);
  }

  world.obstacles.push({minX:x-w/2-.18, maxX:x+w/2+.18, minZ:z-d/2-.18, maxZ:z+d/2+.18});
}
function createBuildingTextureSet(neighborhood = "mixed", officeLike = false){
  const facadeCanvas = document.createElement("canvas");
  facadeCanvas.width = 256;
  facadeCanvas.height = 512;
  const fctx = facadeCanvas.getContext("2d");

  const lightCanvas = document.createElement("canvas");
  lightCanvas.width = 256;
  lightCanvas.height = 512;
  const lctx = lightCanvas.getContext("2d");

  const facadePalettes = {
    downtown: ["#d0d7df", "#c7d1db", "#d9e0e7", "#bfcad6"],
    riverside: ["#d7e0e4", "#cbd9de", "#dce5e8", "#c5d2d8"],
    residential: ["#e4ded4", "#ded8cc", "#e8e1d4", "#d9d3c8"],
    market: ["#d8dde2", "#ced6dd", "#e1e5e8", "#c7d0d7"],
    mixed: ["#d6dee6", "#cfd8e1", "#dce4eb", "#c7d1db"]
  };

  const palette = facadePalettes[neighborhood] || facadePalettes.mixed;
  fctx.fillStyle = palette[Math.floor(Math.random()*palette.length)];
  fctx.fillRect(0,0,256,512);

  // Subtle facade paneling independent of time-of-day.
  fctx.strokeStyle = "rgba(91,110,128,0.16)";
  fctx.lineWidth = 1.1;
  const panelX = officeLike ? 24 : 32;
  const panelY = officeLike ? 30 : 38;
  for(let x=0;x<=256;x+=panelX){
    fctx.beginPath(); fctx.moveTo(x,0); fctx.lineTo(x,512); fctx.stroke();
  }
  for(let y=0;y<=512;y+=panelY){
    fctx.beginPath(); fctx.moveTo(0,y); fctx.lineTo(256,y); fctx.stroke();
  }

  // Emissive map background must be black: only window pixels glow.
  lctx.fillStyle = "#000000";
  lctx.fillRect(0,0,256,512);

  const density = officeLike ? 0.86 :
    neighborhood === "downtown" ? 0.78 :
    neighborhood === "market" ? 0.66 :
    neighborhood === "riverside" ? 0.56 :
    neighborhood === "residential" ? 0.38 :
    0.55;

  const profile = officeLike ? "office" : (neighborhood === "residential" ? "residential" : (neighborhood === "market" ? "shop" : "default"));
  const nightIntensity = officeLike ? 0.92 : (neighborhood === "residential" ? 0.62 : 0.74);

  const cols = officeLike ? 7 + Math.floor(Math.random()*3) : 4 + Math.floor(Math.random()*4);
  const rows = officeLike ? 15 + Math.floor(Math.random()*8) : 10 + Math.floor(Math.random()*8);
  const marginX = officeLike ? 10 : 14;
  const marginY = 14;
  const gapX = officeLike ? 6 : 8;
  const gapY = officeLike ? 6 : 8;
  const winW = (256 - 2*marginX - (cols-1)*gapX)/cols;
  const winH = (512 - 2*marginY - (rows-1)*gapY)/rows;

  const warmColors = ["#ffd166", "#ffe8a3", "#fff0bd", "#f6d28a"];
  const coolColors = ["#f8fafc", "#dbeafe", "#e0f2fe", "#eef2ff"];

  for(let r=0;r<rows;r++){
    for(let c2=0;c2<cols;c2++){
      const x = marginX + c2*(winW+gapX);
      const y = marginY + r*(winH+gapY);

      // Facade map: dark glass panels remain visible both day and night.
      fctx.fillStyle = officeLike ? "rgba(54,72,91,0.34)" : "rgba(75,88,102,0.30)";
      fctx.fillRect(x,y,winW,winH);

      // Reflective daylight highlight; not emissive.
      fctx.fillStyle = "rgba(255,255,255,0.12)";
      fctx.fillRect(x+1, y+1, winW-2, Math.max(1, winH*0.18));

      // Some windows are off. This is baked into the emissive map.
      const lit = Math.random() < density;
      if(lit){
        const useWarm = neighborhood === "residential" ? Math.random() < 0.88 :
                        neighborhood === "market" ? Math.random() < 0.76 :
                        officeLike ? Math.random() < 0.48 :
                        Math.random() < 0.68;
        const colors = useWarm ? warmColors : coolColors;
        lctx.fillStyle = colors[Math.floor(Math.random()*colors.length)];
        lctx.fillRect(x,y,winW,winH);

        // Tiny brighter core gives the window a lit-room feel.
        if(Math.random() < 0.35){
          lctx.fillStyle = "rgba(255,255,255,0.45)";
          lctx.fillRect(x+2, y+2, Math.max(1,winW-4), Math.max(1,winH*0.20));
        }
      }
    }
  }

  const facadeMap = new THREE.CanvasTexture(facadeCanvas);
  facadeMap.colorSpace = THREE.SRGBColorSpace;
  const windowLightMap = new THREE.CanvasTexture(lightCanvas);
  windowLightMap.colorSpace = THREE.SRGBColorSpace;

  return { facadeMap, windowLightMap, profile, nightIntensity };
}

function addLandmarks(){
  addHome("home1","Home A",36,-26);
  addHome("home2","Home B",50,-14);
  addHome("home3","Home C",32,-42);
  addShop("grocery","GROCERY",-22,-30,0x22c55e);
  addShop("warehouse","WAREHOUSE",-26,30,0xef4444);
  addShop("icecream","ICE CREAM",34,34,0x60a5fa);
  addShop("cafe","CAFÉ",-40,16,0xf59e0b);
  addRestingStation("station","REST STOP",6,-42);
}

function addHome(key,label,x,z){
  const group = new THREE.Group();
  const baseMat = new THREE.MeshStandardMaterial({ color:0xf8fafc, roughness:.82, emissive:0xffcc66, emissiveIntensity:isNight ? 0.04 : 0.0 });
  registerNightMaterial(baseMat, 0.04, 0.0);
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(4.6,2.8,3.8),
    baseMat
  );
  base.position.y = 1.4;
  base.castShadow = true;
  group.add(base);
  const roof = new THREE.Mesh(
    new THREE.ConeGeometry(2.6,1.5,4),
    new THREE.MeshStandardMaterial({ color:0x92400e, roughness:.74 })
  );
  roof.position.y = 3.45;
  roof.rotation.y = Math.PI/4;
  roof.castShadow = true;
  group.add(roof);
  addHouseWindow(group,-.9,1.7,1.92);
  addHouseWindow(group,.9,1.7,1.92);
  const door = new THREE.Mesh(
    new THREE.BoxGeometry(.6,1.1,.08),
    new THREE.MeshStandardMaterial({ color:0x8b5a2b, roughness:.72 })
  );
  door.position.set(0,.7,1.95);
  group.add(door);
  group.position.set(x,0,z);
  cityGroup.add(group);
  addYard(x,z,7.2,6.2);
  addFence(x,z,7.0,6.0);
  addTree(x-3,z-1.4,.62); addTree(x+2.8,z+1.2,.55);
  addFlowerPatch(x-1,z+.8);
  world.destinations[key] = { label, x:x, z:z+3.7, reachRadius:1.05 };
}

function addShop(key,text,x,z,color){
  const base = new THREE.Mesh(
    new THREE.BoxGeometry(7.8,.03,6.8),
    new THREE.MeshStandardMaterial({ color:isNight?0xc1c7cf:0xe6eaef, roughness:.9 })
  );
  base.position.set(x,.065,z);
  cityGroup.add(base);

  const h=6.2, w=5.8, d=4.6;
  const facadeMat = new THREE.MeshStandardMaterial({ color:0xffffff, roughness:.80, emissive:0xffe1a8, emissiveIntensity:isNight ? 0.05 : 0.0 });
  registerNightMaterial(facadeMat, 0.05, 0.0);
  const b = new THREE.Mesh(
    new THREE.BoxGeometry(w,h,d),
    facadeMat
  );
  b.position.set(x,h/2,z);
  b.castShadow = true;
  cityGroup.add(b);

  const awning = new THREE.Mesh(
    new THREE.BoxGeometry(w*.88,.12,.6),
    new THREE.MeshStandardMaterial({ color, roughness:.46 })
  );
  awning.position.set(x,2.55,z+d/2+.15);
  cityGroup.add(awning);

  const sign = createSign(color, text);
  sign.position.set(x,3.85,z+d/2+.04);
  cityGroup.add(sign);

  for(let i=-2;i<=2;i++){
    const glassMat = new THREE.MeshStandardMaterial({
      color:0xd9eefb,
      emissive:0xffd47a,
      emissiveIntensity:isNight ? 0.92 : 0.0,
      transparent:true,
      opacity:.82,
      roughness:.08
    });
    registerNightMaterial(glassMat, 0.92, 0.0);
    const glass = new THREE.Mesh(
      new THREE.BoxGeometry(.64,1.25,.04),
      glassMat
    );
    glass.position.set(x+i*.74,1.5,z+d/2+.03);
    cityGroup.add(glass);
  }

  addTree(x-3.2, z+1.8, .54);
  addBench(x+2.9, z+2.0, 0);
  world.obstacles.push({minX:x-w/2-.2, maxX:x+w/2+.2, minZ:z-d/2-.2, maxZ:z+d/2+.2});
  world.destinations[key] = { label:text, x:x, z:z+d/2+.85, reachRadius:1.0 };
}

function addRestingStation(key,text,x,z){
  const pad = new THREE.Mesh(
    new THREE.BoxGeometry(4.2,.03,4.2),
    new THREE.MeshStandardMaterial({ color:0xdbeafe, roughness:.85 })
  );
  pad.position.set(x,.065,z); cityGroup.add(pad);

  const postMat = new THREE.MeshStandardMaterial({ color:0x1d4ed8, roughness:.5, emissive:0x60a5fa, emissiveIntensity:isNight ? 0.12 : 0.0 });
  registerNightMaterial(postMat, 0.12, 0.0);
  const post = new THREE.Mesh(
    new THREE.BoxGeometry(.7,2.5,.5),
    postMat
  );
  post.position.set(x,1.25,z); post.castShadow = true; cityGroup.add(post);

  const sign = createSign(0x1d4ed8, text);
  sign.position.set(x,3.0,z+.4); cityGroup.add(sign);

  const bolt = new THREE.Mesh(
    new THREE.PlaneGeometry(.5,.8),
    new THREE.MeshBasicMaterial({ color:0xfacc15, side:THREE.DoubleSide })
  );
  bolt.position.set(x,1.25,z+.26); cityGroup.add(bolt);

  world.destinations[key] = { label:text, x:x, z:z+2.3, reachRadius:1.0 };
}

function createSign(color,text){
  const c = document.createElement("canvas");
  c.width = 256; c.height = 128;
  const ctx = c.getContext("2d");
  ctx.fillStyle = "rgba(255,255,255,.94)";
  ctx.fillRect(0,0,256,128);
  ctx.strokeStyle = "#" + color.toString(16).padStart(6,"0");
  ctx.lineWidth = 8;
  ctx.strokeRect(10,10,236,108);
  ctx.fillStyle = "#0f172a";
  ctx.font = "bold 40px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text,128,66);
  const tex = new THREE.CanvasTexture(c);
  const signMat = new THREE.MeshStandardMaterial({
    map:tex,
    emissive:0xffffff,
    emissiveMap:tex,
    emissiveIntensity:isNight ? 0.90 : 0.20,
    transparent:true,
    opacity:.95,
    side:THREE.DoubleSide
  });
  registerNightMaterial(signMat, 0.90, 0.20);
  return new THREE.Mesh(
    new THREE.PlaneGeometry(1.9,.86),
    signMat
  );
}

function addHouseWindow(group,x,y,z){
  const mat = new THREE.MeshStandardMaterial({
    color:0xddeeff,
    emissive:0xffd166,
    emissiveIntensity:isNight ? 0.82 : 0.0,
    roughness:.30
  });
  registerNightMaterial(mat, 0.82, 0.0);

  const w = new THREE.Mesh(
    new THREE.BoxGeometry(.52,.52,.05),
    mat
  );
  w.position.set(x,y,z);
  group.add(w);
}
function addYard(x,z,sx,sz){
  const g = new THREE.Mesh(
    new THREE.BoxGeometry(sx,.03,sz),
    new THREE.MeshStandardMaterial({ color:isNight?0x628a6b:0x94c394, roughness:.92 })
  );
  g.position.set(x,.065,z); cityGroup.add(g);
}
function addFence(x,z,sx,sz){
  const mat = new THREE.MeshStandardMaterial({ color:0xf8fafc, roughness:.84 });
  const segs = [
    [sx,.16,.06,0,sz/2],[sx,.16,.06,0,-sz/2],[.06,.16,sz,sx/2,0],[.06,.16,sz,-sx/2,0]
  ];
  for(const [w,h,d,dx,dz] of segs){
    const p = new THREE.Mesh(new THREE.BoxGeometry(w,h,d), mat);
    p.position.set(x+dx,.10,z+dz);
    cityGroup.add(p);
  }
}
function addTree(x,z,s){
  const trunk = new THREE.Mesh(
    new THREE.CylinderGeometry(.07*s,.09*s,.55*s,12),
    new THREE.MeshStandardMaterial({ color:0x8b5a2b, roughness:.74 })
  );
  trunk.position.set(x,.28*s,z); trunk.castShadow = true; cityGroup.add(trunk);
  const crown = new THREE.Mesh(
    new THREE.SphereGeometry(.28*s,16,12),
    new THREE.MeshStandardMaterial({ color:isNight?0x4f7d5d:0x5e9b66, roughness:.84 })
  );
  crown.position.set(x,.72*s,z); crown.castShadow = true; cityGroup.add(crown);
}
function addBench(x,z,rot){
  const group = new THREE.Group();
  const seat = new THREE.Mesh(new THREE.BoxGeometry(.7,.08,.18), new THREE.MeshStandardMaterial({ color:0x8b5a2b, roughness:.65 }));
  seat.position.y = .23; group.add(seat);
  const back = new THREE.Mesh(new THREE.BoxGeometry(.7,.08,.16), new THREE.MeshStandardMaterial({ color:0x8b5a2b, roughness:.65 }));
  back.position.set(0,.38,-.09); back.rotation.x = -.25; group.add(back);
  const l1 = new THREE.Mesh(new THREE.BoxGeometry(.05,.22,.05), new THREE.MeshStandardMaterial({ color:0x475569, roughness:.55 }));
  const l2 = l1.clone();
  l1.position.set(-.25,.11,.06); l2.position.set(.25,.11,.06);
  group.add(l1,l2);
  group.position.set(x,0,z); group.rotation.y = rot;
  cityGroup.add(group);
}
function addFlowerPatch(x,z){
  const colors = [0xf472b6,0xfacc15,0x60a5fa,0xfb7185];
  for(let i=0;i<6;i++){
    const f = new THREE.Mesh(new THREE.SphereGeometry(.05,8,8), new THREE.MeshBasicMaterial({ color:colors[i%colors.length] }));
    f.position.set(x+rand(-.2,.2), .11, z+rand(-.2,.2));
    cityGroup.add(f);
  }
}

function addTrafficSignalsAndCrossings(){
  for(const vr of MAP.verticalRoads){
    for(const hr of MAP.horizontalRoads){
      if(vr.cls !== "major" && hr.cls !== "major") continue;
      const signal = { interX:vr.x, interZ:hr.z };
      world.trafficSignals.push(signal);

      const points = [
        [vr.x-vr.width/2-.45, hr.z-hr.width/2-.45],
        [vr.x+vr.width/2+.45, hr.z-hr.width/2-.45],
        [vr.x-vr.width/2-.45, hr.z+hr.width/2+.45],
        [vr.x+vr.width/2+.45, hr.z+hr.width/2+.45]
      ];
      for(const [x,z] of points){
        const pole = new THREE.Mesh(
          new THREE.CylinderGeometry(.05,.05,1.8,10),
          new THREE.MeshStandardMaterial({ color:0x64748b, roughness:.55, metalness:.2 })
        );
        pole.position.set(x,.9,z); cityGroup.add(pole);

        const box = new THREE.Mesh(
          new THREE.BoxGeometry(.16,.52,.16),
          new THREE.MeshStandardMaterial({ color:0x111827, roughness:.40 })
        );
        box.position.set(x,1.55,z); cityGroup.add(box);

        const red = new THREE.Mesh(new THREE.SphereGeometry(.04,10,10), new THREE.MeshBasicMaterial({ color:0x444 }));
        red.position.set(x,1.68,z); cityGroup.add(red);
        const yellow = new THREE.Mesh(new THREE.SphereGeometry(.04,10,10), new THREE.MeshBasicMaterial({ color:0x444 }));
        yellow.position.set(x,1.56,z); cityGroup.add(yellow);
        const green = new THREE.Mesh(new THREE.SphereGeometry(.04,10,10), new THREE.MeshBasicMaterial({ color:0x444 }));
        green.position.set(x,1.44,z); cityGroup.add(green);

        signal.red = red; signal.yellow = yellow; signal.green = green;
      }

      world.crossings.push({
        x: vr.x, z: hr.z,
        horizontal: { start:{x:vr.x-vr.width*.5-1.2, z:hr.z}, end:{x:vr.x+vr.width*.5+1.2, z:hr.z} },
        vertical: { start:{x:vr.x, z:hr.z-hr.width*.5-1.2}, end:{x:vr.x, z:hr.z+hr.width*.5+1.2} }
      });
    }
  }
}

function buildGraph(){
  const nodes = [];
  for(const vr of MAP.verticalRoads){
    for(const hr of MAP.horizontalRoads){
      nodes.push({ id:`n_${vr.id}_${hr.id}`, x:vr.x, z:hr.z, vr, hr });
    }
  }
  world.graphNodes = nodes;
  world.graphEdges = new Map();
  for(const n of nodes) world.graphEdges.set(n.id, []);

  for(const vr of MAP.verticalRoads){
    const same = nodes.filter(n=>n.vr.id===vr.id).sort((a,b)=>a.z-b.z);
    for(let i=0;i<same.length-1;i++){
      const a=same[i], b=same[i+1];
      const crosses = a.z < MAP.riverMinZ && b.z > MAP.riverMaxZ;
      if(crosses && !vr.bridge) continue;
      connect(a,b,Math.abs(a.z-b.z));
    }
  }
  for(const hr of MAP.horizontalRoads){
    const same = nodes.filter(n=>n.hr.id===hr.id).sort((a,b)=>a.x-b.x);
    for(let i=0;i<same.length-1;i++){
      const a=same[i], b=same[i+1];
      connect(a,b,Math.abs(a.x-b.x));
    }
  }
}
function connect(a,b,w){
  world.graphEdges.get(a.id).push({ to:b.id, w });
  world.graphEdges.get(b.id).push({ to:a.id, w });
}

function addVehicles(){
  const carColors = [0xe11d48,0x2563eb,0x64748b,0xf59e0b,0x16a34a,0xffffff];
  for(let i=0;i<24;i++){
    const seg = world.driveSegments[Math.floor(Math.random()*world.driveSegments.length)];
    const vehicle = createVehicle(carColors[i % carColors.length], i % 7 === 0 ? "bus" : "car");
    const dir = Math.random() < .5 ? 1 : -1;
    vehicle.userData = {
      kind: i % 7 === 0 ? "bus" : "car",
      orientation: seg.orientation,
      fixed: seg.orientation==="v" ? seg.x : seg.z,
      min: seg.min, max: seg.max,
      width: seg.width,
      t: rand(seg.min, seg.max),
      targetSpeed: rand(1.8, 3.8)*dir,
      speed: 0,
      laneOffset: laneOffset(seg.width, dir)
    };
    placeVehicle(vehicle);
    dynamicGroup.add(vehicle);
    world.cars.push(vehicle);
  }
}
function createVehicle(color, kind){
  const group = new THREE.Group();
  const bodyLen = kind === "bus" ? 1.9 : 1.16;
  const bodyH = kind === "bus" ? .42 : .30;
  const bodyW = kind === "bus" ? .68 : .56;
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(bodyW, bodyH, bodyLen),
    new THREE.MeshStandardMaterial({ color, roughness:.36, metalness:.18 })
  );
  body.position.y = kind === "bus" ? .25 : .18;
  body.castShadow = true;
  group.add(body);

  const cabin = new THREE.Mesh(
    new THREE.BoxGeometry(bodyW*.82, bodyH*.8, bodyLen*.5),
    new THREE.MeshStandardMaterial({ color:0xd9e6ef, transparent:true, opacity:.88, roughness:.08 })
  );
  cabin.position.set(0, kind === "bus" ? .44 : .38, -.02);
  group.add(cabin);

  const frontZ = bodyLen / 2 + 0.02;
  const rearZ = -bodyLen / 2 - 0.02;
  const y = kind === "bus" ? 0.20 : 0.14;
  const xOff = bodyW * 0.28;

  const headMaterials = [];
  const tailMaterials = [];
  const pointLights = [];

  for(const sx of [-1, 1]){
    const headMat = new THREE.MeshStandardMaterial({
      color:0xfff8df,
      emissive:0xfff1c2,
      emissiveIntensity:isNight ? 1.8 : 0.0,
      roughness:0.22
    });
    const head = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.05), headMat);
    head.position.set(sx * xOff, y, frontZ);
    group.add(head);
    headMaterials.push(headMat);

    const tailMat = new THREE.MeshStandardMaterial({
      color:0xff6b6b,
      emissive:0xff3333,
      emissiveIntensity:isNight ? 1.1 : 0.0,
      roughness:0.22
    });
    const tail = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.05, 0.05), tailMat);
    tail.position.set(sx * xOff, y, rearZ);
    group.add(tail);
    tailMaterials.push(tailMat);
  }

  // A small forward headlight beam
  const leftLight = new THREE.PointLight(0xfff1c2, isNight ? 0.6 : 0.0, kind === "bus" ? 5.8 : 4.2, 2.0);
  leftLight.position.set(-xOff, y, frontZ + 0.10);
  leftLight.userData.baseIntensity = 0.6;
  group.add(leftLight);

  const rightLight = new THREE.PointLight(0xfff1c2, isNight ? 0.6 : 0.0, kind === "bus" ? 5.8 : 4.2, 2.0);
  rightLight.position.set(xOff, y, frontZ + 0.10);
  rightLight.userData.baseIntensity = 0.6;
  group.add(rightLight);

  pointLights.push(leftLight, rightLight);
  world.vehicleLights.push({ headMaterials, tailMaterials, pointLights });

  return group;
}
function laneOffset(width, dir){
  if(width >= 4.8) return dir > 0 ? -width*.22 : width*.22;
  return dir > 0 ? -.52 : .52;
}
function placeVehicle(v){
  const u = v.userData;
  if(u.orientation === "v"){
    v.position.set(u.fixed + u.laneOffset, .02, u.t);
    v.rotation.y = u.targetSpeed >= 0 ? 0 : Math.PI;
  } else {
    v.position.set(u.t, .02, u.fixed + u.laneOffset);
    v.rotation.y = u.targetSpeed >= 0 ? Math.PI/2 : -Math.PI/2;
  }
}

function getSignalPhase(){
  const T = 24;
  const p = clock.elapsedTime % T;
  if(p < 10) return { vertical:"green", horizontal:"red", label:"Vertical Green", crossing:"horizontal" };
  if(p < 12) return { vertical:"yellow", horizontal:"red", label:"Vertical Yellow", crossing:"stop" };
  if(p < 22) return { vertical:"red", horizontal:"green", label:"Horizontal Green", crossing:"vertical" };
  return { vertical:"red", horizontal:"yellow", label:"Horizontal Yellow", crossing:"stop" };
}
function updateSignals(){
  const s = getSignalPhase();
  signalText.textContent = s.label;
  for(const signal of world.trafficSignals){
    signal.red?.material.color.set((s.vertical==="red" || s.horizontal==="red") ? 0xef4444 : 0x444444);
    signal.yellow?.material.color.set((s.vertical==="yellow" || s.horizontal==="yellow") ? 0xfacc15 : 0x444444);
    signal.green?.material.color.set((s.vertical==="green" || s.horizontal==="green") ? 0x22c55e : 0x444444);
  }
}
function shouldVehicleStop(vehicle, signal){
  const u = vehicle.userData;
  const stopBuffer = u.kind === "bus" ? 3.3 : 2.6;
  if(u.orientation === "v"){
    const relevant = world.graphNodes.filter(n => Math.abs(n.x-u.fixed) < .001);
    for(const node of relevant){
      const delta = node.z - u.t;
      if(Math.sign(delta) === Math.sign(u.targetSpeed) && Math.abs(delta) < stopBuffer){
        if(signal.vertical !== "green") return true;
      }
    }
  } else {
    const relevant = world.graphNodes.filter(n => Math.abs(n.z-u.fixed) < .001);
    for(const node of relevant){
      const delta = node.x - u.t;
      if(Math.sign(delta) === Math.sign(u.targetSpeed) && Math.abs(delta) < stopBuffer){
        if(signal.horizontal !== "green") return true;
      }
    }
  }
  return false;
}
function wouldHitFrontVehicle(vehicle, dt){
  const u = vehicle.userData;
  for(const other of world.cars){
    if(other === vehicle) continue;
    const o = other.userData;
    if(o.orientation !== u.orientation) continue;
    if(Math.abs(o.fixed-u.fixed) > .01) continue;
    if(Math.abs(o.laneOffset-u.laneOffset) > .2) continue;
    if(Math.sign(o.targetSpeed) !== Math.sign(u.targetSpeed)) continue;
    const gap = (o.t - u.t) * Math.sign(u.targetSpeed);
    if(gap > 0 && gap < (u.kind==="bus" ? 2.4 : 1.4)) return true;
  }
  return false;
}
function updateVehicles(dt){
  const signal = getSignalPhase();
  for(const v of world.cars){
    const u = v.userData;
    const stopSignal = shouldVehicleStop(v, signal);
    const stopFront = wouldHitFrontVehicle(v, dt);
    const desired = (stopSignal || stopFront) ? 0 : u.targetSpeed;
    u.speed += (desired-u.speed) * Math.min(1, dt*4.4);
    u.t += u.speed*dt;
    if(u.t > u.max) u.t = u.min;
    if(u.t < u.min) u.t = u.max;
    placeVehicle(v);
  }
}

function addPedestrians(){
  const pedColors = [0x1d4ed8,0x16a34a,0xd97706,0xec4899,0x475569,0xdc2626];
  // lively sidewalks
  for(let i=0;i<32;i++){
    const p = randomSidewalkPoint();
    const ped = createPerson(pedColors[i % pedColors.length], false);
    ped.scale.setScalar(.62);
    ped.position.set(p.x, 0, p.z);
    ped.userData.walk = {
      mode: "wander",
      baseX: ped.position.x,
      baseZ: ped.position.z,
      phase: Math.random()*10,
      radius: rand(.18,.72)
    };
    dynamicGroup.add(ped);
    world.pedestrians.push(ped);
  }
  // crossing pedestrians
  for(let i=0;i<22;i++){
    const cross = world.crossings[i % world.crossings.length];
    const direction = i % 2 === 0 ? "horizontal" : "vertical";
    const ped = createPerson(pedColors[(i+3) % pedColors.length], false);
    ped.scale.setScalar(.62);
    const lane = cross[direction];
    ped.position.set(lane.start.x, 0, lane.start.z);
    ped.userData.walk = {
      mode: "cross",
      crossing: cross,
      direction,
      progress: Math.random(),
      speed: rand(.22,.40),
      forward: Math.random() < .5 ? 1 : -1
    };
    dynamicGroup.add(ped);
    world.pedestrians.push(ped);
  }  // extra pedestrians waiting at curbs make the city feel more alive
  for(let i=0;i<14;i++){
    const cross = world.crossings[i % world.crossings.length];
    const offset = i % 2 === 0 ? -1.55 : 1.55;
    const ped = createPerson(pedColors[(i+5) % pedColors.length], false);
    ped.scale.setScalar(.62);
    ped.position.set(cross.x + offset, 0, cross.z + (i % 3 - 1) * 0.55);
    ped.userData.walk = {
      mode: "wait",
      baseX: ped.position.x,
      baseZ: ped.position.z,
      phase: Math.random()*10,
      radius: rand(.04,.14)
    };
    dynamicGroup.add(ped);
    world.pedestrians.push(ped);
  }

}
function createPerson(color, main=false){
  const g = new THREE.Group();
  const bodyMat = new THREE.MeshStandardMaterial({ color, roughness:.55 });
  const skinMat = new THREE.MeshStandardMaterial({ color:0xf1c27d, roughness:.68 });
  const darkMat = new THREE.MeshStandardMaterial({ color:0x334155, roughness:.62 });

  const torso = new THREE.Mesh(new THREE.CylinderGeometry(.18,.2,.5,18), bodyMat);
  torso.position.y = .7; torso.castShadow = true; g.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(.15,20,20), skinMat);
  head.position.y = 1.08; head.castShadow = true; g.add(head);

  const legLeft = makeLimb(.09,.42,darkMat);
  const legRight = makeLimb(.09,.42,darkMat);
  const armLeft = makeLimb(.07,.34,darkMat);
  const armRight = makeLimb(.07,.34,darkMat);
  legLeft.position.set(-.07,.42,0);
  legRight.position.set(.07,.42,0);
  armLeft.position.set(-.22,.82,0);
  armRight.position.set(.22,.82,0);
  g.add(legLeft,legRight,armLeft,armRight);

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(main ? .3 : .22, 28),
    new THREE.MeshBasicMaterial({ color:0x000000, transparent:true, opacity:main?.20:.14 })
  );
  shadow.rotation.x = -Math.PI/2;
  shadow.position.y = .018;
  g.add(shadow);

  g.userData.legLeft = legLeft;
  g.userData.legRight = legRight;
  g.userData.armLeft = armLeft;
  g.userData.armRight = armRight;
  return g;
}
function makeLimb(w,h,mat){
  const g = new THREE.Group();
  const m = new THREE.Mesh(new THREE.BoxGeometry(w,h,w), mat);
  m.position.y = -h/2;
  m.castShadow = true;
  g.add(m);
  return g;
}
function randomSidewalkPoint(){
  const seg = world.driveSegments[Math.floor(Math.random()*world.driveSegments.length)];
  if(seg.orientation === "v"){
    return { x:seg.x + (Math.random()<.5 ? -2.0 : 2.0), z:rand(seg.min,seg.max) };
  }
  return { x:rand(seg.min,seg.max), z:seg.z + (Math.random()<.5 ? -2.0 : 2.0) };
}
function updatePedestrians(dt){
  const signal = getSignalPhase();
  world.pedestrians.forEach((ped, idx) => {
    const w = ped.userData.walk;
    if(w.mode === "wander"){
      const t = clock.elapsedTime*.35 + w.phase;
      ped.position.x = w.baseX + Math.cos(t)*w.radius;
      ped.position.z = w.baseZ + Math.sin(t)*w.radius;
      ped.rotation.y = t + Math.PI/2;
    } else if(w.mode === "wait"){
      const t = clock.elapsedTime*0.55 + w.phase;
      ped.position.x = w.baseX + Math.cos(t)*w.radius;
      ped.position.z = w.baseZ + Math.sin(t)*w.radius;
      ped.rotation.y = Math.sin(t) * 0.35;
    } else if(w.mode === "cross"){
      const lane = w.crossing[w.direction];
      const allow = (signal.crossing === w.direction);
      if(allow){
        w.progress += dt*w.speed*w.forward;
        if(w.progress > 1){ w.progress = 1; w.forward = -1; }
        if(w.progress < 0){ w.progress = 0; w.forward = 1; }
      }
      ped.position.x = lerp(lane.start.x, lane.end.x, w.progress);
      ped.position.z = lerp(lane.start.z, lane.end.z, w.progress);
      ped.rotation.y = w.direction === "horizontal" ? (w.forward>0 ? Math.PI/2 : -Math.PI/2) : (w.forward>0 ? 0 : Math.PI);
    }

    const swing = Math.sin(clock.elapsedTime*7.0 + idx)*0.42;
    ped.userData.legLeft.rotation.x = swing;
    ped.userData.legRight.rotation.x = -swing;
  });
}

function createAgents(){
  while(agents.length){
    const a = agents.pop();
    dynamicGroup.remove(a.mesh);
  }

  for(let i=0;i<3;i++){
    const mesh = createPerson(agentColors[i], true);
    mesh.scale.setScalar(.92);
    dynamicGroup.add(mesh);

    const agent = {
      id:i,
      name:i===0 ? "You" : (i===1 ? "Courier A" : "Courier B"),
      mesh,
      x:0, z:0,
      vx:0, vz:0,
      speed:i===0 ? 4.9 : 4.2,
      carrying:false,
      energy:100,
      activeTarget:null,
      path:[],
      idle:0,
      collisions:0
    };
    agents.push(agent);
  }

  mainAgent = agents[0];
  spawnAgent(mainAgent, 10, -18);
  spawnAgent(agents[1], -12, 22);
  spawnAgent(agents[2], 28, 20);

  assignSecondaryAgentTask(agents[1]);
  assignSecondaryAgentTask(agents[2]);
}
function spawnAgent(agent,x,z){
  agent.x=x; agent.z=z; agent.vx=0; agent.vz=0; agent.energy=100; agent.carrying=false; agent.path=[]; agent.idle=0;
  agent.mesh.position.set(x,0,z);
}
function assignSecondaryAgentTask(agent){
  const choices = ["grocery","icecream","cafe","home1","home2","home3"];
  const targetKey = choices[Math.floor(Math.random()*choices.length)];
  agent.activeTarget = world.destinations[targetKey];
  agent.path = planPath(agent.x, agent.z, agent.activeTarget.x, agent.activeTarget.z);
}

function resetDailyExperience(incrementDay=true){
  if(incrementDay){
    simulatedDay += 1;
    episode += 1;
  }

  episodeText.textContent = String(simulatedDay);
  dayText.textContent = String(simulatedDay);

  simulatedMinutes = 6 * 60;
  taskTimer = 0;
  cumulativeReward = 0;
  lastReward = 0;
  rewardHistory = [];
  returnHistory = [];
  rewardTickAcc = 0;

  dailyChoresCompleted = 0;
  collisions = 0;
  trafficViolations = 0;
  pedestrianHits = 0;
  collisionCooldown = 0;
  trafficViolationCooldown = 0;
  pedestrianCollisionCooldown = 0;
  manualTarget = null;
  autoMode = true;
  learningMode = true;
  learnModeBtn.textContent = "Pause Learning";
  clickMarker.visible = false;

  spawnAgent(mainAgent, 10, -18);
  spawnAgent(agents[1], -12, 22);
  spawnAgent(agents[2], 28, 20);
  assignSecondaryAgentTask(agents[1]);
  assignSecondaryAgentTask(agents[2]);

  mainAgent.energy = 100;
  mainAgent.path = [];

  randomizeDailyWeather();
  updateDayNightFromClock(true);

  chooseAndStartNextChore();
  resetCamera();
  statusText.textContent = incrementDay ? "New day started" : "Learning day initialized";
  emitSimEvent("daily-reset", { day:simulatedDay });
}

function resetEpisode(increment=true){
  resetDailyExperience(increment);
}

function resetMainPosition(){
  spawnAgent(mainAgent, 10, -18);
  manualTarget = null;
  autoMode = false;
  learningMode = false;
  learnModeBtn.textContent = "Resume Learning";
  mainAgent.path = [];
  clickMarker.visible = false;
  statusText.textContent = "Position reset";
}

function startTask(kind){
  mainAgent.carrying = false;
  currentChoreKey = kind;
  currentTask = { kind, phase:"to_target", targets:[], currentIndex:0 };

  taskTimer = 0;
  lastReward = 0;
  taskStartReturn = cumulativeReward;

  // In manual/non-learning mode, a task is a single episode. In continuous mode,
  // cumulativeReward and rewardHistory are daily quantities and are not reset per chore.
  if(!learningMode){
    cumulativeReward = 0;
    rewardHistory = [];
    returnHistory = [];
    rewardTickAcc = 0;
  }

  if(kind === "home"){
    const homes = ["home1","home2","home3"];
    currentTask.targets = [homes[Math.floor(Math.random()*homes.length)]];
    taskTitle.textContent = "Chore: Return Home";
    taskDesc.textContent = "The agent decides whether going home now is worth the time and energy cost.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +25 at home, -18 collision, -8 traffic violation.";
    taskStatText.textContent = "Return Home";
  } else if(kind === "grocery"){
    currentTask.targets = ["grocery"];
    taskTitle.textContent = "Chore: Grocery Run";
    taskDesc.textContent = "The agent explores whether a grocery trip is efficient under current traffic, energy, and weather.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +25 at grocery, penalties for unsafe behavior.";
    taskStatText.textContent = "Grocery Run";
  } else if(kind === "icecream"){
    currentTask.targets = ["icecream"];
    taskTitle.textContent = "Chore: Visit Ice Cream Shop";
    taskDesc.textContent = "The agent may choose this optional errand if it has learned that the trip is worthwhile.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +25 at ice cream center.";
    taskStatText.textContent = "Get Ice Cream";
  } else if(kind === "delivery"){
    const homes = ["home1","home2","home3"];
    const drop = homes[Math.floor(Math.random()*homes.length)];
    currentTask.targets = ["warehouse", drop];
    currentTask.phase = "pickup";
    taskTitle.textContent = "Chore: Pickup and Delivery";
    taskDesc.textContent = "Pick up a package from the warehouse, then deliver it to a home.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +10 pickup, +30 dropoff, safety penalties.";
    taskStatText.textContent = "Pickup & Dropoff";
  } else if(kind === "multidelivery"){
    const homes = ["home1","home2","home3"];
    const order = shortestHomeOrder(world.destinations["warehouse"], homes);
    currentTask.targets = ["warehouse", ...order];
    currentTask.phase = "pickup";
    taskTitle.textContent = "Chore: Delivery Route";
    taskDesc.textContent = "The agent tries a longer delivery route and learns whether the reward is worth the extra travel cost.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +10 pickup, +20 each delivery, +10 completion.";
    taskStatText.textContent = "Multi-Delivery";
  } else if(kind === "charge"){
    currentTask.targets = ["station"];
    currentTask.phase = "resting";
    taskTitle.textContent = "Chore: Rest and Recover";
    taskDesc.textContent = "When energy is low, the agent can learn that going to the rest stop prevents future failure.";
    taskRewardInfo.textContent = "Reward: -0.08/s, energy cost, +12 when energy is mostly restored.";
    taskStatText.textContent = "Rest";
  }

  setCurrentTarget(world.destinations[currentTask.targets[0]]);
  packageText.textContent = kind === "charge" ? "Rest" : "None";
  drawTargetBeacon();
  statusText.textContent = learningMode ? "Exploring next chore" : "Task active";

  if(learningMode || autoMode){
    autoMode = true;
    mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
    drawPolicyPath(mainAgent.path, 0x2563eb, true);
    autoBtn.textContent = "Learning Autonomously";
  }
}

function shortestHomeOrder(start, homes){
  const perms = permutations(homes);
  let best = perms[0], bestCost = Infinity;
  for(const perm of perms){
    let cost = routeDistance(start, world.destinations[perm[0]]);
    for(let i=0;i<perm.length-1;i++) cost += routeDistance(world.destinations[perm[i]], world.destinations[perm[i+1]]);
    if(cost < bestCost){ bestCost = cost; best = perm; }
  }
  return best;
}
function permutations(arr){
  if(arr.length <= 1) return [arr.slice()];
  const out = [];
  for(let i=0;i<arr.length;i++){
    const head = arr[i];
    const rest = arr.slice(0,i).concat(arr.slice(i+1));
    for(const tail of permutations(rest)) out.push([head, ...tail]);
  }
  return out;
}
function setCurrentTarget(dest){
  currentTarget = { x:dest.x, z:dest.z, label:dest.label, reachRadius:dest.reachRadius || 1.0 };
  targetText.textContent = currentTarget.label;
}
function drawTargetBeacon(){
  world.beacons.forEach(b => dynamicGroup.remove(b));
  world.beacons = [];
  const b = createBeacon();
  b.position.set(currentTarget.x,0,currentTarget.z);
  dynamicGroup.add(b);
  world.beacons.push(b);
}

function createBeacon(){
  const g = new THREE.Group();
  const base = new THREE.Mesh(
    new THREE.CylinderGeometry(.38,.48,.12,28),
    new THREE.MeshStandardMaterial({ color:0xffffff, roughness:.5 })
  );
  base.position.y = .06;
  g.add(base);
  const column = new THREE.Mesh(
    new THREE.CylinderGeometry(.22,.22,1.72,32,1,true),
    new THREE.MeshStandardMaterial({ color:0xf59e0b, emissive:0xf59e0b, emissiveIntensity:.9, transparent:true, opacity:.18, side:THREE.DoubleSide })
  );
  column.position.y = .86; g.add(column);
  const orb = new THREE.Mesh(
    new THREE.SphereGeometry(.24,24,24),
    new THREE.MeshStandardMaterial({ color:0xf59e0b, emissive:0xf59e0b, emissiveIntensity:1.2, roughness:.1 })
  );
  orb.position.y = 1.78; g.add(orb);
  g.userData = { orb, column };
  return g;
}

function createClickMarker(){
  const g = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(.34,.026,12,64),
    new THREE.MeshBasicMaterial({ color:0x2563eb, transparent:true, opacity:.9 })
  );
  ring.rotation.x = Math.PI/2; g.add(ring);
  const dot = new THREE.Mesh(
    new THREE.SphereGeometry(.06,18,18),
    new THREE.MeshBasicMaterial({ color:0x2563eb })
  );
  dot.position.y = .07; g.add(dot);
  g.visible = false;
  return g;
}

function animateBeacon(){
  const t = clock.elapsedTime;
  for(const b of world.beacons){
    b.rotation.y += .01;
    b.userData.orb.position.y = 1.78 + .10*Math.sin(t*2.6);
    b.userData.column.material.opacity = .14 + .05*(Math.sin(t*2.2)**2);
  }
}
function animateClickMarker(){
  if(!clickMarker.visible) return;
  clickMarker.rotation.y += .04;
  clickMarker.position.y = .06 + .03*Math.sin(clock.elapsedTime*5.0);
}

function pointInObstacle(x,z){
  return world.obstacles.some(o => x>o.minX && x<o.maxX && z>o.minZ && z<o.maxZ);
}
function nearestRoadProjection(x,z){
  let best = null, bestD = Infinity;
  for(const seg of world.driveSegments){
    if(seg.orientation === "v"){
      const px = seg.x, pz = clamp(z, seg.min, seg.max);
      const d = Math.hypot(x-px, z-pz);
      if(d < bestD){ bestD = d; best = { x:px, z:pz, seg }; }
    } else {
      const px = clamp(x, seg.min, seg.max), pz = seg.z;
      const d = Math.hypot(x-px, z-pz);
      if(d < bestD){ bestD = d; best = { x:px, z:pz, seg }; }
    }
  }
  return best;
}
function getLinksForProjectedPoint(p){
  const res = [];
  if(p.seg.orientation === "v"){
    const nodes = world.graphNodes.filter(n => Math.abs(n.x-p.seg.x) < .001);
    for(const n of nodes){
      if(!p.seg.bridge){
        if((n.z >= MAP.riverMaxZ && p.z >= MAP.riverMaxZ) || (n.z <= MAP.riverMinZ && p.z <= MAP.riverMinZ)){
          res.push({ node:n, cost:Math.abs(n.z-p.z) });
        }
      } else res.push({ node:n, cost:Math.abs(n.z-p.z) });
    }
  } else {
    const nodes = world.graphNodes.filter(n => Math.abs(n.z-p.seg.z) < .001);
    for(const n of nodes) res.push({ node:n, cost:Math.abs(n.x-p.x) });
  }
  return res;
}
function planPath(startX,startZ,goalX,goalZ){
  const startP = nearestRoadProjection(startX,startZ);
  const goalP = nearestRoadProjection(goalX,goalZ);
  const startLinks = getLinksForProjectedPoint(startP);
  const goalLinks = getLinksForProjectedPoint(goalP);

  const distMap = new Map();
  const prev = new Map();
  const pq = [];
  for(const n of world.graphNodes) distMap.set(n.id, Infinity);
  for(const link of startLinks){
    distMap.set(link.node.id, link.cost);
    pq.push({ id:link.node.id, d:link.cost });
  }

  while(pq.length){
    pq.sort((a,b)=>a.d-b.d);
    const cur = pq.shift();
    if(cur.d !== distMap.get(cur.id)) continue;
    for(const e of world.graphEdges.get(cur.id)){
      const nd = cur.d + e.w;
      if(nd < distMap.get(e.to)){
        distMap.set(e.to, nd);
        prev.set(e.to, cur.id);
        pq.push({ id:e.to, d:nd });
      }
    }
  }

  let bestGoal = null, bestGoalCost = Infinity;
  for(const link of goalLinks){
    const total = distMap.get(link.node.id) + link.cost;
    if(total < bestGoalCost){
      bestGoalCost = total;
      bestGoal = link.node.id;
    }
  }

  const pts = [{x:startP.x, z:startP.z}];
  if(bestGoal){
    const stack = [];
    let cur = bestGoal;
    while(cur){
      stack.push(cur);
      cur = prev.get(cur);
    }
    stack.reverse();
    for(const id of stack){
      const n = world.graphNodes.find(q => q.id === id);
      pts.push({x:n.x, z:n.z});
    }
  }
  pts.push({x:goalP.x, z:goalP.z});
  pts.push({x:goalX, z:goalZ});

  const out = [];
  for(const p of pts){
    if(out.length === 0 || Math.hypot(out[out.length-1].x-p.x, out[out.length-1].z-p.z) > .01) out.push(p);
  }
  return out;
}
function routeDistance(a,b){
  const path = planPath(a.x,a.z,b.x,b.z);
  let total = 0;
  for(let i=0;i<path.length-1;i++) total += Math.hypot(path[i+1].x-path[i].x, path[i+1].z-path[i].z);
  return total;
}

function drawPolicyPath(path, color=0x2563eb, clear=true){
  if(clear) clearGroup(pathGroup);
  if(!path || path.length < 2) return;
  const mat = new THREE.MeshBasicMaterial({ color, transparent:true, opacity:.88 });
  for(let i=0;i<path.length-1;i++){
    const a = path[i], b = path[i+1];
    const dx = b.x-a.x, dz = b.z-a.z;
    const len = Math.hypot(dx,dz);
    const midX = (a.x+b.x)/2, midZ = (a.z+b.z)/2;
    const seg = new THREE.Mesh(new THREE.BoxGeometry(.18,.05,len), mat);
    seg.position.set(midX,.22,midZ);
    seg.rotation.y = Math.atan2(dx,dz);
    pathGroup.add(seg);

    const arrow = new THREE.Mesh(
      new THREE.ConeGeometry(.22,.42,8),
      new THREE.MeshBasicMaterial({ color, transparent:true, opacity:.92 })
    );
    arrow.position.set(b.x,.28,b.z);
    arrow.rotation.x = Math.PI;
    arrow.rotation.y = Math.atan2(dx,dz);
    pathGroup.add(arrow);
  }
}

function addPathNodesPreview(show){
  // placeholder to keep interface clean; route is drawn dynamically
}

function setMainTargetFromPointer(event){
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX-rect.left)/rect.width)*2 - 1;
  pointer.y = -((event.clientY-rect.top)/rect.height)*2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const hit = new THREE.Vector3();
  raycaster.ray.intersectPlane(groundPlane, hit);
  const snap = nearestRoadProjection(clamp(hit.x,-MAP.worldHalf,MAP.worldHalf), clamp(hit.z,-MAP.worldHalf,MAP.worldHalf));
  if(!snap) return;

  manualTarget = { x:snap.x, z:snap.z };
  autoMode = false;
  learningMode = false;
  learnModeBtn.textContent = "Resume Learning";
  mainAgent.path = [];
  autoBtn.textContent = "Autonomous Mode";
  clickMarker.visible = true;
  clickMarker.position.set(snap.x,.06,snap.z);
  drawPolicyPath([{x:mainAgent.x,z:mainAgent.z}, manualTarget], 0x2563eb, true);
}

function updateMainAgent(dt){
  let target = null;
  if(autoMode){
    if(mainAgent.path.length === 0 && currentTarget){
      mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
      drawPolicyPath(mainAgent.path, 0x2563eb, true);
    }
    target = mainAgent.path[0] || null;
    if(target && Math.hypot(mainAgent.x-target.x, mainAgent.z-target.z) < .40){
      mainAgent.path.shift();
      target = mainAgent.path[0] || null;
    }
  } else if(manualTarget){
    target = manualTarget;
    if(Math.hypot(mainAgent.x-target.x, mainAgent.z-target.z) < .40){
      manualTarget = null;
      clickMarker.visible = false;
      clearGroup(pathGroup);
    }
  }

  moveAgentToward(mainAgent, target, dt);

  // energy dynamics
  const speed = Math.hypot(mainAgent.vx, mainAgent.vz);
  mainAgent.energy = Math.max(0, mainAgent.energy - dt*(0.9 + speed*0.25));
  if(mainAgent.energy < 18 && currentTask && currentTask.kind !== "resting"){
    statusText.textContent = "Low energy";
  }
  if(mainAgent.energy <= 8 && currentTask && currentTask.targets[0] !== "station" && currentTask.phase !== "resting"){
    statusText.textContent = "Energy critical: route to rest stop";
  }
}

function moveAgentToward(agent, target, dt){
  if(target){
    const dx = target.x-agent.x, dz = target.z-agent.z;
    const d = Math.max(.001, Math.hypot(dx,dz));
    const desiredVx = agent.speed * dx / d;
    const desiredVz = agent.speed * dz / d;
    agent.vx += (desiredVx-agent.vx)*Math.min(1, dt*4.0);
    agent.vz += (desiredVz-agent.vz)*Math.min(1, dt*4.0);
  } else {
    agent.vx *= Math.pow(.05, dt);
    agent.vz *= Math.pow(.05, dt);
  }

  const nx = agent.x + agent.vx*dt;
  const nz = agent.z + agent.vz*dt;
  if(!pointInObstacle(nx,nz)){
    agent.x = nx; agent.z = nz;
  }
  agent.mesh.position.set(agent.x,0,agent.z);
  animatePerson(agent.mesh, agent.vx, agent.vz);
}

function animatePerson(mesh,vx,vz){
  const speed = Math.hypot(vx,vz);
  if(speed > .05){
    mesh.rotation.y = Math.atan2(vx, vz);
    const swing = Math.sin(clock.elapsedTime*10.0) * Math.min(.65, speed*.25);
    mesh.userData.legLeft.rotation.x = swing;
    mesh.userData.legRight.rotation.x = -swing;
    mesh.userData.armLeft.rotation.x = -.8*swing;
    mesh.userData.armRight.rotation.x = .8*swing;
    mesh.position.y = .02*Math.sin(clock.elapsedTime*10.0);
  } else {
    mesh.userData.legLeft.rotation.x *= .85;
    mesh.userData.legRight.rotation.x *= .85;
    mesh.userData.armLeft.rotation.x *= .85;
    mesh.userData.armRight.rotation.x *= .85;
    mesh.position.y *= .85;
  }
}

function updateSecondaryAgents(dt){
  for(let i=1;i<agents.length;i++){
    const a = agents[i];
    if(!a.activeTarget || a.path.length === 0){
      assignSecondaryAgentTask(a);
    }
    let target = a.path[0] || null;
    if(target && Math.hypot(a.x-target.x, a.z-target.z) < .40){
      a.path.shift();
      target = a.path[0] || null;
    }
    moveAgentToward(a, target, dt);
    a.energy = Math.max(20, a.energy - dt*.25);

    if(a.activeTarget && Math.hypot(a.x-a.activeTarget.x, a.z-a.activeTarget.z) < 1.1){
      a.idle += dt;
      if(a.idle > .3){
        assignSecondaryAgentTask(a);
        a.idle = 0;
      }
    }
  }
}

function checkCollisions(dt){
  if(collisionCooldown > 0){ collisionCooldown -= dt; }

  if(collisionCooldown <= 0){
    for(const v of world.cars){
      const d = Math.hypot(mainAgent.x-v.position.x, mainAgent.z-v.position.z);
      if(d < 0.85){
        collisions += 1;
        mainAgent.collisions += 1;
        collisionCooldown = 1.25;
        lastReward = -18;
        cumulativeReward += lastReward;
        statusText.textContent = "Vehicle collision penalty!";
        emitSimEvent("collision", { kind:"vehicle", penalty:lastReward, x:mainAgent.x, z:mainAgent.z });
        const safe = nearestRoadProjection(mainAgent.x, mainAgent.z);
        if(safe){
          mainAgent.x = safe.x;
          mainAgent.z = safe.z;
        }
        mainAgent.vx = 0; mainAgent.vz = 0;
        mainAgent.mesh.position.set(mainAgent.x, 0, mainAgent.z);
        break;
      }
    }
  }

  checkPedestrianCollisions(dt);
  checkTrafficRuleViolation(dt);
}

function checkPedestrianCollisions(dt){
  if(pedestrianCollisionCooldown > 0){
    pedestrianCollisionCooldown -= dt;
    return;
  }

  for(const ped of world.pedestrians){
    const d = Math.hypot(mainAgent.x - ped.position.x, mainAgent.z - ped.position.z);
    if(d < 0.52){
      pedestrianHits += 1;
      pedestrianCollisionCooldown = 1.2;
      lastReward = -12;
      cumulativeReward += lastReward;
      statusText.textContent = "Pedestrian collision penalty!";
      emitSimEvent("collision", { kind:"pedestrian", penalty:lastReward, x:mainAgent.x, z:mainAgent.z });
      mainAgent.vx *= -0.15;
      mainAgent.vz *= -0.15;
      break;
    }
  }
}

function checkTrafficRuleViolation(dt){
  if(trafficViolationCooldown > 0){
    trafficViolationCooldown -= dt;
    return;
  }

  const speed = Math.hypot(mainAgent.vx, mainAgent.vz);
  if(speed < 0.35) return;

  const signal = getSignalPhase();
  const movingVertical = Math.abs(mainAgent.vz) > Math.abs(mainAgent.vx);
  const direction = movingVertical ? "vertical" : "horizontal";

  for(const n of world.graphNodes){
    const near = Math.hypot(mainAgent.x - n.x, mainAgent.z - n.z) < 1.15;
    if(!near) continue;

    const legal = direction === "vertical" ? signal.vertical === "green" : signal.horizontal === "green";
    if(!legal){
      trafficViolations += 1;
      trafficViolationCooldown = 1.8;
      lastReward = -8;
      cumulativeReward += lastReward;
      statusText.textContent = "Traffic-rule violation penalty!";
      emitSimEvent("traffic-violation", { direction, signal:signal.label, penalty:lastReward, x:mainAgent.x, z:mainAgent.z });
      break;
    }
  }
}

function updateTask(dt){
  taskTimer += dt;
  let reward = -0.08*dt - 0.012*dt*(100-mainAgent.energy)/100; // time + energy usage

  // recharge
  const station = world.destinations["station"];
  if(Math.hypot(mainAgent.x-station.x, mainAgent.z-station.z) < 1.1){
    mainAgent.energy = Math.min(100, mainAgent.energy + dt*18);
    if(mainAgent.energy > 95) stationText.textContent = "Rested";
    else stationText.textContent = "Resting";

    if(currentTask?.kind === "charge" && mainAgent.energy > 94){
      reward += 12;
      finishTask(reward);
      return;
    }
  } else {
    stationText.textContent = "Available";
  }

  if(currentTarget && Math.hypot(mainAgent.x-currentTarget.x, mainAgent.z-currentTarget.z) < (currentTarget.reachRadius || 1.0)){
    if(currentTask.kind === "home" || currentTask.kind === "grocery" || currentTask.kind === "icecream"){
      reward += 25;
      finishTask(reward);
      return;
    }
    if(currentTask.kind === "delivery"){
      if(currentTask.phase === "pickup"){
        mainAgent.carrying = true;
        packageText.textContent = "Package";
        reward += 10;
        currentTask.phase = "dropoff";
        setCurrentTarget(world.destinations[currentTask.targets[1]]);
        drawTargetBeacon();
        statusText.textContent = "Package picked up";
        emitSimEvent("pickup", { target:currentTarget.label, reward });
        if(autoMode){
          mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
          drawPolicyPath(mainAgent.path, 0x2563eb, true);
        }
      } else {
        reward += 30;
        deliveriesCompleted += 1;
        finishTask(reward);
        return;
      }
    }
    if(currentTask.kind === "multidelivery"){
      if(currentTask.phase === "pickup"){
        mainAgent.carrying = true;
        const totalPackages = currentTask.targets.length - 1;
        packageText.textContent = `${totalPackages} Packages`;
        reward += 10;
        currentTask.phase = "deliver";
        currentTask.currentIndex = 1;
        setCurrentTarget(world.destinations[currentTask.targets[1]]);
        drawTargetBeacon();
        statusText.textContent = "Packages picked up";
        if(autoMode){
          mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
          drawPolicyPath(mainAgent.path, 0x2563eb, true);
        }
      } else {
        reward += 20;
        deliveriesCompleted += 1;
        currentTask.currentIndex += 1;
        const remaining = currentTask.targets.length - currentTask.currentIndex;
        if(remaining > 0){
          packageText.textContent = remaining === 1 ? "1 Package" : `${remaining} Packages`;
          setCurrentTarget(world.destinations[currentTask.targets[currentTask.currentIndex]]);
          drawTargetBeacon();
          statusText.textContent = "Delivery completed";
          emitSimEvent("delivery", { target:currentTarget.label, reward });
          if(autoMode){
            mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
            drawPolicyPath(mainAgent.path, 0x2563eb, true);
          }
        } else {
          reward += 10;
          finishTask(reward);
          return;
        }
      }
    }
  }

  lastReward = reward;
  cumulativeReward += reward;
}

function finishTask(rewardBonus){
  lastReward = rewardBonus;
  cumulativeReward += rewardBonus;
  score += 1;
  dailyChoresCompleted += 1;
  bestTime = bestTime === null ? taskTimer : Math.min(bestTime, taskTimer);
  statusText.textContent = "Chore completed";
  mainAgent.carrying = false;
  packageText.textContent = "None";
  emitSimEvent("goal", { target:currentTarget?.label || null, reward:rewardBonus, chore:currentChoreKey });

  if(learningMode){
    updateChoreValue();
    chooseAndStartNextChore();
  } else {
    autoMode = false;
    mainAgent.path = [];
    autoBtn.textContent = "Autonomous Mode";
    clearGroup(pathGroup);
  }
}

function updateChoreValue(){
  const choreReturn = cumulativeReward - taskStartReturn;
  qCounts[currentChoreKey] = (qCounts[currentChoreKey] || 0) + 1;

  const alpha = 0.22;
  qValues[currentChoreKey] = (1 - alpha) * (qValues[currentChoreKey] || 0) + alpha * choreReturn;

  // Slow decay: the agent explores a lot early, then gradually exploits better routines.
  epsilon = Math.max(0.06, 0.45 * Math.pow(0.985, score));

  emitSimEvent("learn-update", {
    chore: currentChoreKey,
    choreReturn,
    qValue: qValues[currentChoreKey],
    epsilon
  });
}

function chooseChore(){
  if(mainAgent.energy < 28){
    return "charge";
  }

  // Late evening: prefer going home, but not deterministically.
  if(simulatedMinutes > 21 * 60 && Math.random() < 0.70){
    return "home";
  }

  const candidates = ["home", "grocery", "icecream", "delivery", "multidelivery"];

  if(Math.random() < epsilon){
    return candidates[Math.floor(Math.random() * candidates.length)];
  }

  let best = candidates[0];
  let bestValue = -Infinity;

  for(const c of candidates){
    const explorationBonus = 1.5 / Math.sqrt(1 + (qCounts[c] || 0));
    const value = (qValues[c] || 0) + explorationBonus;
    if(value > bestValue){
      bestValue = value;
      best = c;
    }
  }

  return best;
}

function chooseAndStartNextChore(){
  if(!learningMode) return;

  const next = chooseChore();
  startTask(next);
}

function updateRewardHistory(dt){
  rewardTickAcc += dt;
  if(rewardTickAcc >= .15){
    rewardTickAcc = 0;
    rewardHistory.push(lastReward);
    returnHistory.push(cumulativeReward);
    if(rewardHistory.length > 240) rewardHistory.shift();
    if(returnHistory.length > 240) returnHistory.shift();
  }
}

function drawRewardGraph(){
  const ctx = rewardCtx, w = rewardCanvas.width, h = rewardCanvas.height;
  ctx.clearRect(0,0,w,h);
  ctx.fillStyle = "#f8fafc"; ctx.fillRect(0,0,w,h);

  ctx.strokeStyle = "rgba(148,163,184,.25)";
  ctx.lineWidth = 1;
  for(let i=1;i<5;i++){
    const y = (h/5)*i;
    ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(w,y); ctx.stroke();
  }

  const series = returnHistory.length ? returnHistory : [0];
  let minV = Math.min(...series, -25);
  let maxV = Math.max(...series, 25);
  if(maxV-minV < 10){ maxV += 5; minV -= 5; }
  const mapY = v => h - 20 - (v-minV)/(maxV-minV)*(h-40);

  if(minV < 0 && maxV > 0){
    const y0 = mapY(0);
    ctx.strokeStyle = "rgba(100,116,139,.35)";
    ctx.beginPath(); ctx.moveTo(0,y0); ctx.lineTo(w,y0); ctx.stroke();
  }

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  for(let i=0;i<series.length;i++){
    const x = 20 + (w-40)*(i/Math.max(1,series.length-1));
    const y = mapY(series[i]);
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  }
  ctx.stroke();

  for(let i=0;i<rewardHistory.length;i++){
    const x = 20 + (w-40)*(i/Math.max(1,rewardHistory.length-1));
    const zeroY = h - 18;
    const barH = Math.max(-42, Math.min(42, rewardHistory[i]/18*35));
    ctx.strokeStyle = rewardHistory[i] >= 0 ? "rgba(22,163,74,.55)" : "rgba(220,38,38,.55)";
    ctx.beginPath(); ctx.moveTo(x, zeroY); ctx.lineTo(x, zeroY - barH); ctx.stroke();
  }

  ctx.fillStyle = "#475569";
  ctx.font = "12px system-ui, sans-serif";
  ctx.fillText("Cumulative return", 20, 16);
  ctx.fillStyle = "#2563eb";
  ctx.fillRect(130, 8, 18, 4);
  ctx.fillStyle = "#475569";
  ctx.fillText("Instant reward bars", 165, 16);
}

function drawMinimap(){
  const ctx = minimapCtx, W = minimapCanvas.width, H = minimapCanvas.height;
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = isNight ? "#203040" : "#eef4fb";
  ctx.fillRect(0,0,W,H);

  const mapX = x => (x+MAP.worldHalf)/(2*MAP.worldHalf)*W;
  const mapZ = z => H - (z+MAP.worldHalf)/(2*MAP.worldHalf)*H;

  ctx.fillStyle = isNight ? "rgba(107,174,214,.70)" : "rgba(96,165,250,.45)";
  const riverY1 = mapZ(MAP.riverMaxZ), riverY2 = mapZ(MAP.riverMinZ);
  ctx.fillRect(0, riverY1, W, riverY2-riverY1);

  ctx.fillStyle = isNight ? "rgba(95,160,115,.35)" : "rgba(95,160,115,.45)";
  const parkX1 = mapX(7), parkX2 = mapX(33), parkZ1 = mapZ(31), parkZ2 = mapZ(13);
  ctx.fillRect(parkX1, parkZ1, parkX2-parkX1, parkZ2-parkZ1);

  ctx.strokeStyle = isNight ? "rgba(255,255,255,.70)" : "rgba(71,85,105,.75)";
  ctx.lineWidth = 2;
  for(const r of MAP.verticalRoads){
    ctx.beginPath(); ctx.moveTo(mapX(r.x), mapZ(-MAP.worldHalf)); ctx.lineTo(mapX(r.x), mapZ(MAP.worldHalf)); ctx.stroke();
  }
  for(const r of MAP.horizontalRoads){
    ctx.beginPath(); ctx.moveTo(mapX(-MAP.worldHalf), mapZ(r.z)); ctx.lineTo(mapX(MAP.worldHalf), mapZ(r.z)); ctx.stroke();
  }

  const mark = (color, x, z, s=4) => {
    ctx.fillStyle = color;
    ctx.beginPath(); ctx.arc(mapX(x), mapZ(z), s, 0, Math.PI*2); ctx.fill();
  };

  mark("#22c55e", world.destinations.grocery.x, world.destinations.grocery.z);
  mark("#ef4444", world.destinations.warehouse.x, world.destinations.warehouse.z);
  mark("#60a5fa", world.destinations.icecream.x, world.destinations.icecream.z);
  mark("#f59e0b", world.destinations.cafe.x, world.destinations.cafe.z);
  mark("#1d4ed8", world.destinations.station.x, world.destinations.station.z);
  mark("#ffffff", world.destinations.home1.x, world.destinations.home1.z, 3.6);
  mark("#ffffff", world.destinations.home2.x, world.destinations.home2.z, 3.6);
  mark("#ffffff", world.destinations.home3.x, world.destinations.home3.z, 3.6);

  ctx.strokeStyle = "#d97706";
  ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(mapX(currentTarget.x), mapZ(currentTarget.z), 7, 0, Math.PI*2); ctx.stroke();

  agents.forEach((a,i) => {
    const colors = ["#2563eb", "#16a34a", "#d97706"];
    mark(colors[i], a.x, a.z, i===0 ? 5 : 4);
  });

  drawMinimapLegend(ctx, W, H);
}

function drawMinimapLegend(ctx, W, H){
  const items = [
    ["#2563eb", "You"],
    ["#d97706", "Target"],
    ["#22c55e", "Grocery"],
    ["#ef4444", "Warehouse"],
    ["#60a5fa", "Ice cream"],
    ["#1d4ed8", "Resting"]
  ];

  const x0 = 8;
  const y0 = H - 48;

  ctx.fillStyle = isNight ? "rgba(15,23,42,.72)" : "rgba(255,255,255,.78)";
  ctx.fillRect(5, y0 - 7, W - 10, 44);

  ctx.font = "9px system-ui, sans-serif";
  ctx.textBaseline = "middle";

  items.forEach((item, i) => {
    const col = i % 3;
    const row = Math.floor(i / 3);
    const x = x0 + col * 76;
    const y = y0 + row * 18;

    ctx.fillStyle = item[0];
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = isNight ? "#e2e8f0" : "#334155";
    ctx.fillText(item[1], x + 8, y);
  });
}


function applyWeatherVisuals(){
  if(weatherMode === "fog"){
    scene.fog = new THREE.Fog(isNight ? 0x6f7f8f : 0xdde7f0, 8, 72);
    hemi.intensity = isNight ? 0.62 : 1.18;
    sun.intensity = isNight ? 0.34 : 0.72;
  } else if(weatherMode === "rain"){
    scene.fog = new THREE.Fog(isNight ? 0x243244 : 0xb7c4cf, 18, 115);
    hemi.intensity = isNight ? 0.58 : 1.05;
    sun.intensity = isNight ? 0.28 : 0.62;
  } else if(weatherMode === "snow"){
    // Night snow should not use a near-white fog color. A darker blue-gray fog
    // makes the flakes read as depth cues instead of bright visual noise.
    scene.fog = new THREE.Fog(isNight ? 0x344455 : 0xe6edf5, isNight ? 18 : 20, isNight ? 125 : 100);
    hemi.intensity = isNight ? 0.66 : 1.35;
    sun.intensity = isNight ? 0.36 : 0.80;
  } else {
    scene.fog = new THREE.Fog(isNight ? 0x1c2b3d : 0xe6eef6, isNight ? 28 : 48, isNight ? 155 : 185);
    hemi.intensity = isNight ? 0.75 : 1.55;
    sun.intensity = isNight ? 0.55 : 1.25;
  }
  weatherText.textContent = weatherMode[0].toUpperCase() + weatherMode.slice(1);
}

function buildWeatherSystem(){
  clearGroup(weatherGroup);
  world.weatherParticles = null;
  world.weatherVelocities = [];

  if(weatherMode === "clear" || weatherMode === "fog"){
    return;
  }

  const snowAtNight = weatherMode === "snow" && isNight;
  const count = weatherMode === "rain" ? 1500 : (snowAtNight ? 620 : 900);
  const positions = new Float32Array(count * 3);

  for(let i=0;i<count;i++){
    positions[i*3] = rand(-MAP.worldHalf, MAP.worldHalf);
    positions[i*3+1] = rand(snowAtNight ? 10 : 8, snowAtNight ? 48 : 55);
    positions[i*3+2] = rand(-MAP.worldHalf, MAP.worldHalf);

    if(weatherMode === "rain"){
      world.weatherVelocities.push({ y: rand(18, 30), x: rand(-1.2, -0.2), z: rand(0.2, 1.2), phase: rand(0, 6.28), amp: 0 });
    } else {
      // Snow should drift slowly with small lateral variation. At night we use
      // fewer, softer flakes so they do not look like a white star field.
      world.weatherVelocities.push({
        y: rand(snowAtNight ? 0.55 : 0.85, snowAtNight ? 1.45 : 2.25),
        x: rand(snowAtNight ? -0.22 : -0.42, snowAtNight ? 0.22 : 0.42),
        z: rand(snowAtNight ? -0.22 : -0.42, snowAtNight ? 0.22 : 0.42),
        phase: rand(0, 6.28),
        amp: rand(snowAtNight ? 0.04 : 0.06, snowAtNight ? 0.12 : 0.18)
      });
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));

  const mat = new THREE.PointsMaterial({
    color: weatherMode === "rain" ? 0xbfd7ef : (snowAtNight ? 0xd8e2ed : 0xffffff),
    size: weatherMode === "rain" ? 0.075 : (snowAtNight ? 0.105 : 0.155),
    transparent: true,
    opacity: weatherMode === "rain" ? 0.55 : (snowAtNight ? 0.50 : 0.78),
    depthWrite: false,
    sizeAttenuation: true
  });

  const particles = new THREE.Points(geo, mat);
  world.weatherParticles = particles;
  weatherGroup.add(particles);

  if(snowAtNight){
    addSubtleSnowAccumulation();
  }
}

function updateWeather(dt){
  if(!world.weatherParticles) return;

  const pos = world.weatherParticles.geometry.attributes.position.array;
  const snow = weatherMode === "snow";

  for(let i=0;i<world.weatherVelocities.length;i++){
    const v = world.weatherVelocities[i];

    const drift = snow ? Math.sin(clock.elapsedTime * 0.8 + v.phase) * v.amp : 0;
    pos[i*3] += (v.x + drift) * dt;
    pos[i*3+1] -= v.y * dt;
    pos[i*3+2] += (v.z + drift * 0.6) * dt;

    if(pos[i*3+1] < 0.2){
      pos[i*3] = rand(mainAgent.x - 48, mainAgent.x + 48);
      pos[i*3+1] = rand(snow ? 22 : 24, snow ? 50 : 58);
      pos[i*3+2] = rand(mainAgent.z - 48, mainAgent.z + 48);
    }

    if(pos[i*3] < -MAP.worldHalf || pos[i*3] > MAP.worldHalf) pos[i*3] = rand(-MAP.worldHalf, MAP.worldHalf);
    if(pos[i*3+2] < -MAP.worldHalf || pos[i*3+2] > MAP.worldHalf) pos[i*3+2] = rand(-MAP.worldHalf, MAP.worldHalf);
  }

  world.weatherParticles.geometry.attributes.position.needsUpdate = true;
}

function addSubtleSnowAccumulation(){
  const mat = new THREE.MeshBasicMaterial({
    color: 0xe7eef6,
    transparent: true,
    opacity: 0.10,
    depthWrite: false
  });

  const patchCenters = [
    [20, 22, 27, 19],
    [36, -26, 8, 7],
    [50, -14, 8, 7],
    [32, -42, 8, 7],
    [-22, -30, 9, 7],
    [34, 34, 9, 7],
    [-40, 16, 9, 7]
  ];

  for(const [x, z, sx, sz] of patchCenters){
    const patch = new THREE.Mesh(new THREE.PlaneGeometry(sx, sz), mat.clone());
    patch.rotation.x = -Math.PI / 2;
    patch.position.set(x, 0.115, z);
    weatherGroup.add(patch);
  }
}


function updateSimulatedClock(dt){
  const minutesPerSecond = (24 * 60) / dayLengthSeconds;
  simulatedMinutes += dt * minutesPerSecond;

  if(simulatedMinutes >= 24 * 60){
    resetDailyExperience(true);
    return;
  }

  updateDayNightFromClock(false);
}

function updateDayNightFromClock(force=false){
  const shouldBeNight = simulatedMinutes < 6 * 60 || simulatedMinutes >= 19 * 60;

  if(force || shouldBeNight !== lastClockNightState){
    lastClockNightState = shouldBeNight;
    isNight = shouldBeNight;
    applyEnvironmentTheme();

    // Keep night lights tied to the clock, regardless of weather.
    updateCityNightLights();

    // Rebuild only weather particles / overlays, not the whole city.
    buildWeatherSystem();

    emitSimEvent("day-night-change", {
      day:simulatedDay,
      clock:formatClock(simulatedMinutes),
      isNight
    });
  }
}

function formatClock(totalMinutes){
  const mins = Math.floor(totalMinutes) % (24 * 60);
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return String(h).padStart(2, "0") + ":" + String(m).padStart(2, "0");
}

function randomizeDailyWeather(){
  const options = ["clear", "clear", "clear", "rain", "fog", "snow"];
  weatherMode = options[Math.floor(Math.random() * options.length)];
  weatherSelect.value = weatherMode;
  applyWeatherVisuals();
  buildWeatherSystem();
}

function bestLearnedChore(){
  let best = choreKeys[0];
  let bestVal = -Infinity;
  for(const c of choreKeys){
    if((qCounts[c] || 0) === 0) continue;
    if(qValues[c] > bestVal){
      bestVal = qValues[c];
      best = c;
    }
  }
  if(bestVal === -Infinity) return "learning...";
  return best + " (" + bestVal.toFixed(1) + ")";
}

function updateStats(){
  modeText.textContent = autoMode ? "Demo Agent" : "Manual";
  distanceText.textContent = currentTarget ? Math.hypot(mainAgent.x-currentTarget.x, mainAgent.z-currentTarget.z).toFixed(1) : "0.0";
  timerText.textContent = taskTimer.toFixed(1) + "s";
  scoreText.textContent = String(dailyChoresCompleted);
  bestTimeText.textContent = bestTime === null ? "—" : bestTime.toFixed(1) + "s";
  deliveryText.textContent = String(deliveriesCompleted);
  neighborhoodText.textContent = humanNeighborhood(mainAgent.x, mainAgent.z);
  lastRewardText.textContent = lastReward.toFixed(2);
  simClockText.textContent = formatClock(simulatedMinutes);
  dayText.textContent = String(simulatedDay);
  epsilonText.textContent = epsilon.toFixed(2);
  bestChoreText.textContent = bestLearnedChore();
  returnText.textContent = cumulativeReward.toFixed(1);
  batteryText.textContent = mainAgent.energy.toFixed(0) + "%";
  weatherText.textContent = weatherMode[0].toUpperCase() + weatherMode.slice(1);
  violationText.textContent = String(trafficViolations);
  pedHitText.textContent = String(pedestrianHits);
  collisionText.textContent = String(collisions);
}
function humanNeighborhood(x,z){
  if(z > MAP.riverMaxZ+2 && x > -4) return "Downtown";
  if(x < -12 && z > MAP.riverMaxZ+2) return "Riverside";
  if(z < -8 && x > 18) return "Residential";
  if(z < -8 && x < 8) return "Market";
  return "Mixed";
}

function updateCamera(){
  let tx = mainAgent.x, tz = mainAgent.z;
  if(cameraMode === "orbit" || cameraMode === "skyline"){
    tx = cameraRig.targetX;
    tz = cameraRig.targetZ;
  }

  if(cameraMode === "top"){
    camera.position.x += (tx - camera.position.x)*.05;
    camera.position.y += (85 - camera.position.y)*.05;
    camera.position.z += ((tz+.01) - camera.position.z)*.05;
    camera.lookAt(tx,0,tz);
    return;
  }

  if(cameraMode === "skyline"){
    const dist = Math.max(cameraRig.distance, 118);
    const h = Math.max(cameraRig.height, 66);
    const cx = tx + Math.sin(cameraRig.theta)*dist;
    const cz = tz + Math.cos(cameraRig.theta)*dist;
    camera.position.x += (cx-camera.position.x)*.032;
    camera.position.y += (h-camera.position.y)*.032;
    camera.position.z += (cz-camera.position.z)*.032;
    camera.lookAt(tx,8,tz);
    return;
  }

  const cx = tx + Math.sin(cameraRig.theta)*cameraRig.distance;
  const cz = tz + Math.cos(cameraRig.theta)*cameraRig.distance;
  camera.position.x += (cx-camera.position.x)*.04;
  camera.position.y += (cameraRig.height-camera.position.y)*.04;
  camera.position.z += (cz-camera.position.z)*.04;
  camera.lookAt(tx,.55,tz);
}
function resetCamera(){
  cameraRig.theta = Math.PI/4;
  cameraRig.distance = 70;
  cameraRig.height = 40;
  cameraRig.targetX = mainAgent.x;
  cameraRig.targetZ = mainAgent.z;
}

function animate(){
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), .033);

  updateSimulatedClock(dt);
  updateSignals();
  updateWeather(dt);
  updateVehicles(dt);
  updatePedestrians(dt);
  updateMainAgent(dt);
  updateSecondaryAgents(dt);
  checkCollisions(dt);
  updateTask(dt);
  updateRewardHistory(dt);
  animateBeacon();
  animateClickMarker();
  updateCamera();
  updateStats();
  drawRewardGraph();
  drawMinimap();

  renderer.render(scene, camera);
}

function resize(){
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w,h,false);
  camera.aspect = w/h;
  camera.updateProjectionMatrix();
}

function onPointerDown(e){
  cameraRig.dragging = true;
  cameraRig.dragMoved = false;
  cameraRig.lastX = e.clientX;
  cameraRig.lastY = e.clientY;
  renderer.domElement.setPointerCapture?.(e.pointerId);
}
function onPointerMove(e){
  if(!cameraRig.dragging) return;
  const dx = e.clientX-cameraRig.lastX;
  const dy = e.clientY-cameraRig.lastY;
  if(Math.abs(dx)+Math.abs(dy) > 3) cameraRig.dragMoved = true;
  cameraRig.lastX = e.clientX;
  cameraRig.lastY = e.clientY;

  if(e.shiftKey || e.buttons === 4){
    const panScale = cameraRig.distance/700;
    cameraRig.targetX -= dx*panScale*Math.cos(cameraRig.theta);
    cameraRig.targetZ += dx*panScale*Math.sin(cameraRig.theta);
    cameraRig.targetX -= dy*panScale*Math.sin(cameraRig.theta);
    cameraRig.targetZ -= dy*panScale*Math.cos(cameraRig.theta);
    cameraRig.targetX = clamp(cameraRig.targetX, -MAP.worldHalf, MAP.worldHalf);
    cameraRig.targetZ = clamp(cameraRig.targetZ, -MAP.worldHalf, MAP.worldHalf);
    cameraMode = "orbit";
    cameraSelect.value = "orbit";
    return;
  }

  cameraRig.theta -= dx*.006;
  cameraRig.height = clamp(cameraRig.height + dy*.05, 14, 130);
}
function onPointerUp(e){
  renderer.domElement.releasePointerCapture?.(e.pointerId);
  const wasDrag = cameraRig.dragMoved;
  cameraRig.dragging = false;
  if(!wasDrag) setMainTargetFromPointer(e);
}
function onWheel(e){
  e.preventDefault();
  cameraRig.distance = clamp(cameraRig.distance + e.deltaY*.06, 14, 190);
  if(cameraMode === "top") cameraRig.height = clamp(cameraRig.height + e.deltaY*.08, 30, 145);
  if(cameraMode === "skyline") cameraRig.height = clamp(cameraRig.height + e.deltaY*.05, 40, 145);
}

renderer.domElement.addEventListener("pointerdown", onPointerDown);
renderer.domElement.addEventListener("pointermove", onPointerMove);
renderer.domElement.addEventListener("pointerup", onPointerUp);
renderer.domElement.addEventListener("pointercancel", ()=>cameraRig.dragging=false);
renderer.domElement.addEventListener("wheel", onWheel, { passive:false });

taskBtn.addEventListener("click", ()=>{
  learningMode = false;
  autoMode = true;
  learnModeBtn.textContent = "Resume Learning";
  startTask(taskSelect.value);
});
autoBtn.addEventListener("click", ()=>{
  autoMode = !autoMode;
  if(autoMode){
    manualTarget = null;
    clickMarker.visible = false;
    mainAgent.path = planPath(mainAgent.x, mainAgent.z, currentTarget.x, currentTarget.z);
    drawPolicyPath(mainAgent.path, 0x2563eb, true);
    autoBtn.textContent = learningMode ? "Learning Autonomously" : "Back to Manual";
  } else {
    mainAgent.path = [];
    clearGroup(pathGroup);
    autoBtn.textContent = "Autonomous Mode";
  }
});

learnModeBtn.addEventListener("click", () => {
  learningMode = !learningMode;

  if(learningMode){
    autoMode = true;
    learnModeBtn.textContent = "Pause Learning";
    chooseAndStartNextChore();
  } else {
    learnModeBtn.textContent = "Resume Learning";
    autoMode = false;
    mainAgent.path = [];
    clearGroup(pathGroup);
    statusText.textContent = "Learning paused";
  }
});

clockSpeedSelect.addEventListener("change", () => {
  dayLengthSeconds = Number(clockSpeedSelect.value);
});
timeBtn.addEventListener("click", ()=>{
  if(isNight){
    simulatedMinutes = 12 * 60;
  } else {
    simulatedMinutes = 21 * 60;
  }
  updateDayNightFromClock(true);
});
weatherSelect.addEventListener("change", () => {
  weatherMode = weatherSelect.value;
  applyEnvironmentTheme();
  updateCityNightLights();
  buildWeatherSystem();
  emitSimEvent("weather-change", { weather: weatherMode });
});

cameraBtn.addEventListener("click", ()=>{
  cameraMode = "follow";
  cameraSelect.value = "follow";
  resetCamera();
});
episodeBtn.addEventListener("click", ()=>{
  resetDailyExperience(true);
});
resetBtn.addEventListener("click", resetMainPosition);
cameraSelect.addEventListener("change", ()=>{
  cameraMode = cameraSelect.value;
  if(cameraMode === "orbit"){
    cameraRig.targetX = mainAgent.x;
    cameraRig.targetZ = mainAgent.z;
  }
  if(cameraMode === "skyline"){
    cameraRig.targetX = 0;
    cameraRig.targetZ = 0;
    cameraRig.distance = Math.max(cameraRig.distance, 118);
    cameraRig.height = Math.max(cameraRig.height, 66);
  }
  if(cameraMode === "top"){
    cameraRig.targetX = mainAgent.x;
    cameraRig.targetZ = mainAgent.z;
  }
});
window.addEventListener("resize", resize);

function rand(a,b){ return a + Math.random()*(b-a); }
function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function lerp(a,b,t){ return a + (b-a)*t; }
function darken(hex, amount){
  const c = new THREE.Color(hex);
  c.multiplyScalar(1-amount);
  return c.getHex();
}
</script>
</body>
</html>
