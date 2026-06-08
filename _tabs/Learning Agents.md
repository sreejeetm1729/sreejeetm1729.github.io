---
title: Learning Agents
icon: fas fa-robot
order: 8
---

<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>3D Drone RL Playground</title>

  <style>
    body {
      margin: 0;
      background: #020617;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #f8fafc;
    }

    #rl3d-drone-widget {
      max-width: 1120px;
      margin: 24px auto;
      padding: 1.4rem;
      border-radius: 22px;
      background: linear-gradient(135deg, #0f172a, #1e293b);
      color: #f8fafc;
      box-shadow: 0 18px 45px rgba(0, 0, 0, 0.28);
      font-family: inherit;
    }

    #rl3d-drone-widget * {
      box-sizing: border-box;
    }

    .rl3d-header h2 {
      margin: 0 0 0.45rem 0;
      color: #ffffff;
      font-size: 1.55rem;
    }

    .rl3d-header p {
      margin: 0 0 1.1rem 0;
      color: #cbd5e1;
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
      color: #cbd5e1;
      font-size: 0.85rem;
      font-weight: 800;
    }

    .rl3d-toolbar select {
      min-width: 220px;
      border: 1px solid rgba(255, 255, 255, 0.16);
      border-radius: 999px;
      padding: 0.65rem 0.85rem;
      color: #0f172a;
      background: #e0f2fe;
      font-weight: 800;
      outline: none;
    }

    #rl3d-stage-wrap {
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
      color: #ffffff;
      font-weight: 900;
      font-size: 0.98rem;
      margin-bottom: 0.2rem;
    }

    .rl3d-task-desc {
      color: #cbd5e1;
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
      background: rgba(255, 255, 255, 0.07);
      border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .rl3d-stats span {
      display: block;
      color: #cbd5e1;
      font-size: 0.82rem;
      margin-bottom: 0.25rem;
    }

    .rl3d-stats strong {
      color: #ffffff;
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
</head>

<body>
  <div id="rl3d-drone-widget">
    <div class="rl3d-header">
      <h2>3D Drone RL Playground</h2>
      <p>
        Move your mouse over the arena to control the drone. Try different environments and tasks:
        hover, deliver cargo, or visit waypoints. The drone receives reward for completing the task
        while compensating for wind and inertia.
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
</body>
</html>
