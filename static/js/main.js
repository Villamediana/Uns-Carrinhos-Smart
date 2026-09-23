// Main simulation loop and visualization.

const trackCanvas = document.getElementById("track-canvas");
const netCanvas = document.getElementById("net-canvas");
const chartCanvas = document.getElementById("chart-canvas");
const tctx = trackCanvas.getContext("2d");
const nctx = netCanvas.getContext("2d");
const cctx = chartCanvas ? chartCanvas.getContext("2d") : null;

const statGen = document.getElementById("stat-gen");
const statAlive = document.getElementById("stat-alive");
const statBest = document.getElementById("stat-best");
const statAvg = document.getElementById("stat-avg");
const statTimer = document.getElementById("stat-timer");
const netSub = document.getElementById("net-sub");
const btnTrain = document.getElementById("btn-train");
const toast = document.getElementById("toast");
const toastGens = document.getElementById("toast-gens");
const toastBest = document.getElementById("toast-best");
const toastTime = document.getElementById("toast-time");

const btnPause = document.getElementById("btn-pause");
const btnReset = document.getElementById("btn-reset");
const speedSlider = document.getElementById("speed");
const popSlider = document.getElementById("pop");
const mutSlider = document.getElementById("mut");
const trackPicker = document.getElementById("track-picker");

// hi-dpi
function fitCanvas(canvas, yOffset = 0) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width;
  const h = canvas.height;
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  canvas.width = Math.floor(w * dpr);
  canvas.height = Math.floor(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  if (yOffset) ctx.translate(0, yOffset);
  return { w, h };
}
// Track canvas is taller than the actual world (720x520) so it visually
// aligns with the right column; we center the world drawing vertically.
const TRACK_WORLD_H = 520;
const TRACK_Y_OFFSET = (trackCanvas.height - TRACK_WORLD_H) / 2;
const trackDim = fitCanvas(trackCanvas, TRACK_Y_OFFSET);
const netDim = fitCanvas(netCanvas);
const chartDim = chartCanvas ? fitCanvas(chartCanvas) : null;

window.SKIDS = [];
const SKID_MAX = 1500;
const SKID_LIFE = 240; // frames until fully faded

const state = {
  trackIdx: 0,
  population: [],
  bestBrain: null,
  bestFitness: -Infinity,
  gen: 1,
  paused: false,
  speed: 2,
  popSize: 40,
  mutRate: 0.08,
  historicalBest: 0,
  curriculum: false,
  curriculumStreak: 0,
  genFrames: 0,
  history: [], // [{gen, best, avg}]
};

const CURRICULUM_LAPS = 2;
const CURRICULUM_STREAK = 3;
const HISTORY_MAX = 400;

const MAX_GEN_FRAMES = 5000;

function buildTrackPicker() {
  for (let i = 0; i < TRACKS.length; i++) {
    const b = document.createElement("button");
    b.textContent = String(i + 1);
    if (i === state.trackIdx) b.classList.add("active");
    b.addEventListener("click", () => {
      switchTrack(i);
      state.curriculum = false;
      state.curriculumStreak = 0;
      saveSettings();
      updateTrackButtons();
    });
    trackPicker.appendChild(b);
  }
  const curr = document.createElement("button");
  curr.textContent = "auto";
  curr.title = "curriculum: avança pra próxima pista quando dominar a atual";
  curr.addEventListener("click", () => {
    state.curriculum = !state.curriculum;
    state.curriculumStreak = 0;
    saveSettings();
    updateTrackButtons();
  });
  curr.id = "track-curriculum";
  trackPicker.appendChild(curr);
  updateTrackButtons();
}

function switchTrack(newIdx) {
  if (newIdx === state.trackIdx && state.population.length) return;
  state.trackIdx = newIdx;
  window.SKIDS.length = 0;
  const track = TRACKS[newIdx];
  // carry over the current population brains, unchanged
  if (state.population.length) {
    state.population = state.population.map((c) => new Car(c.brain, track));
  } else {
    state.population = seedPopulation(state.popSize);
  }
}

function updateTrackButtons() {
  const kids = [...trackPicker.children];
  for (let k = 0; k < kids.length - 1; k++) {
    kids[k].classList.toggle("active", k === state.trackIdx);
  }
  const c = document.getElementById("track-curriculum");
  if (c) c.classList.toggle("active", state.curriculum);
}

function seedPopulation(size) {
  const track = TRACKS[state.trackIdx];
  const pop = [];
  const useBest = !!state.bestBrain;
  for (let i = 0; i < size; i++) {
    let brain;
    if (useBest) {
      brain = state.bestBrain.clone();
      if (i > 0) brain.mutate(state.mutRate, 0.35);
    } else {
      brain = new NeuralNet();
    }
    pop.push(new Car(brain, track));
  }
  return pop;
}

function nextGeneration() {
  // Sort by fitness
  state.population.sort((a, b) => b.fitness - a.fitness);
  const survivors = state.population.slice(0, Math.max(4, Math.floor(state.popSize * 0.2)));
  const best = survivors[0];
  state.bestBrain = best.brain.clone();
  state.bestFitness = best.fitness;
  state.historicalBest = Math.max(state.historicalBest, best.fitness);

  // record history for the learning curve
  const avgFit = state.population.reduce((s, c) => s + Math.max(0, c.fitness), 0) / Math.max(1, state.population.length);
  state.history.push({ gen: state.gen, best: best.fitness, avg: avgFit });
  if (state.history.length > HISTORY_MAX) state.history.splice(0, state.history.length - HISTORY_MAX);

  // curriculum: if leader mastered current track, advance
  if (state.curriculum) {
    if (best.laps >= CURRICULUM_LAPS) state.curriculumStreak++;
    else state.curriculumStreak = 0;
    if (state.curriculumStreak >= CURRICULUM_STREAK && state.trackIdx < TRACKS.length - 1) {
      state.trackIdx++;
      state.curriculumStreak = 0;
      window.SKIDS.length = 0;
      updateTrackButtons();
    }
  }
  const track = TRACKS[state.trackIdx];
  const next = [];
  // elitism: keep best unmutated
  next.push(new Car(state.bestBrain.clone(), track));
  while (next.length < state.popSize) {
    const a = survivors[Math.floor(Math.random() * survivors.length)].brain;
    const b = survivors[Math.floor(Math.random() * survivors.length)].brain;
    const child = NeuralNet.crossover(a, b);
    child.mutate(state.mutRate, 0.35);
    next.push(new Car(child, track));
  }
  state.population = next;
  state.gen++;
  state.genFrames = 0;
  saveBrain();
}

function resetGeneration(hard = false) {
  if (hard) {
    state.bestBrain = null;
    state.bestFitness = -Infinity;
    state.gen = 1;
    state.historicalBest = 0;
    state.history = [];
    state.curriculumStreak = 0;
    clearSavedBrain();
  }
  window.SKIDS.length = 0;
  state.genFrames = 0;
  state.population = seedPopulation(state.popSize);
}

const SETTINGS_KEY = "carrinho-nn:settings-v1";
let saveTimer = null;

function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      trackIdx: state.trackIdx,
      speed: state.speed,
      popSize: state.popSize,
      mutRate: state.mutRate,
      curriculum: state.curriculum,
    }));
  } catch (e) {}
}

function loadSettings() {
  try {
    const raw = localStorage.getItem(SETTINGS_KEY);
    if (!raw) return;
    const s = JSON.parse(raw);
    if (typeof s.trackIdx === "number" && s.trackIdx >= 0 && s.trackIdx < TRACKS.length) state.trackIdx = s.trackIdx;
    if (typeof s.speed === "number") state.speed = s.speed;
    if (typeof s.popSize === "number") state.popSize = s.popSize;
    if (typeof s.mutRate === "number") state.mutRate = s.mutRate;
    if (typeof s.curriculum === "boolean") state.curriculum = s.curriculum;
  } catch (e) {}
}

function saveBrain() {
  if (!state.bestBrain) return;
  if (bgTraining) return; // server owns the brain while background training is on
  // debounce: coalesce rapid saves into one PUT
  if (saveTimer) return;
  saveTimer = setTimeout(async () => {
    saveTimer = null;
    if (!state.bestBrain) return;
    const payload = {
      brain: state.bestBrain.serialize(),
      gen: state.gen,
      historicalBest: state.historicalBest,
      history: state.history,
    };
    try {
      await fetch("/api/brain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } catch (e) { /* offline: skip */ }
  }, 400);
}

async function loadBrain() {
  try {
    const res = await fetch("/api/brain");
    if (!res.ok) return false;
    const payload = await res.json();
    if (!payload.present) return false;
    const brain = NeuralNet.deserialize(payload.brain);
    if (!brain) return false;
    state.bestBrain = brain;
    state.gen = payload.gen || 1;
    state.historicalBest = payload.historicalBest || 0;
    state.bestFitness = state.historicalBest;
    if (Array.isArray(payload.history)) state.history = payload.history.slice(-HISTORY_MAX);
    return true;
  } catch (e) { return false; }
}

async function clearSavedBrain() {
  try { await fetch("/api/brain", { method: "DELETE" }); } catch (e) {}
}

// ---------- rendering ----------

function drawTrack() {
  const track = TRACKS[state.trackIdx];
  tctx.clearRect(0, -TRACK_Y_OFFSET, trackDim.w, trackDim.h);

  // asphalt (soft) fill
  tctx.beginPath();
  drawPolyline(tctx, track.outer, true);
  tctx.moveTo(track.inner[0].x, track.inner[0].y);
  for (let i = track.inner.length - 1; i >= 0; i--) {
    tctx.lineTo(track.inner[i].x, track.inner[i].y);
  }
  tctx.closePath();
  tctx.fillStyle = "#EFEAE1";
  tctx.fill("evenodd");

  // surface patches (grass, ice, dirt)
  if (track.surfaces && track.surfaces.length) {
    for (const s of track.surfaces) {
      tctx.beginPath();
      tctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      tctx.fillStyle = s.color;
      tctx.globalAlpha = 0.85;
      tctx.fill();
      tctx.globalAlpha = 1;
    }
  }

  drawSkids();

  // centerline dashed
  tctx.setLineDash([4, 8]);
  tctx.strokeStyle = "#D9D2C6";
  tctx.lineWidth = 1.2;
  tctx.beginPath();
  drawPolyline(tctx, track.center, true);
  tctx.stroke();
  tctx.setLineDash([]);

  // walls
  tctx.strokeStyle = "#B9A6E6";
  tctx.lineWidth = 2.2;
  tctx.beginPath();
  drawPolyline(tctx, track.outer, true);
  tctx.stroke();
  tctx.strokeStyle = "#F4C7C3";
  tctx.beginPath();
  drawPolyline(tctx, track.inner, true);
  tctx.stroke();

  // start marker
  const s = track.start;
  const a = track.startAngle;
  tctx.save();
  tctx.translate(s.x, s.y);
  tctx.rotate(a + Math.PI / 2);
  tctx.fillStyle = "rgba(185,166,230,0.5)";
  tctx.fillRect(-18, -3, 36, 6);
  tctx.restore();
}

function drawSkids() {
  const skids = window.SKIDS;
  if (!skids.length) return;
  // age + cull
  for (let i = skids.length - 1; i >= 0; i--) {
    skids[i].age += 1;
    if (skids[i].age >= SKID_LIFE) skids.splice(i, 1);
  }
  if (skids.length > SKID_MAX) skids.splice(0, skids.length - SKID_MAX);
  tctx.lineCap = "round";
  for (let i = 0; i < skids.length; i++) {
    const s = skids[i];
    const life = 1 - s.age / SKID_LIFE;
    const alpha = life * 0.4 * s.strength;
    tctx.strokeStyle = s.boost
      ? `rgba(240, 140, 90, ${alpha})`   // warm orange for nitro
      : `rgba(70, 55, 90, ${alpha})`;    // deep purple for drift
    tctx.lineWidth = s.boost ? 2.4 : 1.8;
    tctx.beginPath();
    tctx.moveTo(s.ax, s.ay);
    tctx.lineTo(s.bx, s.by);
    tctx.stroke();
  }
}

function drawPolyline(ctx, pts, closed) {
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  if (closed) ctx.closePath();
}

function drawCars() {
  // dead cars ghosted
  for (const car of state.population) {
    if (car.alive) continue;
    tctx.globalAlpha = 0.18;
    drawCarShape(car);
  }
  tctx.globalAlpha = 1;
  // alive
  const leader = getLeader();
  for (const car of state.population) {
    if (!car.alive) continue;
    if (car === leader) continue;
    drawCarShape(car);
  }
  if (leader && leader.alive) drawCarShape(leader, true);
}

function drawCarShape(car, highlight = false) {
  tctx.save();
  tctx.translate(car.x, car.y);
  tctx.rotate(car.angle);
  if (highlight) {
    // sensor rays
    tctx.strokeStyle = "rgba(185,166,230,0.35)";
    tctx.lineWidth = 1;
    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      const d = car.sensors[i] * SENSOR_RANGE;
      tctx.beginPath();
      tctx.moveTo(0, 0);
      tctx.lineTo(Math.cos(SENSOR_ANGLES[i]) * d, Math.sin(SENSOR_ANGLES[i]) * d);
      tctx.stroke();
    }
  }
  // body
  const w = 14, h = 8;
  tctx.fillStyle = car.color;
  roundRect(tctx, -w / 2, -h / 2, w, h, 3);
  tctx.fill();
  if (highlight) {
    tctx.strokeStyle = "#3B3A36";
    tctx.lineWidth = 1.3;
    tctx.stroke();
  }
  tctx.restore();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

function getLeader() {
  let best = null;
  let bf = -Infinity;
  for (const c of state.population) {
    if (!c.alive) continue;
    if (c.fitness > bf) { bf = c.fitness; best = c; }
  }
  return best;
}

// ---------- network viz ----------

const LABELS_IN = [
  "← ray", "↖ ray", "↑ ray", "↗ ray", "→ ray",
  "▲▲ long", "speed", "slip", "yaw",
];
const LABELS_OUT = ["steer", "throttle"];

function drawNet() {
  nctx.clearRect(0, 0, netDim.w, netDim.h);
  const leader = getLeader() || state.population[0];
  if (!leader) return;
  const brain = leader.brain;
  const layers = brain.layers;

  const padX = 52;
  const padY = 26;
  const W = netDim.w;
  const H = netDim.h;
  const stepX = (W - padX * 2) / (layers.length - 1);

  const positions = layers.map((n, li) => {
    const x = padX + li * stepX;
    const gap = (H - padY * 2) / Math.max(1, n - 1);
    const arr = [];
    for (let i = 0; i < n; i++) {
      const y = n === 1 ? H / 2 : padY + i * gap;
      arr.push({ x, y });
    }
    return arr;
  });

  // connections
  for (let l = 0; l < brain.weights.length; l++) {
    const w = brain.weights[l];
    const rows = layers[l + 1];
    const cols = layers[l];
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const wv = w[r * cols + c];
        const prevAct = brain.activations[l][c];
        const flow = Math.abs(wv * prevAct);
        const alpha = Math.min(0.85, 0.05 + flow * 0.9);
        const width = Math.min(2.4, 0.2 + Math.abs(wv) * 1.1);
        nctx.strokeStyle = wv > 0
          ? `rgba(185,166,230,${alpha})`  // lavender +
          : `rgba(244,199,195,${alpha})`; // pink -
        nctx.lineWidth = width;
        nctx.beginPath();
        const a = positions[l][c];
        const b = positions[l + 1][r];
        nctx.moveTo(a.x, a.y);
        // slight curve for airy feel
        const mx = (a.x + b.x) / 2;
        nctx.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y);
        nctx.stroke();
      }
    }
  }

  // nodes
  for (let li = 0; li < layers.length; li++) {
    for (let i = 0; i < layers[li]; i++) {
      const p = positions[li][i];
      const act = brain.activations[li][i];
      const norm = (act + 1) / 2; // 0..1
      const r = 8 + Math.abs(act) * 4;
      // ring
      nctx.beginPath();
      nctx.arc(p.x, p.y, r + 4, 0, Math.PI * 2);
      nctx.fillStyle = `rgba(220,220,220,0.35)`;
      nctx.fill();
      // core
      nctx.beginPath();
      nctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      const hue = act > 0 ? 265 : 350;
      const sat = 45 + Math.abs(act) * 25;
      const light = 85 - Math.abs(act) * 22;
      nctx.fillStyle = `hsl(${hue}, ${sat}%, ${light}%)`;
      nctx.fill();
    }
  }

  // labels
  nctx.fillStyle = "#A6A19A";
  nctx.font = "11px ui-rounded, system-ui, sans-serif";
  nctx.textAlign = "right";
  nctx.textBaseline = "middle";
  for (let i = 0; i < layers[0]; i++) {
    const p = positions[0][i];
    nctx.fillText(LABELS_IN[i] || `in${i}`, p.x - 14, p.y);
  }
  nctx.textAlign = "left";
  const last = layers.length - 1;
  for (let i = 0; i < layers[last]; i++) {
    const p = positions[last][i];
    nctx.fillText(LABELS_OUT[i] || `out${i}`, p.x + 14, p.y);
  }
  nctx.textAlign = "center";
  const layerNames = layers.map((_, i) => {
    if (i === 0) return "entrada";
    if (i === layers.length - 1) return "saída";
    return "oculta " + i;
  });
  for (let li = 0; li < layers.length; li++) {
    nctx.fillText(layerNames[li] || `L${li}`, positions[li][0].x, 18);
  }
}

// ---------- HUD ----------

function updateHUD() {
  statGen.textContent = state.gen;
  const alive = state.population.filter(c => c.alive).length;
  statAlive.textContent = alive + " / " + state.population.length;
  const leader = getLeader();
  const best = leader ? leader.fitness : (state.bestFitness === -Infinity ? 0 : state.bestFitness);
  statBest.textContent = Math.max(0, Math.round(best));
  const avg = state.population.reduce((s, c) => s + Math.max(0, c.fitness), 0) / Math.max(1, state.population.length);
  statAvg.textContent = Math.round(avg);
  if (netSub && leader) {
    netSub.textContent = leader.brain.layers.join("·") + " · líder";
  }
  if (statTimer) {
    const remaining = Math.max(0, MAX_GEN_FRAMES - state.genFrames);
    // frames per real second ≈ 60fps × substeps (state.speed)
    const secs = remaining / (60 * Math.max(1, state.speed));
    statTimer.textContent = secs >= 10 ? `${Math.ceil(secs)}s` : `${secs.toFixed(1)}s`;
  }
}

// ---------- loop ----------

function simStep() {
  if (state.paused) return;
  const substeps = state.speed;
  window.POP = null;
  for (let s = 0; s < substeps; s++) {
    state.genFrames++;
    let anyAlive = false;
    for (const car of state.population) {
      if (!car.alive) continue;
      car.step(1);
      if (car.alive) anyAlive = true;
    }
    if (!anyAlive || state.genFrames > MAX_GEN_FRAMES) {
      nextGeneration();
      break;
    }
  }
}

function drawAll() {
  drawTrack();
  drawCars();
  drawNet();
  drawChart();
  updateHUD();
}

function drawChart() {
  if (!cctx || !chartDim) return;
  const W = chartDim.w, H = chartDim.h;
  cctx.clearRect(0, 0, W, H);
  const pts = state.history;
  if (!pts.length) return;

  const padL = 40, padR = 12, padT = 12, padB = 22;
  const iw = W - padL - padR;
  const ih = H - padT - padB;

  let maxVal = 0;
  for (const p of pts) if (p.best > maxVal) maxVal = p.best;
  maxVal = Math.max(maxVal, 10);
  const minGen = pts[0].gen;
  const maxGen = Math.max(pts[pts.length - 1].gen, minGen + 1);
  const genRange = maxGen - minGen;

  const xOf = (g) => padL + (iw * (g - minGen) / genRange);
  const yOf = (v) => padT + ih - (ih * (v / maxVal));

  // grid
  cctx.strokeStyle = "#ECE7DF";
  cctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ih * i / 4);
    cctx.beginPath();
    cctx.moveTo(padL, y);
    cctx.lineTo(padL + iw, y);
    cctx.stroke();
  }

  // axis labels
  cctx.fillStyle = "#A6A19A";
  cctx.font = "10px ui-rounded, system-ui, sans-serif";
  cctx.textAlign = "right";
  cctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const y = padT + (ih * i / 4);
    const v = Math.round(maxVal * (1 - i / 4));
    cctx.fillText(String(v), padL - 6, y);
  }
  cctx.textAlign = "left";
  cctx.textBaseline = "top";
  cctx.fillText(`gen ${minGen}`, padL, padT + ih + 4);
  cctx.textAlign = "right";
  cctx.fillText(`gen ${maxGen}`, padL + iw, padT + ih + 4);

  // avg (soft pink fill under line)
  cctx.beginPath();
  cctx.moveTo(xOf(pts[0].gen), padT + ih);
  for (const p of pts) cctx.lineTo(xOf(p.gen), yOf(p.avg));
  cctx.lineTo(xOf(pts[pts.length - 1].gen), padT + ih);
  cctx.closePath();
  cctx.fillStyle = "rgba(244,199,195,0.35)";
  cctx.fill();

  cctx.beginPath();
  cctx.moveTo(xOf(pts[0].gen), yOf(pts[0].avg));
  for (const p of pts) cctx.lineTo(xOf(p.gen), yOf(p.avg));
  cctx.strokeStyle = "rgba(244,140,140,0.85)";
  cctx.lineWidth = 1.6;
  cctx.stroke();

  // best (lavender line)
  cctx.beginPath();
  cctx.moveTo(xOf(pts[0].gen), yOf(pts[0].best));
  for (const p of pts) cctx.lineTo(xOf(p.gen), yOf(p.best));
  cctx.strokeStyle = "rgba(150,120,220,0.95)";
  cctx.lineWidth = 2.2;
  cctx.stroke();

  // legend
  cctx.font = "11px ui-rounded, system-ui, sans-serif";
  cctx.textAlign = "left";
  cctx.textBaseline = "middle";
  cctx.fillStyle = "rgba(150,120,220,0.95)";
  cctx.fillRect(padL + 6, padT + 6, 12, 3);
  cctx.fillStyle = "#4A4A4A";
  cctx.fillText("melhor", padL + 24, padT + 7);
  cctx.fillStyle = "rgba(244,140,140,0.85)";
  cctx.fillRect(padL + 90, padT + 6, 12, 3);
  cctx.fillStyle = "#4A4A4A";
  cctx.fillText("média", padL + 108, padT + 7);
}

function tick() {
  if (!document.hidden) {
    simStep();
    drawAll();
  }
  requestAnimationFrame(tick);
}

// Web Worker heartbeat: keeps the simulation running even when the tab
// is hidden. rAF is paused in background tabs, and main-thread timers get
// throttled to ~1s. A Worker's setInterval keeps firing at ~16ms, and its
// postMessage wakes the main thread event loop so we can step the sim.
const HEARTBEAT_CODE = `
let id = setInterval(() => postMessage(0), 16);
onmessage = (e) => {
  if (e.data === 'stop') { clearInterval(id); }
};
`;
try {
  const _hbBlob = new Blob([HEARTBEAT_CODE], { type: "application/javascript" });
  const _hbWorker = new Worker(URL.createObjectURL(_hbBlob));
  _hbWorker.onmessage = () => {
    if (document.hidden) simStep();
  };
} catch (e) { /* worker unsupported: fall back to rAF-only */ }

// ---------- controls ----------

btnPause.addEventListener("click", () => {
  state.paused = !state.paused;
  btnPause.textContent = state.paused ? "retomar" : "pausar";
});
btnReset.addEventListener("click", () => resetGeneration(true));
speedSlider.addEventListener("input", (e) => { state.speed = +e.target.value; saveSettings(); });
popSlider.addEventListener("input", (e) => { state.popSize = +e.target.value; saveSettings(); });
mutSlider.addEventListener("input", (e) => { state.mutRate = +e.target.value / 100; saveSettings(); });

loadSettings();
const urlTrack = new URLSearchParams(location.search).get("t");
if (urlTrack !== null) {
  const t = parseInt(urlTrack, 10) - 1;
  if (t >= 0 && t < TRACKS.length) state.trackIdx = t;
}
// reflect loaded state in the sliders
speedSlider.value = state.speed;
popSlider.value = state.popSize;
mutSlider.value = Math.round(state.mutRate * 100);

buildTrackPicker();
(async () => {
  await migrateLocalBrainIfAny();
  await loadBrain();
  resetGeneration(false);
  tick();
})();

// ---------- background training toggle ----------

let bgTraining = false;
let bgSnapshot = null; // {gen, historicalBest, startedAtMs}
let bgSyncTimer = null;

function updateTrainButton() {
  if (!btnTrain) return;
  btnTrain.classList.toggle("on", bgTraining);
  btnTrain.textContent = bgTraining ? "treinando no server" : "treinar no server";
}

async function fetchTrainingStatus() {
  try {
    const res = await fetch("/api/training");
    if (!res.ok) return null;
    return await res.json();
  } catch (e) { return null; }
}

async function setTraining(enabled) {
  try {
    const res = await fetch("/api/training", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ enabled }),
    });
    if (!res.ok) return null;
    return await res.json();
  } catch (e) { return null; }
}

async function syncBrainFromServer() {
  if (!bgTraining) return;
  try {
    const res = await fetch("/api/brain");
    if (!res.ok) return;
    const p = await res.json();
    if (!p.present) return;
    const brain = NeuralNet.deserialize(p.brain);
    if (!brain) return;
    state.bestBrain = brain;
    state.gen = p.gen || state.gen;
    state.historicalBest = Math.max(state.historicalBest, p.historicalBest || 0);
  } catch (e) {}
}

function showToast(gens, bestDelta, seconds) {
  if (!toast) return;
  toastGens.textContent = gens;
  toastBest.textContent = (bestDelta >= 0 ? "+" : "") + Math.round(bestDelta);
  toastTime.textContent = secondsHuman(seconds);
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 8000);
}

function secondsHuman(s) {
  s = Math.max(0, Math.round(s));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  if (m < 60) return `${m}m ${rem}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${m % 60}m`;
}

if (btnTrain) {
  btnTrain.addEventListener("click", async () => {
    if (!bgTraining) {
      const cur = await fetchTrainingStatus();
      const startGen = cur?.gen ?? 0;
      const startBest = cur?.historicalBest ?? 0;
      bgSnapshot = { gen: startGen, historicalBest: startBest, startedAtMs: performance.now() };
      const r = await setTraining(true);
      bgTraining = !!(r && r.enabled);
      updateTrainButton();
      if (bgTraining) {
        bgSyncTimer = setInterval(syncBrainFromServer, 5000);
      }
    } else {
      const r = await setTraining(false);
      bgTraining = !!(r && r.enabled);
      updateTrainButton();
      if (bgSyncTimer) { clearInterval(bgSyncTimer); bgSyncTimer = null; }
      // fetch final stats and show summary
      const final = await fetchTrainingStatus();
      if (final && bgSnapshot) {
        const gens = Math.max(0, (final.gen || 0) - bgSnapshot.gen);
        const bestDelta = (final.historicalBest || 0) - bgSnapshot.historicalBest;
        const secs = (performance.now() - bgSnapshot.startedAtMs) / 1000;
        showToast(gens, bestDelta, secs);
      }
      bgSnapshot = null;
      // pick up the final server brain
      await syncBrainFromServer.call(null);
    }
  });
}

(async () => {
  const cur = await fetchTrainingStatus();
  bgTraining = !!(cur && cur.enabled);
  updateTrainButton();
  if (bgTraining) {
    bgSnapshot = { gen: cur.gen || 0, historicalBest: cur.historicalBest || 0, startedAtMs: performance.now() };
    bgSyncTimer = setInterval(syncBrainFromServer, 5000);
  }
})();

// one-time upload of a previously saved localStorage brain to the server
async function migrateLocalBrainIfAny() {
  const LEGACY_KEY = "carrinho-nn:save-v1";
  try {
    const raw = localStorage.getItem(LEGACY_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (!payload || !payload.brain) return;
    const res = await fetch("/api/brain");
    if (!res.ok) return;
    const cur = await res.json();
    if (cur.present) { localStorage.removeItem(LEGACY_KEY); return; }
    await fetch("/api/brain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    localStorage.removeItem(LEGACY_KEY);
  } catch (e) {}
}
