// Background trainer. Runs the same JS simulation the browser uses, but
// headless — no canvas, no rendering. Polls the server; only trains when
// no viewer has sent a heartbeat recently. Saves the champion back to /api/brain.
//
// Start manually:   node trainer.js
// Under systemd:    see README (write a user service pointing here)

const path = require("path");
const staticDir = path.join(__dirname, "static", "js");
const { NeuralNet } = require(path.join(staticDir, "nn.js"));
const { TRACKS } = require(path.join(staticDir, "tracks.js"));
const { Car } = require(path.join(staticDir, "car.js"));

const SERVER = process.env.SERVER || "http://127.0.0.1:5055";
const POP_SIZE = 40;
const MUT_RATE = 0.08;
const MAX_FRAMES = 5000;
const IDLE_POLL_MS = 3000;

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

async function loadBrain() {
  try {
    const res = await fetch(SERVER + "/api/brain");
    if (!res.ok) return null;
    const p = await res.json();
    if (!p.present) return null;
    return NeuralNet.deserialize(p.brain);
  } catch (e) { return null; }
}

async function saveBrain(brain, gen, historicalBest) {
  try {
    await fetch(SERVER + "/api/brain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ brain: brain.serialize(), gen, historicalBest }),
    });
  } catch (e) { /* ignore */ }
}

async function trainingEnabled() {
  try {
    const res = await fetch(SERVER + "/api/training");
    if (!res.ok) return false;
    const j = await res.json();
    return !!j.enabled;
  } catch (e) { return false; }
}

function seedPop(bestBrain, track) {
  const pop = [];
  for (let i = 0; i < POP_SIZE; i++) {
    const b = bestBrain.clone();
    if (i > 0) b.mutate(MUT_RATE, 0.35);
    pop.push(new Car(b, track));
  }
  return pop;
}

function runGeneration(pop, track) {
  let frames = 0;
  while (frames < MAX_FRAMES) {
    frames++;
    const alive = pop.filter((c) => c.alive);
    if (!alive.length) break;
    alive.sort((a, b) => b.fitness - a.fitness);
    const denom = Math.max(1, alive.length - 1);
    for (let i = 0; i < alive.length; i++) {
      alive[i].rankNorm = 1 - i / denom;
    }
    for (const c of pop) if (c.alive) c.step(1);
  }
  return frames;
}

function nextChampion(pop) {
  pop.sort((a, b) => b.fitness - a.fitness);
  const survivors = pop.slice(0, Math.max(4, Math.floor(pop.length * 0.2)));
  // build next-gen elite (survivors crossbred, best kept)
  const bestBrain = survivors[0].brain.clone();
  return { bestBrain, bestFitness: survivors[0].fitness, survivors };
}

async function main() {
  console.log("[trainer] starting; SERVER=" + SERVER);
  let bestBrain = await loadBrain();
  if (!bestBrain) {
    bestBrain = new NeuralNet();
    console.log("[trainer] no saved brain, starting fresh " + bestBrain.layers.join("·"));
  } else {
    console.log("[trainer] loaded brain " + bestBrain.layers.join("·"));
  }
  let gen = 1;
  let historicalBest = 0;
  let trackIdx = 0;

  while (true) {
    if (!(await trainingEnabled())) {
      // toggle is off — sleep and refresh our brain so we pick up any browser progress
      await sleep(IDLE_POLL_MS);
      const fresh = await loadBrain();
      if (fresh) bestBrain = fresh;
      continue;
    }

    const track = TRACKS[trackIdx];
    const pop = seedPop(bestBrain, track);
    const frames = runGeneration(pop, track);
    const { bestBrain: champ, bestFitness } = nextChampion(pop);
    bestBrain = champ;
    if (bestFitness > historicalBest) historicalBest = bestFitness;
    gen++;
    trackIdx = (trackIdx + 1) % TRACKS.length;
    await saveBrain(bestBrain, gen, historicalBest);
    if (gen % 10 === 0) {
      console.log(`[trainer] gen=${gen} best=${Math.round(historicalBest)} frames=${frames} topo=${bestBrain.layers.join("·")}`);
    }
  }
}

main().catch((e) => { console.error("[trainer] fatal", e); process.exit(1); });
