// Simple feedforward NN with 4 layers.
// Layers: [inputs, hidden1, hidden2, outputs]
// Activation: tanh on hidden, tanh on output (mapped to control range).

const NN_LAYERS = [9, 8, 6, 2];
const HIDDEN_MIN = 2;
const HIDDEN_MAX = 16;

function randn() {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class NeuralNet {
  constructor(layers = NN_LAYERS) {
    this.layers = layers.slice();
    this.weights = [];
    this.biases = [];
    for (let i = 0; i < layers.length - 1; i++) {
      const rows = layers[i + 1];
      const cols = layers[i];
      const w = new Float32Array(rows * cols);
      const b = new Float32Array(rows);
      // Xavier-ish init
      const s = Math.sqrt(2 / cols);
      for (let j = 0; j < w.length; j++) w[j] = randn() * s;
      for (let j = 0; j < b.length; j++) b[j] = randn() * 0.1;
      this.weights.push(w);
      this.biases.push(b);
    }
    // Cache last activations for visualization
    this.activations = layers.map((n) => new Float32Array(n));
  }

  forward(input) {
    let a = this.activations[0];
    for (let i = 0; i < input.length; i++) a[i] = input[i];
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const b = this.biases[l];
      const prev = this.activations[l];
      const next = this.activations[l + 1];
      const rows = this.layers[l + 1];
      const cols = this.layers[l];
      for (let r = 0; r < rows; r++) {
        let sum = b[r];
        const base = r * cols;
        for (let c = 0; c < cols; c++) sum += w[base + c] * prev[c];
        next[r] = Math.tanh(sum);
      }
    }
    return this.activations[this.activations.length - 1];
  }

  // Reward-modulated Hebbian nudge on the output layer.
  // Positive reward strengthens weights whose pre/post neurons fired together.
  reinforce(reward, lr = 0.0006) {
    if (reward === 0 || !isFinite(reward)) return;
    const last = this.weights.length - 1;
    const w = this.weights[last];
    const b = this.biases[last];
    const prev = this.activations[last];
    const post = this.activations[last + 1];
    const rows = this.layers[last + 1];
    const cols = this.layers[last];
    const step = lr * reward;
    for (let r = 0; r < rows; r++) {
      const pr = post[r];
      const base = r * cols;
      for (let c = 0; c < cols; c++) {
        w[base + c] += step * pr * prev[c];
      }
      b[r] += step * pr * 0.5;
    }
    // clamp to keep weights bounded
    for (let i = 0; i < w.length; i++) {
      if (w[i] > 6) w[i] = 6; else if (w[i] < -6) w[i] = -6;
    }
    for (let i = 0; i < b.length; i++) {
      if (b[i] > 6) b[i] = 6; else if (b[i] < -6) b[i] = -6;
    }
  }

  clone() {
    const n = new NeuralNet(this.layers);
    for (let i = 0; i < this.weights.length; i++) {
      n.weights[i].set(this.weights[i]);
      n.biases[i].set(this.biases[i]);
    }
    return n;
  }

  mutate(rate = 0.08, strength = 0.35) {
    for (let l = 0; l < this.weights.length; l++) {
      const w = this.weights[l];
      const b = this.biases[l];
      for (let i = 0; i < w.length; i++) {
        if (Math.random() < rate) w[i] += randn() * strength;
      }
      for (let i = 0; i < b.length; i++) {
        if (Math.random() < rate) b[i] += randn() * strength;
      }
    }
  }

  // Structural mutation: add one neuron to a hidden layer.
  // hiddenIdx is the position in this.layers (1..layers.length-2).
  addNeuron(hiddenIdx) {
    if (hiddenIdx < 1 || hiddenIdx >= this.layers.length - 1) return false;
    if (this.layers[hiddenIdx] >= HIDDEN_MAX) return false;

    // weights[hiddenIdx - 1] connects layers[hiddenIdx-1] -> layers[hiddenIdx]
    // We add a NEW ROW: rows++ , cols unchanged.
    const inW = this.weights[hiddenIdx - 1];
    const inRows = this.layers[hiddenIdx];
    const inCols = this.layers[hiddenIdx - 1];
    const newInW = new Float32Array((inRows + 1) * inCols);
    newInW.set(inW);
    const s = Math.sqrt(2 / inCols);
    for (let c = 0; c < inCols; c++) newInW[inRows * inCols + c] = randn() * s * 0.5;
    this.weights[hiddenIdx - 1] = newInW;
    // bias: append one
    const inB = this.biases[hiddenIdx - 1];
    const newInB = new Float32Array(inRows + 1);
    newInB.set(inB);
    newInB[inRows] = randn() * 0.1;
    this.biases[hiddenIdx - 1] = newInB;

    // weights[hiddenIdx] connects layers[hiddenIdx] -> layers[hiddenIdx+1]
    // We add a NEW COLUMN: cols++ , rows unchanged.
    const outW = this.weights[hiddenIdx];
    const outRows = this.layers[hiddenIdx + 1];
    const outCols = this.layers[hiddenIdx];
    const newOutW = new Float32Array(outRows * (outCols + 1));
    for (let r = 0; r < outRows; r++) {
      for (let c = 0; c < outCols; c++) {
        newOutW[r * (outCols + 1) + c] = outW[r * outCols + c];
      }
      // start new outgoing weights near zero so the new neuron enters gently
      newOutW[r * (outCols + 1) + outCols] = randn() * 0.05;
    }
    this.weights[hiddenIdx] = newOutW;

    this.layers[hiddenIdx] += 1;
    this.activations[hiddenIdx] = new Float32Array(this.layers[hiddenIdx]);
    return true;
  }

  // Structural mutation: remove one neuron from a hidden layer (if allowed).
  removeNeuron(hiddenIdx) {
    if (hiddenIdx < 1 || hiddenIdx >= this.layers.length - 1) return false;
    if (this.layers[hiddenIdx] <= HIDDEN_MIN) return false;

    const rmRow = Math.floor(Math.random() * this.layers[hiddenIdx]);

    // shrink weights[hiddenIdx-1]: drop row rmRow
    const inW = this.weights[hiddenIdx - 1];
    const inRows = this.layers[hiddenIdx];
    const inCols = this.layers[hiddenIdx - 1];
    const newInW = new Float32Array((inRows - 1) * inCols);
    let wi = 0;
    for (let r = 0; r < inRows; r++) {
      if (r === rmRow) continue;
      for (let c = 0; c < inCols; c++) newInW[wi++] = inW[r * inCols + c];
    }
    this.weights[hiddenIdx - 1] = newInW;
    // biases: drop one
    const inB = this.biases[hiddenIdx - 1];
    const newInB = new Float32Array(inRows - 1);
    let bi = 0;
    for (let i = 0; i < inRows; i++) if (i !== rmRow) newInB[bi++] = inB[i];
    this.biases[hiddenIdx - 1] = newInB;

    // shrink weights[hiddenIdx]: drop column rmRow
    const outW = this.weights[hiddenIdx];
    const outRows = this.layers[hiddenIdx + 1];
    const outCols = this.layers[hiddenIdx];
    const newOutW = new Float32Array(outRows * (outCols - 1));
    for (let r = 0; r < outRows; r++) {
      let cc = 0;
      for (let c = 0; c < outCols; c++) {
        if (c === rmRow) continue;
        newOutW[r * (outCols - 1) + cc++] = outW[r * outCols + c];
      }
    }
    this.weights[hiddenIdx] = newOutW;

    this.layers[hiddenIdx] -= 1;
    this.activations[hiddenIdx] = new Float32Array(this.layers[hiddenIdx]);
    return true;
  }

  sameTopology(other) {
    if (this.layers.length !== other.layers.length) return false;
    for (let i = 0; i < this.layers.length; i++) {
      if (this.layers[i] !== other.layers[i]) return false;
    }
    return true;
  }

  serialize() {
    return {
      layers: this.layers.slice(),
      weights: this.weights.map((w) => Array.from(w)),
      biases: this.biases.map((b) => Array.from(b)),
    };
  }

  static deserialize(obj) {
    if (!obj || !Array.isArray(obj.layers)) return null;
    // reject brains whose IO shape doesn't match the current sensor/output spec
    if (obj.layers[0] !== NN_LAYERS[0]) return null;
    if (obj.layers[obj.layers.length - 1] !== NN_LAYERS[NN_LAYERS.length - 1]) return null;
    const n = new NeuralNet(obj.layers);
    for (let i = 0; i < n.weights.length; i++) {
      if (!obj.weights[i] || obj.weights[i].length !== n.weights[i].length) return null;
      if (!obj.biases[i] || obj.biases[i].length !== n.biases[i].length) return null;
      n.weights[i].set(obj.weights[i]);
      n.biases[i].set(obj.biases[i]);
    }
    return n;
  }

  static crossover(a, b) {
    if (!a.sameTopology(b)) {
      // fall back to cloning the (arbitrary) first parent when shapes differ
      return a.clone();
    }
    const child = new NeuralNet(a.layers);
    for (let l = 0; l < a.weights.length; l++) {
      const wa = a.weights[l];
      const wb = b.weights[l];
      const wc = child.weights[l];
      for (let i = 0; i < wc.length; i++) {
        wc[i] = Math.random() < 0.5 ? wa[i] : wb[i];
      }
      const ba = a.biases[l];
      const bb = b.biases[l];
      const bc = child.biases[l];
      for (let i = 0; i < bc.length; i++) {
        bc[i] = Math.random() < 0.5 ? ba[i] : bb[i];
      }
    }
    return child;
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { NeuralNet, NN_LAYERS, HIDDEN_MIN, HIDDEN_MAX };
}
