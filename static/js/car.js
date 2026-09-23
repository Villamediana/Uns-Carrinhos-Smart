// Car simulation.

const SENSOR_ANGLES = [
  -Math.PI / 2,       // left
  -Math.PI / 4,       // diag-left
  0,                  // front
  Math.PI / 4,        // diag-right
  Math.PI / 2,        // right
];
const SENSOR_RANGE = 140;
const LONG_RANGE = 280;

function rayCircleIntersect(ox, oy, dx, dy, maxT, cx, cy, r) {
  const ex = cx - ox, ey = cy - oy;
  const t = ex * dx + ey * dy;
  if (t < 0 || t > maxT) return -1;
  const closest2 = ex * ex + ey * ey - t * t;
  const r2 = r * r;
  if (closest2 > r2) return -1;
  const back = Math.sqrt(r2 - closest2);
  const hit = t - back;
  return hit > 0 ? hit : -1;
}

function raySegIntersect(ox, oy, dx, dy, maxT, s) {
  // Ray: (ox,oy) + t*(dx,dy). Segment: (ax,ay)->(bx,by).
  const rx = dx, ry = dy;
  const sx = s.bx - s.ax, sy = s.by - s.ay;
  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < 1e-9) return -1;
  const qx = s.ax - ox, qy = s.ay - oy;
  const t = (qx * sy - qy * sx) / denom;
  const u = (qx * ry - qy * rx) / denom;
  if (t < 0 || t > maxT) return -1;
  if (u < 0 || u > 1) return -1;
  return t;
}

class Car {
  constructor(brain, track) {
    this.brain = brain;
    this.track = track;
    this.x = track.start.x;
    this.y = track.start.y;
    this.angle = track.startAngle;
    // velocity in world coords
    this.vx = 0;
    this.vy = 0;
    this.speed = 0;             // signed forward speed component
    this.slip = 0;              // lateral speed magnitude, for skid detection
    this.alive = true;
    this.age = 0;
    this.staleness = 0;
    this.progress = 0;         // index along centerline
    this.laps = 0;
    this.lastLapAge = 0;
    this.speedBonus = 0;
    this.driftBonus = 0;
    this.slowPenalty = 0;
    this.fitness = 0;
    this.sensors = new Float32Array(SENSOR_ANGLES.length);
    this.longSensor = 1;
    this.prevAngle = this.angle;
    this.prevSpeed = 0;
    this.yawRate = 0;
    this.accel = 0;
    this.rankNorm = 0;
    // pastel palette for cars
    const hues = [340, 200, 150, 30, 270, 20, 180, 60];
    const h = hues[Math.floor(Math.random() * hues.length)];
    this.color = `hsl(${h}, 55%, 78%)`;
    // last rear wheel positions (for skid segments)
    this._lastRL = null;
    this._lastRR = null;
  }

  sense() {
    const walls = this.track.walls;
    const pop = (typeof window !== "undefined") ? window.POP : null;
    for (let i = 0; i < SENSOR_ANGLES.length; i++) {
      const a = this.angle + SENSOR_ANGLES[i];
      const dx = Math.cos(a);
      const dy = Math.sin(a);
      let best = SENSOR_RANGE;
      for (let w = 0; w < walls.length; w++) {
        const t = raySegIntersect(this.x, this.y, dx, dy, best, walls[w]);
        if (t >= 0 && t < best) best = t;
      }
      if (pop) {
        for (let k = 0; k < pop.length; k++) {
          const o = pop[k];
          if (o === this || !o.alive) continue;
          const t = rayCircleIntersect(this.x, this.y, dx, dy, best, o.x, o.y, 8);
          if (t >= 0 && t < best) best = t;
        }
      }
      this.sensors[i] = best / SENSOR_RANGE; // 0..1 (1 = clear)
    }
    // long-range front ray
    let bestLong = LONG_RANGE;
    const ldx = Math.cos(this.angle);
    const ldy = Math.sin(this.angle);
    for (let w = 0; w < walls.length; w++) {
      const t = raySegIntersect(this.x, this.y, ldx, ldy, bestLong, walls[w]);
      if (t >= 0 && t < bestLong) bestLong = t;
    }
    if (pop) {
      for (let k = 0; k < pop.length; k++) {
        const o = pop[k];
        if (o === this || !o.alive) continue;
        const t = rayCircleIntersect(this.x, this.y, ldx, ldy, bestLong, o.x, o.y, 8);
        if (t >= 0 && t < bestLong) bestLong = t;
      }
    }
    this.longSensor = bestLong / LONG_RANGE;
  }

  step(dt) {
    if (!this.alive) return;
    this.sense();

    // derived signals
    let dAngle = this.angle - this.prevAngle;
    if (dAngle > Math.PI) dAngle -= 2 * Math.PI;
    else if (dAngle < -Math.PI) dAngle += 2 * Math.PI;
    this.yawRate = dAngle / dt;
    this.accel = (this.speed - this.prevSpeed) / dt;

    const input = [
      this.sensors[0], this.sensors[1], this.sensors[2],
      this.sensors[3], this.sensors[4],
      this.longSensor,
      Math.tanh(this.speed / 3.5),
      Math.tanh(this.slip / 2),
      Math.tanh(this.yawRate * 3),
    ];
    const out = this.brain.forward(input);
    const steer = out[0];               // -1..1
    const throttle = out[1];            // -1..1

    // --- bicycle-ish physics, no hard speed cap ---
    // Forward speed in current frame (for yaw calc)
    const fx0 = Math.cos(this.angle), fy0 = Math.sin(this.angle);
    const vf0 = this.vx * fx0 + this.vy * fy0;

    // Steering: yaw rate depends on speed. Faster car → bigger turn radius.
    const yawRate = steer * 0.075 * (vf0 / (Math.abs(vf0) + 2.5));
    this.angle += yawRate * dt;

    // New car axes
    const fx = Math.cos(this.angle), fy = Math.sin(this.angle);
    const lx = -fy, ly = fx;

    // Decompose velocity in new car frame
    let vf = this.vx * fx + this.vy * fy;
    let vs = this.vx * lx + this.vy * ly;

    // Grip: pulls lateral velocity toward zero, capped by max grip force per frame.
    const GRIP_MAX = 0.22;
    const gripPull = GRIP_MAX * dt;
    if (Math.abs(vs) < gripPull) vs = 0;
    else vs -= Math.sign(vs) * gripPull;

    // Longitudinal: throttle applies acceleration.
    const ACCEL = 0.10, REVERSE = 0.07;
    if (throttle >= 0) vf += throttle * ACCEL * dt;
    else vf += throttle * REVERSE * dt;

    // Drag: linear + quadratic (softer so top speed stays around ~5)
    const speedMag = Math.hypot(vf, vs);
    vf -= vf * (0.003 + 0.002 * speedMag) * dt;
    vs -= vs * (0.006 + 0.002 * speedMag) * dt;

    // Recompose world velocity
    this.vx = vf * fx + vs * lx;
    this.vy = vf * fy + vs * ly;
    this.speed = vf;
    this.slip = Math.abs(vs);

    const nx = this.x + this.vx * dt;
    const ny = this.y + this.vy * dt;

    // collision: quickly check if any wall segment lies within car body radius
    if (this.collides(nx, ny)) {
      this.alive = false;
      return;
    }

    this.x = nx;
    this.y = ny;
    this.age += dt;

    // skid marks: emit segments at rear wheels when slipping
    const rearOffX = -5 * fx, rearOffY = -5 * fy;
    const sideOffX = 3 * lx, sideOffY = 3 * ly;
    const rlx = this.x + rearOffX + sideOffX;
    const rly = this.y + rearOffY + sideOffY;
    const rrx = this.x + rearOffX - sideOffX;
    const rry = this.y + rearOffY - sideOffY;
    if (this.slip > 0.35 && this._lastRL && typeof window !== "undefined" && window.SKIDS) {
      const strength = Math.min(1, (this.slip - 0.35) / 1.2);
      window.SKIDS.push({
        ax: this._lastRL.x, ay: this._lastRL.y, bx: rlx, by: rly,
        age: 0, strength,
      });
      window.SKIDS.push({
        ax: this._lastRR.x, ay: this._lastRR.y, bx: rrx, by: rry,
        age: 0, strength,
      });
    }
    this._lastRL = { x: rlx, y: rly };
    this._lastRR = { x: rrx, y: rry };

    // update fitness by finding nearest centerline point ahead
    const center = this.track.center;
    const n = center.length;
    let bestIdx = this.progress;
    let bestD = Infinity;
    // search a window ahead
    for (let k = 0; k < 20; k++) {
      const idx = (this.progress + k) % n;
      const cx = center[idx].x, cy = center[idx].y;
      const d = (cx - this.x) * (cx - this.x) + (cy - this.y) * (cy - this.y);
      if (d < bestD) { bestD = d; bestIdx = idx; }
    }
    let frameReward = 0;
    // gentle penalty only for truly parked cars
    const SLOW_THRESHOLD = 0.6;
    if (this.speed < SLOW_THRESHOLD) {
      const gap = (SLOW_THRESHOLD - this.speed) * dt;
      this.slowPenalty += gap * 0.05;
      frameReward -= gap * 0.02;
    }
    if (bestIdx !== this.progress) {
      let delta = bestIdx - this.progress;
      if (delta < 0) delta += n;
      if (delta > 0 && delta < 20) {
        this.progress = bestIdx;
        this.staleness = 0;
        frameReward += delta * 0.6;
        // drift is only rewarded when it comes with actual progress
        if (this.slip > 0.5) {
          const d = Math.min(2, this.slip) * delta * 0.4;
          this.driftBonus += d;
          frameReward += d;
        }
        if (bestIdx === 0 && this.age > 2) {
          this.laps++;
          const lapTime = Math.max(1, this.age - this.lastLapAge);
          this.speedBonus += 4000 / lapTime;
          this.lastLapAge = this.age;
          frameReward += 8;
        }
      }
    } else {
      this.staleness += dt;
      frameReward -= 0.05;
    }
    this.fitness = this.laps * n + this.progress + this.speedBonus + this.driftBonus - this.staleness * 0.5 - this.slowPenalty;

    // reward flows back into the network (online reinforcement)
    if (frameReward !== 0) this.brain.reinforce(frameReward);

    if (this.staleness > 90) this.alive = false; // idle timeout

    this.prevAngle = this.angle;
    this.prevSpeed = this.speed;
  }

  collides(x, y) {
    const walls = this.track.walls;
    const r = 7; // car body radius
    for (let i = 0; i < walls.length; i++) {
      const w = walls[i];
      // distance from point (x,y) to segment
      const vx = w.bx - w.ax, vy = w.by - w.ay;
      const wx = x - w.ax, wy = y - w.ay;
      const c1 = vx * wx + vy * wy;
      const c2 = vx * vx + vy * vy;
      let t = c2 > 0 ? c1 / c2 : 0;
      if (t < 0) t = 0; else if (t > 1) t = 1;
      const px = w.ax + t * vx, py = w.ay + t * vy;
      const dx = x - px, dy = y - py;
      if (dx * dx + dy * dy < r * r) return true;
    }
    return false;
  }
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = { Car, SENSOR_ANGLES, SENSOR_RANGE, LONG_RANGE };
}
