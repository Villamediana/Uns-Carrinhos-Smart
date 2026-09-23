// Each track is defined by a centerline (closed loop of points) and a width.
// Walls are built by offsetting the centerline perpendicular by +/- w/2.

function offsetPolyline(points, offset) {
  const n = points.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const p = points[i];
    const prev = points[(i - 1 + n) % n];
    const next = points[(i + 1) % n];
    // average normal from adjacent segments
    const dx1 = p.x - prev.x, dy1 = p.y - prev.y;
    const dx2 = next.x - p.x, dy2 = next.y - p.y;
    const l1 = Math.hypot(dx1, dy1) || 1;
    const l2 = Math.hypot(dx2, dy2) || 1;
    // normals rotated 90deg (right)
    const nx1 = dy1 / l1, ny1 = -dx1 / l1;
    const nx2 = dy2 / l2, ny2 = -dx2 / l2;
    const nx = (nx1 + nx2) * 0.5;
    const ny = (ny1 + ny2) * 0.5;
    const nl = Math.hypot(nx, ny) || 1;
    out.push({ x: p.x + (nx / nl) * offset, y: p.y + (ny / nl) * offset });
  }
  return out;
}

function polylineToSegments(points, closed = true) {
  const segs = [];
  const n = points.length;
  const end = closed ? n : n - 1;
  for (let i = 0; i < end; i++) {
    const a = points[i];
    const b = points[(i + 1) % n];
    segs.push({ ax: a.x, ay: a.y, bx: b.x, by: b.y });
  }
  return segs;
}

function makeSmoothLoop(rawPoints, samplesPerSeg = 6) {
  // Catmull-Rom-ish smoothing for nicer curves
  const n = rawPoints.length;
  const out = [];
  for (let i = 0; i < n; i++) {
    const p0 = rawPoints[(i - 1 + n) % n];
    const p1 = rawPoints[i];
    const p2 = rawPoints[(i + 1) % n];
    const p3 = rawPoints[(i + 2) % n];
    for (let t = 0; t < samplesPerSeg; t++) {
      const s = t / samplesPerSeg;
      const s2 = s * s;
      const s3 = s2 * s;
      const x = 0.5 * ((2 * p1.x) + (-p0.x + p2.x) * s + (2*p0.x - 5*p1.x + 4*p2.x - p3.x) * s2 + (-p0.x + 3*p1.x - 3*p2.x + p3.x) * s3);
      const y = 0.5 * ((2 * p1.y) + (-p0.y + p2.y) * s + (2*p0.y - 5*p1.y + 4*p2.y - p3.y) * s2 + (-p0.y + 3*p1.y - 3*p2.y + p3.y) * s3);
      out.push({ x, y });
    }
  }
  return out;
}

const SURFACE_TYPES = {
  grass: { gripMul: 0.5, accelMul: 0.7, color: "#DFECD5" },
  ice:   { gripMul: 0.28, accelMul: 1.0, color: "#DAE9F0" },
  dirt:  { gripMul: 0.75, accelMul: 0.55, color: "#F2DFC6" },
};

function buildTrack(rawCenter, width, surfaceSpecs) {
  const center = makeSmoothLoop(rawCenter, 8);
  const outer = offsetPolyline(center, width / 2);
  const inner = offsetPolyline(center, -width / 2);
  const walls = polylineToSegments(outer).concat(polylineToSegments(inner));
  // start on first centerline point, facing next
  const start = center[0];
  const nx = center[1].x - start.x;
  const ny = center[1].y - start.y;
  const startAngle = Math.atan2(ny, nx);
  const surfaces = (surfaceSpecs || []).map((s) => ({
    x: s.x, y: s.y, r: s.r, type: s.type,
    gripMul: SURFACE_TYPES[s.type].gripMul,
    accelMul: SURFACE_TYPES[s.type].accelMul,
    color: SURFACE_TYPES[s.type].color,
  }));
  return {
    center,
    outer,
    inner,
    walls,
    width,
    surfaces,
    start: { x: start.x, y: start.y },
    startAngle,
  };
}

const TRACKS = [
  // 1) Oval
  buildTrack([
    { x: 200, y: 130 }, { x: 480, y: 120 }, { x: 620, y: 180 },
    { x: 620, y: 340 }, { x: 480, y: 400 }, { x: 200, y: 400 },
    { x: 100, y: 340 }, { x: 100, y: 180 },
  ], 90),

  // 2) Rounded rect with pinch
  buildTrack([
    { x: 150, y: 130 }, { x: 360, y: 110 }, { x: 560, y: 140 },
    { x: 640, y: 220 }, { x: 560, y: 300 }, { x: 400, y: 300 },
    { x: 260, y: 340 }, { x: 130, y: 380 }, { x: 80, y: 260 },
  ], 82),

  // 3) Wavy oval
  buildTrack([
    { x: 120, y: 220 }, { x: 240, y: 140 }, { x: 400, y: 200 },
    { x: 560, y: 140 }, { x: 660, y: 240 }, { x: 620, y: 360 },
    { x: 480, y: 420 }, { x: 340, y: 360 }, { x: 200, y: 420 },
    { x: 90, y: 320 },
  ], 72),

  // 4) Zigzag snake
  buildTrack([
    { x: 100, y: 130 }, { x: 220, y: 150 }, { x: 260, y: 250 },
    { x: 360, y: 260 }, { x: 400, y: 140 }, { x: 520, y: 130 },
    { x: 560, y: 250 }, { x: 640, y: 280 }, { x: 620, y: 400 },
    { x: 460, y: 420 }, { x: 300, y: 400 }, { x: 160, y: 400 },
    { x: 80, y: 280 },
  ], 74),

  // 5) Complex
  buildTrack([
    { x: 140, y: 140 }, { x: 300, y: 100 }, { x: 460, y: 140 },
    { x: 580, y: 100 }, { x: 660, y: 220 }, { x: 580, y: 320 },
    { x: 620, y: 420 }, { x: 480, y: 440 }, { x: 360, y: 380 },
    { x: 240, y: 440 }, { x: 120, y: 400 }, { x: 90, y: 280 },
    { x: 160, y: 220 },
  ], 72),
];

if (typeof module !== "undefined" && module.exports) {
  module.exports = { TRACKS };
}
