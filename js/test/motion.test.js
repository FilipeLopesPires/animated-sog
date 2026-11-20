// The motion reference (spec 4.1), clip timing (spec 5) and the sort worker built from it.
import assert from "node:assert/strict";
import { before, test } from "node:test";
import vm from "node:vm";
import {
    clipWeight, createMotion, decodeAsog, evaluateCenters, evaluateSplat, evaluateSplatAt, getHalfTable,
    quatMultiply, sortWorkerSource, temporalWeight, TEMPORAL_THRESHOLD,
} from "../src/index.js";
import { readRubloff, rng } from "./helpers.js";

/** v rotated by the unit quaternion q (x y z w): q v q* */
function rotate(q, v) {
    const p = quatMultiply(quatMultiply(q, [v[0], v[1], v[2], 0]), [-q[0], -q[1], -q[2], q[3]]);
    return [p[0], p[1], p[2]];
}

function randomMotion(r, scale = 1) {
    const m = createMotion();
    m.x = r() - 0.5; m.y = r() - 0.5; m.z = r() - 0.5;
    m.mu = 0.5; m.sigma = 0.2;
    for (let c = 0; c < 3; c++) m.anchor[c] = r() - 0.5;
    for (let k = 0; k < 9; k++) m.omega[k] = (r() - 0.5) * 6 * scale;
    for (let j = 0; j < 5; j++) m.g[j] = (r() - 0.5) * 4 * scale;
    return m;
}

test("temporal weight: 1 at the centre time, 0 where the Gaussian falls under the threshold", () => {
    assert.equal(temporalWeight(0, 0.3), 1);
    const cut = Math.sqrt(-2 * Math.log(TEMPORAL_THRESHOLD)); // ~2.448 sigma, the encoder's sm = 2.45
    assert.ok(temporalWeight(0.99 * cut * 0.3, 0.3) > 0);
    assert.equal(temporalWeight(1.01 * cut * 0.3, 0.3), 0);
});

test("evaluateSplat: at t = mu the splat is at its canonical pose", () => {
    const r = rng(11);
    const out = new Float64Array(7);
    for (let i = 0; i < 100; i++) {
        const m = randomMotion(r);
        assert.equal(evaluateSplat(m, m.mu, out), 1);
        [m.x, m.y, m.z, 0, 0, 0, 1].forEach((v, k) => assert.ok(Math.abs(out[k] - v) < 1e-15));
    }
});

test("evaluateSplat: the rotation q(w) turns the splat the same way its centre moves (also in the small-angle branch)", () => {
    const r = rng(5);
    const out = new Float64Array(7);
    for (let i = 0; i < 1000; i++) {
        const m = randomMotion(r, i % 2 ? 1 : 1e-7);
        m.g.fill(0); // pure rotation about the axis through the projected anchor
        const t = m.mu + (r() - 0.5) * 0.8;
        if (evaluateSplat(m, t, out) <= 0) continue;
        // the axis point a_perp: anchor minus its component along omega(tau)
        const tau = Math.min(1, Math.max(0, (t - (m.mu - 2 * m.sigma)) / (4 * m.sigma))), u = 1 - tau;
        const om = [0, 1, 2].map((c) => u * u * m.omega[c] + 2 * u * tau * m.omega[3 + c] + tau * tau * m.omega[6 + c]);
        const oo = om[0] ** 2 + om[1] ** 2 + om[2] ** 2;
        const ad = (m.anchor[0] * om[0] + m.anchor[1] * om[1] + m.anchor[2] * om[2]) / oo;
        const ap = [0, 1, 2].map((c) => m.anchor[c] - ad * om[c]);
        const q = [out[3], out[4], out[5], out[6]];
        assert.ok(Math.abs(Math.hypot(...q) - 1) < 1e-12);
        const want = rotate(q, [m.x - ap[0], m.y - ap[1], m.z - ap[2]]).map((v, c) => v + ap[c]);
        for (let c = 0; c < 3; c++) assert.ok(Math.abs(out[c] - want[c]) < 1e-9, `centre ${c}: ${out[c]} vs ${want[c]}`);
    }
});

test("evaluateSplat: the screw's translation runs along omega", () => {
    const out = new Float64Array(7);
    const m = createMotion();
    m.mu = 0; m.sigma = 1;
    m.omega.set([0, 0, 1, 0, 0, 1, 0, 0, 1]);  // constant omega = z
    m.g.fill(2);                                // nu . omega = 2  ->  nu_par = 2 z
    m.x = 1;
    evaluateSplat(m, 0.5, out);                 // w = 0.5 rad about z through the origin, v = (0, 0, 1)
    assert.ok(Math.abs(out[0] - Math.cos(0.5)) < 1e-12 && Math.abs(out[1] - Math.sin(0.5)) < 1e-12 && Math.abs(out[2] - 1) < 1e-12);
});

test("clip weight: active range and smoothstep fades", () => {
    // offset 0, active [1, 5), core [2, 4), fade in 1 s, fade out 0.5 s
    const clips = new Float32Array([0, 1, 5, 0, 2, 4, 1, 0.5]);
    assert.equal(clipWeight(clips, 0, 0.99), 0);
    assert.equal(clipWeight(clips, 0, 1), 0);
    assert.equal(clipWeight(clips, 0, 1.5), 0.5);
    assert.equal(clipWeight(clips, 0, 3), 1);
    assert.equal(clipWeight(clips, 0, 4.25), 0.5);
    assert.equal(clipWeight(clips, 0, 4.6), 0);
    assert.equal(clipWeight(clips, 0, 5), 0);
});

/** @type {import("../src/core/decode.js").AsogData} */
let data;
before(async () => {
    data = await decodeAsog(readRubloff());
});

test("rubloff: evaluateCenters equals per-splat evaluation; the person moves but stays together", () => {
    const n = data.count;
    const out = new Float32Array(n * 3);
    for (const t of [0.05, 0.5, 0.95]) {
        evaluateCenters(n, data.centers, data.motion.A, data.motion.B, data.clips, getHalfTable(), t, out);
        const d = [];
        let visible = 0;
        for (let i = 0; i < n; i++) {
            const dx = out[i * 3] - data.centers[i * 3], dy = out[i * 3 + 1] - data.centers[i * 3 + 1], dz = out[i * 3 + 2] - data.centers[i * 3 + 2];
            d.push(Math.hypot(dx, dy, dz));
            if (i % 101 === 0) {
                const e = evaluateSplatAt(data, i, t);
                if (e.weight > 0) {
                    visible++;
                    for (let c = 0; c < 3; c++) assert.ok(Math.abs(e.center[c] - out[i * 3 + c]) < 1e-5);
                    assert.ok(Math.abs(Math.hypot(...e.rotation) - 1) < 0.02);
                }
            }
        }
        d.sort((a, b) => a - b);
        assert.ok(visible > 100, `visible splats at ${t}`);
        assert.ok(!d.some(Number.isNaN), "NaN centres");
        assert.ok(d[Math.floor(n * 0.99)] < 0.1, `p99 displacement ${d[Math.floor(n * 0.99)]} m at t=${t}`);
        assert.ok(d[n - 1] < 0.5, `max displacement ${d[n - 1]} m at t=${t}`);
    }
});

test("the sort worker source runs and gives evaluateCenters' result", async () => {
    const n = data.count, t = 0.4;
    const posted = [];
    const self = { onmessage: null, postMessage: (msg) => posted.push(msg) };
    vm.runInNewContext(sortWorkerSource(), { self, Float32Array, Float64Array, Uint32Array, Uint16Array, Math });
    self.onmessage({
        data: {
            init: true, n,
            centers: data.centers.slice().buffer, A: data.motion.A.slice().buffer,
            B: data.motion.B.slice().buffer, clips: data.clips.slice().buffer,
        },
    });
    self.onmessage({ data: { time: t, out: new Float32Array(n * 3).buffer } });
    assert.equal(posted.length, 1);
    assert.equal(posted[0].time, t);
    const want = new Float32Array(n * 3);
    evaluateCenters(n, data.centers, data.motion.A, data.motion.B, data.clips, getHalfTable(), t, want);
    assert.deepEqual(new Float32Array(posted[0].out), want);
});
