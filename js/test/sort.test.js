// The depth sort (core/sort.js) and the sort worker's sort message.
import assert from "node:assert/strict";
import { before, test } from "node:test";
import vm from "node:vm";
import {
    clipWeight, decodeAsog, evaluateCenters, getHalfTable, sortSplats, sortWorkerSource, temporalWeight,
} from "../src/index.js";
import { readRubloff } from "./helpers.js";

/** @type {import("../src/core/decode.js").AsogData} */
let data;
before(async () => {
    data = await decodeAsog(readRubloff(), { audio: false });
});

/** The splats visible at t, the way the GPU modifier weighs them. */
function visibleAt(t) {
    const { count: n, clips } = data;
    const A = data.motion.A, Af = new Float32Array(A.buffer);
    const set = new Set();
    for (let i = 0; i < n; i++) {
        const aw = A[i * 4 + 3], clip = aw >>> 17;
        if (clipWeight(clips, clip, t) <= 0) continue;
        if (!(aw & 0x10000) && temporalWeight(t - clips[clip * 8] - Af[i * 4], Af[i * 4 + 1]) <= 0) continue;
        set.add(i);
    }
    return set;
}

function centersAt(t) {
    const out = new Float32Array(data.count * 3);
    evaluateCenters(data.count, data.centers, data.motion.A, data.motion.B, data.clips, getHalfTable(), t, out);
    return out;
}

test("sortSplats: exactly the visible splats, each once, back to front", () => {
    const n = data.count;
    const row = [0.3, -0.5, 0.81, 2]; // any direction and offset
    for (const t of [0.05, 0.4, 0.93]) {
        const centers = centersAt(t);
        const indices = new Uint32Array(n);
        const count = sortSplats(n, centers, data.motion.A, data.clips, t, row, indices);
        const want = visibleAt(t);
        assert.ok(count > 1000 && count < n, `visible ${count} of ${n} at ${t}`);
        assert.equal(count, want.size);
        const got = new Set(indices.subarray(0, count));
        assert.equal(got.size, count, "duplicate indices");
        for (const i of got) assert.ok(want.has(i), `splat ${i} is not visible at ${t}`);

        // depths never increase by more than one 16-bit key step
        const depth = (i) => row[0] * centers[i * 3] + row[1] * centers[i * 3 + 1] + row[2] * centers[i * 3 + 2] + row[3];
        let min = Infinity, max = -Infinity;
        for (let k = 0; k < count; k++) {
            const d = depth(indices[k]);
            min = Math.min(min, d);
            max = Math.max(max, d);
        }
        const step = (max - min) / 65535;
        for (let k = 1; k < count; k++) assert.ok(depth(indices[k]) <= depth(indices[k - 1]) + 1.0001 * step, `order at ${k}`);
        assert.equal(depth(indices[0]), max);
        assert.equal(depth(indices[count - 1]), min);
    }
});

test("sortSplats: no visible splats outside every clip's active range", () => {
    const n = data.count;
    const end = Math.max(...Array.from({ length: data.clips.length / 8 }, (_, c) => data.clips[c * 8 + 2]));
    const count = sortSplats(n, data.centers, data.motion.A, data.clips, end + 1, [0, 0, 1, 0], new Uint32Array(n));
    assert.equal(count, 0);
});

test("sortSplats margin: also keeps the splats visible within that time, nothing else", () => {
    const n = data.count, t = 0.4, margin = 0.1, row = [0, 0, 1, 0];
    const want = new Set();
    for (let k = -10; k <= 10; k++) for (const i of visibleAt(t + (k / 10) * margin)) want.add(i);
    const indices = new Uint32Array(n);
    const count = sortSplats(n, centersAt(t), data.motion.A, data.clips, t, row, indices, {}, margin);
    const got = new Set(indices.subarray(0, count));
    assert.equal(got.size, count);
    for (const i of want) assert.ok(got.has(i), `splat ${i} visible within the margin is missing`);
    assert.ok(count > visibleAt(t).size, "the margin adds splats");
    assert.ok(count <= visibleAt(t).size * 3, "but not the whole clip");
});

test("the sort worker's sort message gives sortSplats' order on the centres at that time", () => {
    const n = data.count, t = 0.4, row = [0.1, 0.2, -0.97, 3];
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
    for (let k = 0; k < 2; k++) { // the second message reuses the centres evaluated for the first
        self.onmessage({ data: { sort: true, time: t, depthRow: row, indices: new Uint32Array(n).buffer } });
    }
    assert.equal(posted.length, 2);
    const want = new Uint32Array(n);
    const count = sortSplats(n, centersAt(t), data.motion.A, data.clips, t, row, want);
    for (const msg of posted) {
        assert.equal(msg.sort, true);
        assert.equal(msg.time, t);
        assert.equal(msg.count, count);
        assert.deepEqual(new Uint32Array(msg.indices).subarray(0, count), want.subarray(0, count));
    }
});
