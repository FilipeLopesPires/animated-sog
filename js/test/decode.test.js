// Decoding test-data/rubloff-1s.asog: container, streams, GPU layouts, and the dequantisation
// against the spec's formulas (docs/format.md 3.3, 4.2), re-derived here from the raw stream integers.
import assert from "node:assert/strict";
import { before, test } from "node:test";
import { inflateSync } from "node:zlib";
import {
    HEADER_SIZE, MOTION_B_TEXELS, MOTION_TEX_WIDTH, centerBounds, decodeAsog, decodeStream, getHalfTable,
    openAsog, parseHeader, parseManifest, readSplat,
} from "../src/index.js";
import { readRubloff } from "./helpers.js";

const bytes = readRubloff();
const header = parseHeader(bytes.subarray(0, HEADER_SIZE));
const manifest = parseManifest(bytes.subarray(header.manifestOffset, header.manifestOffset + header.manifestLength));
const n = manifest.count;
const half = getHalfTable();
/** @type {Record<string, Float64Array>} raw stream integers, row-major */
const S = {};
/** @type {import("../src/core/decode.js").AsogData} */
let data;

before(async () => {
    for (const s of manifest.streams) {
        const start = header.payloadOffset + s.byteOffset;
        S[s.name] = decodeStream(s, new Uint8Array(inflateSync(bytes.subarray(start, start + s.byteLength))));
    }
    data = await decodeAsog(bytes);
});

/** |got - want| within `ulps` float16 steps of want (plus the smallest subnormal). */
function closeHalf(got, want, ulps = 1) {
    const exp = Math.floor(Math.log2(Math.max(Math.abs(want), 2 ** -14)));
    return Math.abs(got - want) <= ulps * 2 ** (exp - 10) + 2 ** -24;
}

test("container and manifest", () => {
    assert.equal(String.fromCharCode(...bytes.subarray(0, 4)), "ASOG");
    assert.equal(header.manifestOffset, 64);
    assert.equal(header.payloadOffset % 64, 0);
    assert.ok(header.payloadOffset >= header.manifestOffset + header.manifestLength);
    assert.equal(header.payloadOffset + header.payloadLength, bytes.length);
    assert.deepEqual([...bytes.subarray(4, 16)], new Array(12).fill(0));
    assert.deepEqual([...bytes.subarray(48, 64)], new Array(16).fill(0));
    assert.equal(manifest.format, "asog");
    assert.equal(manifest.version, 1);
    assert.equal(n, 105000);
    assert.equal(manifest.timeline.durationSeconds, 1);
    assert.equal(manifest.audio, undefined);
    assert.equal(manifest.scales.codebook.length, 256);
    assert.equal(manifest.sh0.codebook.length, 256);
    assert.equal(manifest.shN.codebook.length, 256);
});

test("streams: in payload order, inside the payload, each inflates to its declared size", () => {
    const names = manifest.streams.map((s) => s.name);
    assert.deepEqual(names, ["means", "quats", "quatmode", "scales", "sh0", "opacity", "shlabels", "shpalette",
        "mu", "logsigma", "anchor", "omega", "g", "flags"]);
    let end = 0;
    for (const s of manifest.streams) {
        assert.ok(s.byteOffset >= end, `${s.name} overlaps the stream before it`);
        end = s.byteOffset + s.byteLength;
        const start = header.payloadOffset + s.byteOffset;
        const raw = inflateSync(bytes.subarray(start, start + s.byteLength));
        assert.equal(raw.length, s.count * s.channels * s.planes, s.name);
        assert.equal(s.count, s.name === "shpalette" ? manifest.shN.count : n, s.name);
    }
    assert.ok(end <= header.payloadLength);
});

test("decodeAsog: counts and array sizes", () => {
    assert.equal(data.count, n);
    assert.equal(data.shBands, 3);
    assert.deepEqual([data.sog.width, data.sog.height], [325, 324]);
    for (const k of ["means_l", "means_u", "quats", "scales", "sh0", "sh_labels"]) assert.equal(data.sog[k].length, 325 * 324 * 4, k);
    assert.deepEqual([data.sog.shWidth, data.sog.shHeight], [64 * 15, 4096 / 64]);
    assert.equal(data.sog.sh_centroids.length, 960 * 64 * 4);
    assert.equal(data.motion.A.length, MOTION_TEX_WIDTH * Math.ceil(n / MOTION_TEX_WIDTH) * 4);
    assert.equal(data.motion.B.length, MOTION_TEX_WIDTH * Math.ceil(n * MOTION_B_TEXELS / MOTION_TEX_WIDTH) * 4);
    assert.equal(data.centers.length, n * 3);
    assert.equal(data.clips.length, 8);
    assert.equal(data.stats.fileBytes, bytes.length);
    assert.equal(data.stats.halfOverflow, 0);
    assert.equal(data.audio, null);
});

test("SOG v2 arrays hold the stream integers", () => {
    const { sog } = data;
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const q = S.means[i * 3 + c];
            assert.ok(q >= 0 && q <= 65535);
            if (sog.means_l[i * 4 + c] + 256 * sog.means_u[i * 4 + c] !== q) assert.fail(`means ${i}`);
            if (sog.quats[i * 4 + c] !== S.quats[i * 3 + c]) assert.fail(`quats ${i}`);
            if (sog.scales[i * 4 + c] !== S.scales[i * 3 + c]) assert.fail(`scales ${i}`);
            if (sog.sh0[i * 4 + c] !== S.sh0[i * 3 + c]) assert.fail(`sh0 ${i}`);
        }
        if (sog.quats[i * 4 + 3] !== 252 + S.quatmode[i]) assert.fail(`quatmode ${i}`);
        if (sog.sh0[i * 4 + 3] !== S.opacity[i]) assert.fail(`opacity ${i}`);
        if (sog.sh_labels[i * 4] + 256 * sog.sh_labels[i * 4 + 1] !== S.shlabels[i]) assert.fail(`shlabels ${i}`);
    }
    const { count, coeffs } = manifest.shN;
    for (let k = 0; k < count; k += 37) {
        for (let j = 0; j < coeffs; j++) {
            for (let c = 0; c < 3; c++) {
                const texel = (Math.floor(k / 64) * data.sog.shWidth + (k % 64) * coeffs + j) * 4 + c;
                assert.equal(data.sog.sh_centroids[texel], S.shpalette[k * coeffs * 3 + j * 3 + c]);
            }
        }
    }
});

test("value ranges", () => {
    for (let i = 0; i < n; i++) {
        if (S.quatmode[i] < 0 || S.quatmode[i] > 3) assert.fail(`quatmode ${S.quatmode[i]}`);
        if (S.opacity[i] < 0 || S.opacity[i] > 255) assert.fail(`opacity ${S.opacity[i]}`);
        if (S.shlabels[i] < 0 || S.shlabels[i] >= manifest.shN.count) assert.fail(`shlabel ${S.shlabels[i]}`);
        if (S.logsigma[i] < 0 || S.logsigma[i] > 2 ** manifest.logsigma.bits - 1) assert.fail(`logsigma ${S.logsigma[i]}`);
        if (S.flags[i] >> 1 >= manifest.clips.length) assert.fail(`clip of ${i}`);
    }
    for (const name of ["scales", "sh0", "shpalette"]) {
        for (const q of S[name]) if (q < 0 || q > 255) assert.fail(`${name} ${q}`);
    }
    for (const name of ["scales", "sh0", "shN"]) {
        const cb = manifest[name].codebook;
        for (let k = 1; k < 256; k++) assert.ok(cb[k] >= cb[k - 1], `${name} codebook sorted`);
    }
    // centres inside the means box, mapped
    const map = (v) => Math.sign(v) * Math.expm1(Math.abs(v));
    for (let i = 0; i < n; i++) {
        for (let c = 0; c < 3; c++) {
            const x = data.centers[i * 3 + c];
            if (x < map(manifest.means.mins[c]) - 1e-6 || x > map(manifest.means.maxs[c]) + 1e-6) assert.fail(`centre ${i}`);
        }
    }
    // a person, y down: ~1.7 m tall, ~0.6 m wide
    const b = centerBounds(data.centers);
    assert.ok(b.max[1] - b.min[1] > 1.4 && b.max[1] - b.min[1] < 2.0, `height ${b.max[1] - b.min[1]}`);
    assert.ok(b.max[1] < 0.1, "y down: the person is at negative y");
    // no NaN / infinity in the motion
    for (const h of data.motion.B) if ((h & 0x7c00) === 0x7c00) assert.fail("non-finite half in motionB");
});

test("static attributes match spec 3.3", () => {
    const { mins, maxs } = manifest.means;
    for (let i = 0; i < n; i += 13) {
        const s = readSplat(data, i);
        for (let c = 0; c < 3; c++) {
            const v = mins[c] + S.means[i * 3 + c] / 65535 * (maxs[c] - mins[c]);
            const x = Math.sign(v) * (Math.exp(Math.abs(v)) - 1);
            assert.ok(Math.abs(s.position[c] - x) < 1e-12);
            assert.ok(Math.abs(data.centers[i * 3 + c] - x) <= 1e-6 * Math.max(1, Math.abs(x)));
            assert.equal(s.scale[c], Math.exp(manifest.scales.codebook[S.scales[i * 3 + c]]));
            assert.ok(Math.abs(s.color[c] - (0.5 + 0.28209479 * manifest.sh0.codebook[S.sh0[i * 3 + c]])) < 1e-8);
        }
        // rotation: r_k = (q / 255 - 0.5) sqrt 2 in w x y z order without the largest; largest = sqrt(1 - sum r^2)
        const r = [0, 1, 2].map((c) => (S.quats[i * 3 + c] / 255 - 0.5) * Math.SQRT2);
        const mode = S.quatmode[i];
        const big = Math.sqrt(Math.max(0, 1 - r[0] ** 2 - r[1] ** 2 - r[2] ** 2));
        const wxyz = [];
        for (let k = 0, j = 0; k < 4; k++) wxyz.push(k === mode ? big : r[j++]);
        const [w, x, y, z] = wxyz;
        assert.deepEqual(s.rotation.map((v) => +v.toFixed(12)), [x, y, z, w].map((v) => +v.toFixed(12)));
        assert.ok(Math.abs(Math.hypot(...s.rotation) - 1) < 0.02, "unit quaternion");
        assert.ok(big >= Math.max(...r.map(Math.abs)) - 0.01, "the left-out component is the largest");
        assert.equal(s.opacity, S.opacity[i] / 255);
        const label = S.shlabels[i], K = manifest.shN.coeffs;
        for (let j = 0; j < K; j++) {
            for (let c = 0; c < 3; c++) assert.equal(s.sh[j * 3 + c], Math.fround(manifest.shN.codebook[S.shpalette[label * K * 3 + j * 3 + c]]));
        }
    }
});

test("motion matches spec 4.2 (to float16)", () => {
    const { A, B } = data.motion;
    const Af = new Float32Array(A.buffer);
    const mo = manifest.motion, ls = manifest.logsigma, qpv = mo.qp / 1000;
    const wO = [1, mo.midO, 1], wG = [1, mo.midG, mo.midG, mo.midG, 1];
    const { mins, maxs } = manifest.means;
    let gRelVsExact = 0;
    for (let i = 0; i < n; i++) {
        const mu = S.mu[i] / manifest.mu.scale;
        const sigma = Math.exp(ls.min + S.logsigma[i] / (2 ** ls.bits - 1) * (ls.max - ls.min));
        if (Af[i * 4] !== Math.fround(mu) || Af[i * 4 + 1] !== Math.fround(sigma)) assert.fail(`mu/sigma ${i}`);
        const L = mo.sm * sigma;
        const a = [0, 1, 2].map((c) => S.anchor[i * 3 + c] * mo.qa / 1000);
        const x = [0, 1, 2].map((c) => {
            const v = mins[c] + S.means[i * 3 + c] / 65535 * (maxs[c] - mins[c]);
            return Math.sign(v) * Math.expm1(Math.abs(v));
        });
        const ah = [half[A[i * 4 + 2] & 0xffff], half[A[i * 4 + 2] >>> 16], half[A[i * 4 + 3] & 0xffff]];
        for (let c = 0; c < 3; c++) if (!closeHalf(ah[c], a[c], 0.5)) assert.fail(`anchor ${i}`);
        const flags = S.flags[i];
        if (((A[i * 4 + 3] >>> 16) & 1) !== (flags & 1) || A[i * 4 + 3] >>> 17 !== flags >> 1) assert.fail(`flags ${i}`);

        const stepO = qpv / Math.max(Math.hypot(x[0] - a[0], x[1] - a[1], x[2] - a[2]), mo.r_min);
        const O = [], Oh = [];
        for (let k = 0; k < 9; k++) {
            O.push(S.omega[i * 9 + k] * stepO * wO[Math.floor(k / 3)] / L);
            Oh.push(half[B[i * 16 + k]]);
            if (!closeHalf(Oh[k], O[k])) assert.fail(`omega ${i}.${k}: ${Oh[k]} vs ${O[k]}`);
        }
        // m_ref from the stored control points (spec 4.2 note), and from the exact ones for comparison
        const mref = (C) => {
            let m = Infinity;
            for (let s = 0; s < mo.mref_samples; s++) {
                const tau = s / (mo.mref_samples - 1), u = 1 - tau, b = [u * u, 2 * u * tau, tau * tau];
                let q = 0;
                for (let c = 0; c < 3; c++) q += (b[0] * C[c] * L + b[1] * C[3 + c] * L + b[2] * C[6 + c] * L) ** 2;
                m = Math.min(m, Math.sqrt(q));
            }
            return Math.max(m, 1e-9) * qpv;
        };
        const stepG = mref(Oh), stepGExact = mref(O);
        for (let j = 0; j < 5; j++) {
            const G = S.g[i * 5 + j] * stepG * wG[j] / (L * L);
            const got = half[B[i * 16 + 9 + j]];
            if (!closeHalf(got, G)) assert.fail(`g ${i}.${j}: ${got} vs ${G}`);
            const exact = S.g[i * 5 + j] * stepGExact * wG[j] / (L * L);
            if (exact !== 0) gRelVsExact = Math.max(gRelVsExact, Math.abs(G - exact) / Math.abs(exact));
        }
        if (B[i * 16 + 14] !== 0 || B[i * 16 + 15] !== 0) assert.fail("unused halves");
    }
    // m_ref from float16 vs exact control points: a small relative change of G
    assert.ok(gRelVsExact < 0.05, `G relative change from float16 m_ref: ${gRelVsExact}`);
});

test("streamed decode (Response body) equals the in-memory decode; keep and onArray", async () => {
    const emitted = [];
    const streamed = await decodeAsog(new Response(new Blob([bytes])), {
        keep: ["motionB"],
        onArray: (name, array, info) => { emitted.push([name, array.length, info.format, info.width, info.height]); },
    });
    assert.deepEqual(emitted.map((e) => e[0]), ["means_l", "means_u", "quats", "scales", "sh0", "sh_labels", "sh_centroids",
        "centers", "motionB", "motionA"]);
    assert.deepEqual(emitted.find((e) => e[0] === "motionA").slice(2), ["rgba32u", 4096, 26]);
    assert.deepEqual(streamed.motion.B, data.motion.B);
    assert.equal(streamed.motion.A, undefined);
    assert.equal(streamed.sog.means_l, undefined);
    assert.equal(streamed.centers, null);
});

test("errors: bad magic, truncated file, abort", async () => {
    const bad = bytes.slice(0, 1000);
    bad[0] = 0x58;
    await assert.rejects(openAsog(bad), /not an .asog file/);
    await assert.rejects(decodeAsog(bytes.slice(0, bytes.length - 100)), /ends early/);
    const ctrl = new AbortController();
    ctrl.abort(new Error("stop"));
    await assert.rejects(decodeAsog(bytes, { signal: ctrl.signal }), /stop/);
});
