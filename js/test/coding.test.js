// Integer coding (spec 3.1), inflate and float16 conversion on synthetic data.
import assert from "node:assert/strict";
import { test } from "node:test";
import { decodeChannel, decodeStream, floatToHalf, halfTable, halfToFloat, inflate } from "../src/index.js";
import { encodeStream, rng } from "./helpers.js";

test("byte planes, zigzag and delta round-trip", async () => {
    const r = rng(7);
    const cases = [
        { range: [0, 255], delta: false },             // 1 plane, unsigned
        { range: [0, 65535], delta: false },           // 2 planes
        { range: [-40000, 40000], delta: false },      // zigzag, 3 planes
        { range: [-3, 3], delta: true },               // delta of a random walk
        { range: [-(2 ** 30), 2 ** 30], delta: true }, // 4+ planes
    ];
    for (const { range: [lo, hi], delta } of cases) {
        const rows = Array.from({ length: 1000 }, () => Array.from({ length: 3 }, () => lo + Math.floor(r() * (hi - lo + 1))));
        if (delta) for (let i = 1; i < rows.length; i++) rows[i] = rows[i].map((v, c) => rows[i - 1][c] + v);
        const { desc, raw, comp } = encodeStream(rows, delta);
        const inflated = await inflate(comp, desc.count * desc.channels * desc.planes);
        assert.deepEqual(inflated, raw);
        const x = decodeStream(desc, inflated);
        for (let i = 0; i < rows.length; i++) {
            for (let c = 0; c < 3; c++) assert.equal(x[i * 3 + c], rows[i][c]);
        }
        const col = new Float64Array(rows.length);
        decodeChannel(desc, inflated, 2, col);
        assert.equal(col[rows.length - 1], rows[rows.length - 1][2]);
    }
});

test("inflate rejects a stream that does not inflate to its declared size", async () => {
    const { desc, comp } = encodeStream([[1, 2, 3], [4, 5, 6]], false);
    const size = desc.count * desc.channels * desc.planes;
    await assert.rejects(inflate(comp, size + 1), /inflates to 6 bytes/);
    await assert.rejects(inflate(comp, size - 1));
});

test("floatToHalf rounds like Math.f16round, clamps beyond the float16 range", () => {
    const table = halfTable();
    const r = rng(3);
    const values = [0, -0, 1, -1, 65504, 65519, 1e-8, 5.960464477539063e-8, 6.1e-5, 0.1, 1 / 3, 2049, 2051];
    for (let i = 0; i < 20000; i++) values.push((r() - 0.5) * 10 ** (r() * 12 - 7));
    for (const v of values) {
        const h = floatToHalf(v);
        assert.equal(Object.is(table[h], -0) ? 0 : table[h], Object.is(Math.f16round(v), -0) ? 0 : Math.f16round(v), `value ${v}`);
    }
    assert.equal(halfToFloat(floatToHalf(1e6)), 65504);
    assert.equal(halfToFloat(floatToHalf(-70000)), -65504);
    assert.equal(halfToFloat(floatToHalf(Infinity)), Infinity);
    assert.ok(Number.isNaN(halfToFloat(floatToHalf(NaN))));
});
