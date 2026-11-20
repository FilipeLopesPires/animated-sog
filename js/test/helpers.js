import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { deflateSync } from "node:zlib";

export const RUBLOFF = fileURLToPath(new URL("../../test-data/rubloff-1s.asog", import.meta.url));

/** @returns {Uint8Array} */
export function readRubloff() {
    return new Uint8Array(readFileSync(RUBLOFF));
}

/**
 * Encodes integers the way the reference encoder does (encode.py `add`): optional delta, zigzag
 * if any value is negative, byte planes plane-major then channel-major, zlib.
 * @param {number[][]} rows - count × channels
 * @param {boolean} delta
 */
export function encodeStream(rows, delta) {
    const count = rows.length, channels = rows[0].length;
    const x = rows.map((r, i) => r.map((v, c) => (delta && i > 0 ? v - rows[i - 1][c] : v)));
    const signed = x.some((r) => r.some((v) => v < 0));
    const z = x.map((r) => r.map((v) => (signed ? (v < 0 ? -2 * v - 1 : 2 * v) : v)));
    let max = 0;
    for (const r of z) for (const v of r) max = Math.max(max, v);
    const planes = Math.max(1, Math.ceil(Math.log2(max + 1) / 8));
    const raw = new Uint8Array(count * channels * planes);
    for (let p = 0; p < planes; p++) {
        for (let c = 0; c < channels; c++) {
            for (let i = 0; i < count; i++) raw[(p * channels + c) * count + i] = Math.floor(z[i][c] / 256 ** p) % 256;
        }
    }
    const desc = { name: "test", count, channels, planes, delta, zigzag: signed, byteOffset: 0, byteLength: 0 };
    const comp = new Uint8Array(deflateSync(raw));
    desc.byteLength = comp.length;
    return { desc, raw, comp };
}

/** Deterministic pseudo-random numbers in [0, 1). */
export function rng(seed = 1) {
    let s = seed >>> 0;
    return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 4294967296;
    };
}
