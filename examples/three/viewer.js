// Animated SOG viewer: three.js from the CDN (import map), the library from ../../js/src.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AsogThreePlayer } from "../../js/src/three/index.js";

const params = new URLSearchParams(location.search);
const src = params.get("src") ?? "../../test-data/rubloff-1s.asog";
const soundVolume = params.has("volume") ? Number(params.get("volume")) : 1;

const $ = (id) => document.getElementById(id);
const info = $("info"), playButton = $("play"), scrub = $("scrub"), clockLabel = $("clock"), soundButton = $("sound");

// ---------------------------------------------------------------- renderer
const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.setClearColor(0x000000, 1);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.01, 500);
scene.add(camera);
window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// ---------------------------------------------------------------- clock: play / pause / scrub
const clock = {
    playing: true,
    base: 0,
    start: performance.now() / 1000,
    now() {
        return this.playing ? this.base + performance.now() / 1000 - this.start : this.base;
    },
    set(t) {
        this.base = t;
        this.start = performance.now() / 1000;
    },
};

// ---------------------------------------------------------------- orbit camera: left orbit, right pan, wheel zoom
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = false;
controls.autoRotateSpeed = 4 / 3; // 8° a second, as the PlayCanvas viewer
controls.minDistance = 0.05;
controls.maxDistance = 200;
let interacting = false, idleSince = 0;
controls.addEventListener("start", () => (interacting = true));
controls.addEventListener("end", () => {
    interacting = false;
    idleSince = performance.now();
});
canvas.addEventListener("wheel", () => (idleSince = performance.now()), { passive: true });
camera.position.set(0, 1, 3);
controls.target.set(0, 1, 0);

// ---------------------------------------------------------------- UI
const fmt = (t) => t.toFixed(2);
let scrubbing = false;
playButton.addEventListener("click", () => {
    const t = clock.now();
    clock.playing = !clock.playing;
    clock.set(t);
    playButton.textContent = clock.playing ? "Pause" : "Play";
});
scrub.addEventListener("pointerdown", () => (scrubbing = true));
scrub.addEventListener("pointerup", () => (scrubbing = false));
scrub.addEventListener("input", () => clock.set(Number(scrub.value)));

let fps = 0, frames = 0, fpsStart = performance.now();
const viewer = { renderer, scene, camera, controls, device: "WebGL 2", clock, player: null, fps: 0, longTasks: window.asogLongTasks };
window.asogViewer = viewer;

function showInfo() {
    const p = viewer.player;
    if (!p?.stats) return;
    info.textContent = `${p.numSplats.toLocaleString("en")} splats · ${(p.stats.fileBytes / 1e6).toFixed(1)} MB · loaded in ${Math.round(p.stats.loadMs)} ms`
        + ` (decode ${Math.round(p.stats.decodeMs)} ms) · ${fps} fps · three.js ${THREE.REVISION} ${viewer.device}`;
}

const timer = new THREE.Timer();
renderer.setAnimationLoop((timestamp) => {
    timer.update(timestamp);
    controls.autoRotate = !interacting && performance.now() - idleSince > 2500;
    controls.update(timer.getDelta());
    renderer.render(scene, camera);
    frames++;
    const now = performance.now();
    if (now - fpsStart >= 1000) {
        fps = viewer.fps = Math.round(frames * 1000 / (now - fpsStart));
        frames = 0;
        fpsStart = now;
        showInfo();
    }
    const p = viewer.player;
    if (p?.stats) {
        if (!scrubbing) scrub.value = String(p.time);
        clockLabel.textContent = `${fmt(p.time)} / ${fmt(p.duration)} s`;
    }
});

// ---------------------------------------------------------------- load
info.textContent = `Loading ${src} (${viewer.device})…`;
try {
    const player = await AsogThreePlayer.load(renderer, src, scene, {
        getTime: () => clock.now(),
        volume: 0, // muted until "Enable sound"
        onProgress: (p) => (info.textContent = `Loading ${src}: ${Math.round(p * 100)} %`),
    });
    viewer.player = player;
    scrub.max = String(player.duration);
    // frame the clip: its robust bounds, in world space (the player turns it y up)
    const { min, max } = player.bounds;
    const centre = new THREE.Vector3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    player.object.updateMatrixWorld();
    centre.applyMatrix4(player.object.matrixWorld);
    const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    const distance = size / (2 * Math.tan(25 * Math.PI / 180)) * 1.25;
    controls.target.copy(centre);
    // yaw 0, pitch -5°: slightly from above, looking along -z (as the PlayCanvas viewer)
    camera.position.copy(centre).add(new THREE.Vector3(0, Math.sin(5 * Math.PI / 180), Math.cos(5 * Math.PI / 180)).multiplyScalar(distance));
    controls.update();
    if (player.hasAudio) {
        soundButton.hidden = false;
        soundButton.addEventListener("click", async () => {
            const unmute = player.volume === 0;
            if (unmute) await player.listener.context.resume();
            player.volume = unmute ? soundVolume : 0;
            soundButton.textContent = unmute ? "Mute" : "Enable sound";
        });
    }
    if (player.audioError) console.error("soundtrack:", player.audioError);
    showInfo();
} catch (e) {
    info.innerHTML = "";
    const err = document.createElement("span");
    err.id = "error";
    err.textContent = `Could not load ${src}: ${e.message}`;
    info.append(err);
    throw e;
}
