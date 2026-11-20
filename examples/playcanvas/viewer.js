// Animated SOG viewer: PlayCanvas from the CDN (import map), the library from ../../js/src.
import * as pc from "playcanvas";
import { AsogPlayer } from "../../js/src/playcanvas/index.js";

const params = new URLSearchParams(location.search);
const src = params.get("src") ?? "../../test-data/rubloff-6s.asog";
const useWebgpu = params.has("webgpu") ? params.get("webgpu") !== "0" : !!navigator.gpu;
const soundVolume = params.has("volume") ? Number(params.get("volume")) : 1;

const $ = (id) => document.getElementById(id);
const info = $("info"), playButton = $("play"), scrub = $("scrub"), clockLabel = $("clock"), soundButton = $("sound");

// ---------------------------------------------------------------- app
const canvas = $("canvas");
const device = await pc.createGraphicsDevice(canvas, {
    deviceTypes: useWebgpu ? [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2] : [pc.DEVICETYPE_WEBGL2],
    antialias: false,
});
device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
const app = new pc.Application(canvas, { graphicsDevice: device });
app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
app.setCanvasResolution(pc.RESOLUTION_AUTO);
window.addEventListener("resize", () => app.resizeCanvas());
app.start();

const camera = new pc.Entity("camera");
camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0), fov: 50, nearClip: 0.01, farClip: 500 });
camera.addComponent("audiolistener");
app.root.addChild(camera);

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

// ---------------------------------------------------------------- orbit camera
const orbit = { yaw: 0, pitch: -5, distance: 3, target: new pc.Vec3(0, 1, 0), idleSince: 0 };
let drag = null;
canvas.addEventListener("contextmenu", (e) => e.preventDefault());
canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY, button: e.button };
    canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
    drag.x = e.clientX;
    drag.y = e.clientY;
    if (drag.button === 2) {
        orbit.target.y += dy * orbit.distance * 0.0015;
    } else {
        orbit.yaw -= dx * 0.3;
        orbit.pitch = Math.max(-85, Math.min(85, orbit.pitch - dy * 0.3));
    }
    orbit.idleSince = performance.now();
});
canvas.addEventListener("pointerup", () => {
    drag = null;
    orbit.idleSince = performance.now();
});
canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    orbit.distance = Math.max(0.05, Math.min(200, orbit.distance * Math.exp(e.deltaY * 0.001)));
    orbit.idleSince = performance.now();
}, { passive: false });

const camRotation = new pc.Quat();
const camOffset = new pc.Vec3();
function updateCamera(dt) {
    if (!drag && performance.now() - orbit.idleSince > 2500) orbit.yaw += 8 * dt; // slow auto-rotate
    camRotation.setFromEulerAngles(orbit.pitch, orbit.yaw, 0);
    camRotation.transformVector(pc.Vec3.BACK, camOffset).mulScalar(orbit.distance);
    camera.setPosition(camOffset.add(orbit.target));
    camera.setRotation(camRotation);
}

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
const viewer = { app, device: device.isWebGPU ? "WebGPU" : "WebGL 2", clock, orbit, player: null, fps: 0, longTasks: window.asogLongTasks };
window.asogViewer = viewer;

function showInfo() {
    const p = viewer.player;
    const lines = [`${viewer.device}`];
    if (p?.stats) {
        lines[0] = `${p.numSplats.toLocaleString("en")} splats · ${(p.stats.fileBytes / 1e6).toFixed(1)} MB · loaded in ${Math.round(p.stats.loadMs)} ms`
            + ` (decode ${Math.round(p.stats.decodeMs)} ms) · ${fps} fps · ${viewer.device}`;
    }
    info.textContent = lines.join("\n");
}

app.on("update", (dt) => {
    updateCamera(dt);
    frames++;
    const now = performance.now();
    if (now - fpsStart >= 1000) {
        fps = viewer.fps = Math.round(frames * 1000 / (now - fpsStart));
        frames = 0;
        fpsStart = now;
        if (viewer.player?.stats) showInfo();
    }
    const p = viewer.player;
    if (p?.stats) {
        if (!scrubbing) scrub.value = String(p.time);
        clockLabel.textContent = `${fmt(p.time)} / ${fmt(p.duration)} s`;
    }
});

// ---------------------------------------------------------------- load
showInfo();
info.textContent = `Loading ${src} (${viewer.device})…`;
try {
    const player = await AsogPlayer.load(app, src, app.root, {
        getTime: () => clock.now(),
        volume: 0, // muted until "Enable sound"
        onProgress: (p) => (info.textContent = `Loading ${src}: ${Math.round(p * 100)} %`),
    });
    viewer.player = player;
    scrub.max = String(player.duration);
    // frame the clip: its robust bounds, in world space (the player turns it y up)
    const { min, max } = player.bounds;
    const centre = new pc.Vec3((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2);
    player.entity.getWorldTransform().transformPoint(centre, orbit.target);
    const size = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]);
    orbit.distance = size / (2 * Math.tan(25 * Math.PI / 180)) * 1.25;
    if (player.hasAudio) {
        soundButton.hidden = false;
        soundButton.addEventListener("click", async () => {
            const unmute = player.volume === 0;
            if (unmute) await app.systems.sound.context.resume();
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
