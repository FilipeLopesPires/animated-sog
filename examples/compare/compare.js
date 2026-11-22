// Animated SOG compare: two clips, one camera and one clock, split by a line. Each clip runs in its own PlayCanvas
// app on its own canvas; the top canvas (B) is clipped at the line, so A shows left of it and B right of it.
import * as pc from "playcanvas";
import { AsogPlayer } from "../../js/src/playcanvas/index.js";

const params = new URLSearchParams(location.search);
const srcA = params.get("a") ?? "../../test-data/rubloff-6s.asog";
const srcB = params.get("b") ?? "https://ugc.arrival.space/56323439/rubloff-ultra.asog";
const titleA = params.get("la") ?? (params.has("a") ? fileName(srcA) : "Before");
const titleB = params.get("lb") ?? (params.has("b") ? fileName(srcB) : "After");
const useWebgpu = params.has("webgpu") ? params.get("webgpu") !== "0" : !!navigator.gpu;
const soundVolume = params.has("volume") ? Number(params.get("volume")) : 1;

const $ = (id) => document.getElementById(id);
const touch = $("touch"), line = $("line"), canvasB = $("canvas-b");
const playButton = $("play"), speedButton = $("speed"), scrub = $("scrub"), clockLabel = $("clock"), soundButton = $("sound");
const loader = $("loader"), loaderPct = $("loader-pct"), loaderFill = $("loader-fill");

function fileName(src) {
    return decodeURIComponent(src.split("/").pop().replace(/\.asog$/i, ""));
}

// ---------------------------------------------------------------- shared clock and camera
const clock = {
    playing: true,
    rate: 1, // slow motion: 0.5, 0.25
    base: 0,
    start: performance.now() / 1000,
    now() {
        return this.playing ? this.base + (performance.now() / 1000 - this.start) * this.rate : this.base;
    },
    set(t) {
        this.base = t;
        this.start = performance.now() / 1000;
    },
    setRate(rate) {
        const t = this.now();
        this.rate = rate;
        this.set(t);
    },
};

// zoom 0 = the whole performer, 1 = the face; the target and distance follow it
const view = { yaw: 0, pitch: -4, zoom: 0, body: new pc.Vec3(0, 1, 0), head: new pc.Vec3(0, 1.6, 0), far: 3, near: 0.4 };
const camTarget = new pc.Vec3(), camRotation = new pc.Quat(), camOffset = new pc.Vec3();

function placeCamera(camera) {
    const z = Math.min(Math.max(view.zoom, 0), 1);
    const ease = z * z * (3 - 2 * z);
    camTarget.lerp(view.body, view.head, ease);
    const distance = Math.exp(Math.log(view.far) + (Math.log(view.near) - Math.log(view.far)) * z);
    camRotation.setFromEulerAngles(view.pitch, view.yaw, 0);
    camRotation.transformVector(pc.Vec3.BACK, camOffset).mulScalar(distance);
    camera.setPosition(camOffset.add(camTarget));
    camera.setRotation(camRotation);
}

// ---------------------------------------------------------------- two apps
async function makeApp(canvas) {
    const device = await pc.createGraphicsDevice(canvas, {
        deviceTypes: useWebgpu ? [pc.DEVICETYPE_WEBGPU, pc.DEVICETYPE_WEBGL2] : [pc.DEVICETYPE_WEBGL2],
        antialias: false,
    });
    device.maxPixelRatio = Math.min(window.devicePixelRatio, 2);
    const app = new pc.Application(canvas, { graphicsDevice: device });
    app.setCanvasFillMode(pc.FILLMODE_FILL_WINDOW);
    app.setCanvasResolution(pc.RESOLUTION_AUTO);
    window.addEventListener("resize", () => app.resizeCanvas());
    const camera = new pc.Entity("camera", app);
    camera.addComponent("camera", { clearColor: new pc.Color(0, 0, 0), fov: 50, nearClip: 0.01, farClip: 500 });
    app.root.addChild(camera);
    app.on("update", () => placeCamera(camera));
    app.start();
    return { app, camera, device: device.isWebGPU ? "WebGPU" : "WebGL 2" };
}

const A = await makeApp($("canvas-a"));
const B = await makeApp(canvasB);
B.camera.addComponent("audiolistener");

// ---------------------------------------------------------------- the line
let split = 0.5;
function setSplit(x) {
    split = Math.min(Math.max(x / window.innerWidth, 0), 1);
    const pct = `${(split * 100).toFixed(3)}%`;
    line.style.left = pct;
    canvasB.style.clipPath = `inset(0 0 0 ${pct})`;
}

// ---------------------------------------------------------------- input: mouse moves the line, drag turns, wheel / pinch zooms
const pointers = new Map();
let turning = null, lineDrag = false, pinch = null;
touch.addEventListener("contextmenu", (e) => e.preventDefault());
touch.addEventListener("pointerdown", (e) => {
    touch.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pointers.size === 2) {
        const [p, q] = [...pointers.values()];
        pinch = { d: Math.hypot(p.x - q.x, p.y - q.y), zoom: view.zoom };
        turning = null;
        lineDrag = false;
        return;
    }
    // touch / pen: grabbing near the line moves it, anywhere else turns; a mouse always turns (the line follows hover)
    if (e.pointerType !== "mouse" && Math.abs(e.clientX - split * window.innerWidth) < 40) lineDrag = true;
    else {
        turning = { x: e.clientX, y: e.clientY };
        touch.classList.add("turning");
    }
});
touch.addEventListener("pointermove", (e) => {
    if (pointers.has(e.pointerId)) pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (pinch && pointers.size === 2) {
        const [p, q] = [...pointers.values()];
        view.zoom = Math.min(Math.max(pinch.zoom + Math.log(Math.hypot(p.x - q.x, p.y - q.y) / pinch.d) * 0.8, 0), 1);
    } else if (turning) {
        view.yaw -= (e.clientX - turning.x) * 0.3;
        view.pitch = Math.max(-80, Math.min(80, view.pitch - (e.clientY - turning.y) * 0.3));
        turning = { x: e.clientX, y: e.clientY };
    } else if (lineDrag || (e.pointerType === "mouse" && e.buttons === 0)) {
        setSplit(e.clientX);
    }
});
function release(e) {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = null;
    if (pointers.size === 0) {
        turning = null;
        lineDrag = false;
        touch.classList.remove("turning");
    }
}
touch.addEventListener("pointerup", release);
touch.addEventListener("pointercancel", release);
touch.addEventListener("wheel", (e) => {
    e.preventDefault();
    view.zoom = Math.min(Math.max(view.zoom - e.deltaY * 0.0012, 0), 1);
}, { passive: false });

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

// slow motion: 1x -> 1/2x -> 1/4x; the soundtrack pauses while slowed (the player would keep re-syncing it)
const SPEEDS = [1, 0.5, 0.25];
let sound = null, soundOn = false;
function applySound() {
    if (sound) sound[0].volume = soundOn && clock.rate === 1 ? soundVolume : 0;
    soundButton.textContent = soundOn ? "Mute" : "Enable sound";
}
function setSpeed(rate) {
    clock.setRate(rate);
    speedButton.textContent = rate === 1 ? "1×" : rate === 0.5 ? "½×" : "¼×";
    speedButton.title = rate === 1 ? "Slow motion" : `Slow motion: ${rate}× speed`;
    applySound();
}
speedButton.addEventListener("click", () => setSpeed(SPEEDS[(SPEEDS.indexOf(clock.rate) + 1) % SPEEDS.length]));
setSpeed(SPEEDS.find((x) => x === Number(params.get("speed"))) ?? 1);

const viewer = { A, B, clock, view, players: [], setSplit };
window.asogCompare = viewer;
B.app.on("update", () => {
    const p = viewer.players[1] ?? viewer.players[0];
    if (!p) return;
    if (!scrubbing) scrub.value = String(p.time);
    clockLabel.textContent = `${fmt(p.time)} / ${fmt(p.duration)} s`;
});

function setLabel(id, title, player) {
    const el = $(id);
    el.querySelector("b").textContent = title;
    el.querySelector("span").textContent = `${(player.stats.fileBytes / 1e6).toFixed(0)} MB · ${(player.numSplats / 1e6).toFixed(2)}M splats`;
}

// ---------------------------------------------------------------- load both
$("loader-name").textContent = `${fileName(srcA)} | ${fileName(srcB)} · ${B.device}`;
loader.hidden = false;
const progress = [0, 0];
const showProgress = () => {
    const p = (progress[0] + progress[1]) / 2;
    loaderPct.textContent = `${Math.round(p * 100)} %`;
    loaderFill.style.width = `${p * 100}%`;
};
try {
    const load = (app, src, i) => AsogPlayer.load(app, src, app.root, {
        getTime: () => clock.now(),
        volume: 0, // muted until "Enable sound"
        onProgress: (p) => {
            progress[i] = p;
            showProgress();
        },
    });
    const [pa, pb] = await Promise.all([load(A.app, srcA, 0), load(B.app, srcB, 1)]);
    viewer.players = [pa, pb];
    scrub.max = String(Math.max(pa.duration, pb.duration));
    setLabel("label-a", titleA, pa);
    setLabel("label-b", titleB, pb);

    // where the performer is: B's robust bounds in world space (the player turns the clip y up)
    const { min, max } = pb.bounds;
    const m = pb.entity.getWorldTransform();
    const c0 = m.transformPoint(new pc.Vec3(min[0], min[1], min[2]));
    const c1 = m.transformPoint(new pc.Vec3(max[0], max[1], max[2]));
    const lo = new pc.Vec3(Math.min(c0.x, c1.x), Math.min(c0.y, c1.y), Math.min(c0.z, c1.z));
    const hi = new pc.Vec3(Math.max(c0.x, c1.x), Math.max(c0.y, c1.y), Math.max(c0.z, c1.z));
    const height = hi.y - lo.y, size = Math.max(hi.x - lo.x, height, hi.z - lo.z);
    view.body.set((lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2);
    view.head.set(view.body.x, hi.y - 0.08 * height, view.body.z);
    view.far = size / (2 * Math.tan(25 * Math.PI / 180)) * 1.15;
    view.near = 0.22 * height;

    sound = pb.hasAudio ? [pb, B.app] : pa.hasAudio ? [pa, A.app] : null;
    if (sound) {
        soundButton.hidden = false;
        soundButton.addEventListener("click", async () => {
            soundOn = !soundOn;
            if (soundOn) await sound[1].systems.sound.context.resume();
            applySound();
        });
    }
    loader.hidden = true;
} catch (e) {
    loader.hidden = true;
    const err = $("error");
    err.hidden = false;
    err.textContent = `Could not load: ${e.message}`;
    throw e;
}
