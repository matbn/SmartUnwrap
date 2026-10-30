/* ============================================================
   SmartUnwrapper — frontend logic
   Three.js (MIT) for the 3D viewport, 2D canvas for UV layout.
   ============================================================ */

/* --- three.js (dynamically imported so the rest of the UI
       still works if the CDN is unreachable) --- */
let THREE = null;
let OrbitControls = null;
try {
    THREE = await import('three');
    ({ OrbitControls } = await import('three/addons/controls/OrbitControls.js'));
} catch (err) {
    console.warn('three.js unavailable:', err);
}

const $ = (id) => document.getElementById(id);

const ui = {
    healthPill: $('health-pill'),
    healthText: $('health-text'),

    modelDrop: $('model-drop'),
    modelInput: $('model-upload'),
    modelEmpty: $('model-empty'),
    modelFile: $('model-file'),
    modelName: $('model-name'),
    modelSize: $('model-size'),
    modelClear: $('model-clear'),

    methodSelect: $('method-select'),
    sam2Options: $('sam2-options'),
    partDetection: $('part-detection-thresh'),
    partDetectionValue: $('part-detection-value'),
    sam2Iou: $('sam2-iou'),
    sam2IouValue: $('sam2-iou-value'),
    sam2Stability: $('sam2-stability'),
    sam2StabilityValue: $('sam2-stability-value'),
    btnUnwrap: $('btn-unwrap'),

    textureDrop: $('texture-drop'),
    textureInput: $('texture-upload'),
    textureEmpty: $('texture-empty'),
    textureFile: $('texture-file'),
    textureThumb: $('texture-thumb'),
    textureName: $('texture-name'),
    textureSize: $('texture-size'),
    textureClear: $('texture-clear'),
    projectionOptions: $('projection-options'),
    projectionAxis: $('projection-axis'),
    btnProject: $('btn-project'),
    btnRestoreUv: $('btn-restore-uv'),

    btnApply: $('btn-apply'),
    btnDownloadTex: $('btn-download-tex'),

    stV: $('st-v'),
    stF: $('st-f'),
    stMethod: $('st-method'),
    stTime: $('st-time'),
    btnExport: $('btn-export'),

    viewport: $('viewport'),
    viewportEmpty: $('viewport-empty'),
    viewportLoading: $('viewport-loading'),
    loadingText: $('loading-text'),
    viewModes: $('view-modes'),
    semanticLegend: $('semantic-legend'),

    uvCanvas: $('uv-canvas'),
    uvMeta: $('uv-meta'),
    uvTools: $('uv-tools'),
    uvStatus: $('uv-status'),
    uvFit: $('uv-fit'),
    uvReset: $('uv-reset'),

    toasts: $('toasts'),
};

const state = {
    file: null,            // uploaded model File
    vertices: null,        // [[x,y,z], ...]
    faces: null,           // [[a,b,c], ...]
    uv: null,              // Float64Array, flat [u0,v0,u1,v1,...]
    uvInitial: null,       // snapshot of the automatic layout (for Reset)
    islands: null,         // Int32Array, island index per vertex
    semanticIds: null,     // Int32Array, semantic class index per vertex (separate from UV charts)
    method: 'SAM2',
    textureDataUrl: null,  // user texture (data URL)
    textureImageEl: null,  // HTMLImageElement of user texture
    bakedUrl: null,        // server-baked texture (data URL)
    bakedImageEl: null,
    projection: null,      // active image projection metadata
    viewMode: 'shaded',
    threeFailed: false,
    busy: false,
};

const MODEL_EXTS = ['obj', 'stl', 'ply', 'glb', 'gltf', 'fbx'];
const IMAGE_EXTS = ['png', 'jpg', 'jpeg', 'webp'];

/* ============================================================
   Toasts
   ============================================================ */
function toast(message, type = 'info', ms = 4500) {
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.textContent = message;
    ui.toasts.appendChild(el);
    while (ui.toasts.children.length > 4) ui.toasts.firstChild.remove();
    setTimeout(() => {
        el.classList.add('leaving');
        setTimeout(() => el.remove(), 300);
    }, ms);
}

/* ============================================================
   Three.js viewport
   ============================================================ */
let renderer = null, scene = null, camera = null, controls = null;
let meshGroup = null, meshObj = null, wireMesh = null;
const glCanvas = $('gl');

function makeCheckerTexture(cells = 8) {
    const c = document.createElement('canvas');
    c.width = c.height = 512;
    const g = c.getContext('2d');
    const s = c.width / cells;
    for (let y = 0; y < cells; y++) {
        for (let x = 0; x < cells; x++) {
            g.fillStyle = (x + y) % 2 === 0 ? '#c8cfdd' : '#39415a';
            g.fillRect(x * s, y * s, s, s);
        }
    }
    g.strokeStyle = 'rgba(0,0,0,0.3)';
    g.lineWidth = 1;
    for (let i = 0; i <= cells; i++) {
        g.beginPath(); g.moveTo(i * s, 0); g.lineTo(i * s, c.height); g.stroke();
        g.beginPath(); g.moveTo(0, i * s); g.lineTo(c.width, i * s); g.stroke();
    }
    const t = new THREE.CanvasTexture(c);
    t.colorSpace = THREE.SRGBColorSpace;
    return t;
}

let checkerTex = null;
let matShaded, matChecker, matTextured, matWire, matIslands;

function initThree() {
    if (!THREE) { state.threeFailed = true; return; }

    renderer = new THREE.WebGLRenderer({ canvas: glCanvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0e15);

    camera = new THREE.PerspectiveCamera(50, 1, 0.01, 500);
    camera.position.set(2.8, 1.9, 3.4);

    controls = new OrbitControls(camera, glCanvas);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;

    scene.add(new THREE.HemisphereLight(0xdfe7ff, 0x141821, 1.0));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(4, 6, 5);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x8fb0ff, 0.5);
    rim.position.set(-5, -2, -4);
    scene.add(rim);

    const grid = new THREE.GridHelper(10, 20, 0x27314a, 0x1a2130);
    grid.position.y = -1.35;
    scene.add(grid);

    checkerTex = makeCheckerTexture();
    matShaded = new THREE.MeshStandardMaterial({
        color: 0xaab4c4, roughness: 0.85, metalness: 0.08, side: THREE.DoubleSide,
    });
    matChecker = new THREE.MeshStandardMaterial({
        map: checkerTex, roughness: 0.9, metalness: 0, side: THREE.DoubleSide,
    });
    matTextured = new THREE.MeshStandardMaterial({
        color: 0xffffff, roughness: 0.85, metalness: 0, side: THREE.DoubleSide,
    });
    matIslands = new THREE.MeshStandardMaterial({
        vertexColors: true, roughness: 0.78, metalness: 0.02, side: THREE.DoubleSide,
    });
    matWire = new THREE.MeshBasicMaterial({
        color: 0x63b3ff, wireframe: true, transparent: true, opacity: 0.4,
    });

    const resize = () => {
        const w = ui.viewport.clientWidth, h = ui.viewport.clientHeight;
        if (!w || !h) return;
        renderer.setSize(w, h, false);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
    };
    resize();
    new ResizeObserver(resize).observe(ui.viewport);

    renderer.setAnimationLoop(() => {
        controls.update();
        renderer.render(scene, camera);
    });
}

function clearMesh() {
    if (!meshGroup) return;
    scene.remove(meshGroup);
    meshGroup.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
    });
    meshGroup = null;
    meshObj = null;
    wireMesh = null;
}

function islandHue(id) {
    // Golden-angle hue spacing gives neighboring island ids very different hues.
    return ((Math.max(0, id) * 0.618033988749895) + 0.035) % 1;
}

function buildIslandColors() {
    if (!THREE || !state.vertices || !state.islands) return null;
    const count = state.vertices.length;
    if (state.islands.length !== count) return null;
    const colorIds = state.semanticIds?.length === count ? state.semanticIds : state.islands;

    let maxId = -1;
    for (let i = 0; i < count; i++) maxId = Math.max(maxId, colorIds[i]);
    const islandCount = maxId + 1;
    const lowY = new Float32Array(islandCount); lowY.fill(Infinity);
    const highY = new Float32Array(islandCount); highY.fill(-Infinity);
    for (let i = 0; i < count; i++) {
        const id = colorIds[i];
        if (id < 0) continue;
        const y = state.vertices[i][1];
        lowY[id] = Math.min(lowY[id], y);
        highY[id] = Math.max(highY[id], y);
    }

    const colors = new Float32Array(count * 3);
    const color = new THREE.Color();
    for (let i = 0; i < count; i++) {
        const id = colorIds[i];
        if (id < 0) {
            color.setRGB(0.42, 0.44, 0.49);
        } else {
            const range = highY[id] - lowY[id];
            const gradient = range > 1e-8
                ? (state.vertices[i][1] - lowY[id]) / range
                : 0.5;
            color.setHSL(islandHue(id), 0.76, 0.34 + 0.30 * gradient);
        }
        color.toArray(colors, i * 3);
    }
    return colors;
}

function updateSemanticLegend(status) {
    const legend = ui.semanticLegend;
    legend.replaceChildren();
    const names = status?.semantic_region_names;
    if (!status?.semantic_model || !names) {
        legend.classList.add('hidden');
        return;
    }
    const seenNames = new Set();
    for (const [idText, name] of Object.entries(names).sort((a, b) => Number(a[0]) - Number(b[0]))) {
        if (seenNames.has(name)) continue;
        seenNames.add(name);
        const id = Number(idText);
        const hue = islandHue(id) * 360;
        const item = document.createElement('span');
        item.className = 'semantic-legend-item';
        const swatch = document.createElement('i');
        swatch.className = 'semantic-legend-swatch';
        swatch.style.backgroundColor = `hsl(${hue} 76% 49%)`;
        const label = document.createElement('span');
        label.textContent = name === 'unclassified'
            ? 'Unclassified'
            : name.charAt(0).toUpperCase() + name.slice(1);
        item.append(swatch, label);
        legend.appendChild(item);
    }
    legend.classList.remove('hidden');
}

function buildMeshObject() {
    if (!THREE) return;
    clearMesh();
    if (!state.vertices || !state.faces) return;

    const g = new THREE.BufferGeometry();
    const pos = new Float32Array(state.vertices.length * 3);
    for (let i = 0; i < state.vertices.length; i++) {
        const v = state.vertices[i];
        pos[i * 3] = v[0]; pos[i * 3 + 1] = v[1]; pos[i * 3 + 2] = v[2];
    }
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(state.faces.flat());
    g.computeBoundingBox();

    // Center & normalize scale so any model fits nicely
    const bb = g.boundingBox;
    const size = new THREE.Vector3();
    bb.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    const center = bb.getCenter(new THREE.Vector3());
    g.translate(-center.x, -center.y, -center.z);
    const s = 2.2 / maxDim;
    g.scale(s, s, s);

    if (state.uv && state.uv.length === state.vertices.length * 2) {
        const uvs = new Float32Array(state.uv.length);
        uvs.set(state.uv);
        g.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));
    }
    const islandColors = buildIslandColors();
    if (islandColors) g.setAttribute('color', new THREE.BufferAttribute(islandColors, 3));
    g.computeVertexNormals();

    meshObj = new THREE.Mesh(g, materialFor(state.viewMode));
    wireMesh = new THREE.Mesh(g, matWire);
    wireMesh.visible = state.viewMode === 'wireframe';

    meshGroup = new THREE.Group();
    meshGroup.add(meshObj, wireMesh);
    scene.add(meshGroup);

    camera.position.set(2.8, 1.9, 3.4);
    controls.target.set(0, 0, 0);
    controls.update();
}

/* Push edited UVs into the live three.js geometry (uv is per-vertex, so a
   single buffer update is enough — meshObj and wireMesh share the geometry). */
function syncUVToMesh() {
    if (!meshObj || !state.uv) return;
    const attr = meshObj.geometry.getAttribute('uv');
    if (!attr || attr.count * 2 !== state.uv.length) return;
    attr.array.set(state.uv);
    attr.needsUpdate = true;
}

function materialFor(mode) {
    if (mode === 'islands') return matIslands;
    if (mode === 'checkered') return matChecker;
    if (mode === 'textured') {
        if (state.textureImageEl && THREE) {
            const tex = new THREE.TextureLoader().load(state.textureDataUrl);
            tex.colorSpace = THREE.SRGBColorSpace;
            matTextured.map = tex;
            matTextured.needsUpdate = true;
            return matTextured;
        }
        return matChecker;
    }
    return matShaded; // shaded & wireframe base
}

function setViewMode(mode) {
    state.viewMode = mode;
    ui.viewModes.querySelectorAll('button').forEach((b) =>
        b.classList.toggle('active', b.dataset.mode === mode));
    if (!meshObj) return;
    wireMesh.visible = mode === 'wireframe';
    meshObj.material = materialFor(mode);
}

/* ============================================================
   UV editor — island selection + Blender-style transforms
   ------------------------------------------------------------
   state.uv      : Float64Array of 2*N (flat u,v)
   state.islands : Int32Array of N (island index per vertex)
   ============================================================ */
const uvCtx = ui.uvCanvas.getContext('2d');

const uvEd = {
    view: { zoom: 1, panX: 0, panY: 0 },
    tool: 'select',              // select | rotate | scale
    selection: new Set(),
    hover: -1,
    transform: null,
    version: 0,
    undo: [],
    islandCount: 0,
    islandFaces: null,           // { offsets: Int32Array, order: Int32Array }
    islandBoxes: null,           // Float32Array(4 * islandCount)
    panning: null,
    boxSelectArmed: false,       // B: next primary drag selects islands in a rectangle
    boxSelect: null,             // { start, current, additive }
    checker: null,
};

function makeUvChecker() {
    const c = document.createElement('canvas');
    c.width = c.height = 256;
    const g = c.getContext('2d');
    const cells = 16, s = c.width / cells;
    g.fillStyle = '#0f1420';
    g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#161d2d';
    for (let y = 0; y < cells; y++)
        for (let x = 0; x < cells; x++)
            if ((x + y) % 2 === 0) g.fillRect(x * s, y * s, s, s);
    return c;
}

/* ---------- island bookkeeping ---------- */

function rebuildIslandData() {
    const nv = state.vertices ? state.vertices.length : 0;
    const nf = state.faces ? state.faces.length : 0;

    if (!state.islands || state.islands.length !== nv) {
        // Fallback: treat the whole layout as one island so editing still works
        state.islands = new Int32Array(nv);
    }
    const isl = state.islands;

    let maxId = -1;
    for (let i = 0; i < nv; i++) if (isl[i] > maxId) maxId = isl[i];
    const nIsl = maxId + 1;
    uvEd.islandCount = nIsl;

    // CSR: faces grouped per island
    const offsets = new Int32Array(nIsl + 1);
    for (let fi = 0; fi < nf; fi++) offsets[isl[state.faces[fi][0]] + 1]++;
    for (let i = 1; i <= nIsl; i++) offsets[i] += offsets[i - 1];
    const order = new Int32Array(nf);
    const cursor = offsets.slice(0, nIsl);
    for (let fi = 0; fi < nf; fi++) {
        const id = isl[state.faces[fi][0]];
        order[cursor[id]++] = fi;
    }
    uvEd.islandFaces = { offsets, order };

    // per-island UV bounding boxes (for fast picking)
    const boxes = new Float32Array(nIsl * 4);
    for (let i = 0; i < nIsl; i++) {
        boxes[i * 4] = Infinity; boxes[i * 4 + 1] = Infinity;
        boxes[i * 4 + 2] = -Infinity; boxes[i * 4 + 3] = -Infinity;
    }
    const uv = state.uv;
    if (uv && uv.length === nv * 2) {
        for (let v = 0; v < nv; v++) {
            const id = isl[v];
            if (id < 0) continue;
            const u = uv[v * 2], w = uv[v * 2 + 1], b = id * 4;
            if (u < boxes[b]) boxes[b] = u;
            if (w < boxes[b + 1]) boxes[b + 1] = w;
            if (u > boxes[b + 2]) boxes[b + 2] = u;
            if (w > boxes[b + 3]) boxes[b + 3] = w;
        }
    }
    uvEd.islandBoxes = boxes;
    uvEd.selection.clear();
    uvEd.hover = -1;
    clearIslandCache();
}

function refreshIslandBoxes() {
    const isl = state.islands, uv = state.uv;
    if (!isl || !uv) return;
    const nIsl = uvEd.islandCount;
    const boxes = uvEd.islandBoxes;
    for (let i = 0; i < nIsl; i++) {
        boxes[i * 4] = Infinity; boxes[i * 4 + 1] = Infinity;
        boxes[i * 4 + 2] = -Infinity; boxes[i * 4 + 3] = -Infinity;
    }
    for (let v = 0; v < isl.length; v++) {
        const id = isl[v];
        if (id < 0) continue;
        const u = uv[v * 2], w = uv[v * 2 + 1], b = id * 4;
        if (u < boxes[b]) boxes[b] = u;
        if (w < boxes[b + 1]) boxes[b + 1] = w;
        if (u > boxes[b + 2]) boxes[b + 2] = u;
        if (w > boxes[b + 3]) boxes[b + 3] = w;
    }
}

/* ---------- paths & drawing ---------- */

let drawScheduled = false;

function requestDraw() {
    if (!drawScheduled) {
        drawScheduled = true;
        requestAnimationFrame(() => {
            drawScheduled = false;
            drawUVLayoutNow();
        });
    }
}

// Per-island cached Path2D. Only the island(s) being transformed are rebuilt
// each frame; everything else replays from cache, so dragging a single island
// over a 900+ island atlas no longer reparses 30k segments per pointermove.
const islandPathCache = new Map();
const islandDirty = new Set();

function islandPath(id) {
    const cached = islandPathCache.get(id);
    if (cached && !islandDirty.has(id)) return cached;
    const path = new Path2D();
    const { offsets, order } = uvEd.islandFaces;
    const uv = state.uv, faces = state.faces;
    for (let k = offsets[id]; k < offsets[id + 1]; k++) {
        const f = faces[order[k]];
        const a = f[0] * 2, b = f[1] * 2, c = f[2] * 2;
        path.moveTo(uv[a], 1 - uv[a + 1]);
        path.lineTo(uv[b], 1 - uv[b + 1]);
        path.lineTo(uv[c], 1 - uv[c + 1]);
        path.closePath();
    }
    islandPathCache.set(id, path);
    islandDirty.delete(id);
    return path;
}

function clearIslandCache() {
    islandPathCache.clear();
    islandDirty.clear();
}

function setUvMeta() {
    if (!state.uv) {
        ui.uvMeta.textContent = '—';
        return;
    }
    const sel = uvEd.selection.size;
    ui.uvMeta.textContent = `${uvEd.islandCount} island${uvEd.islandCount === 1 ? '' : 's'}`
        + (sel ? ` · ${sel} selected` : '');
}

function setUvStatus(text) {
    ui.uvStatus.textContent = text;
}

const UV_HINT = 'Click an island to select · B then drag for box select · arrow keys nudge · G / R / S · scroll to zoom';

function drawUVLayoutNow() {
    const S = ui.uvCanvas.width;
    const ctx = uvCtx;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, S, S);
    ctx.fillStyle = '#080b12';
    ctx.fillRect(0, 0, S, S);

    const { zoom, panX, panY } = uvEd.view;
    const k = S * zoom;
    ctx.setTransform(k, 0, 0, k, S * panX, S * panY);

    // texture / checker inside the 0..1 square
    const bg = state.bakedImageEl || state.textureImageEl;
    if (bg && bg.complete && bg.naturalWidth > 0) {
        ctx.drawImage(bg, 0, 0, 1, 1);
    } else {
        if (!uvEd.checker) uvEd.checker = makeUvChecker();
        ctx.drawImage(uvEd.checker, 0, 0, 1, 1);
    }

    if (!state.uv || !state.faces) {
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        setUvMeta();
        return;
    }

    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';

    // Each UV island uses the same hue as its 3D mesh region. Lightness varies
    // over the island to make its shape easier to read against the checker.
    ctx.lineWidth = 1 / k;
    for (let id = 0; id < uvEd.islandCount; id++) {
        const ip = islandPath(id);
        const b = id * 4;
        const hue = islandHue(id) * 360;
        const gradient = ctx.createLinearGradient(0, 1 - uvEd.islandBoxes[b + 3], 0, 1 - uvEd.islandBoxes[b + 1]);
        gradient.addColorStop(0, `hsla(${hue}, 76%, 60%, 0.34)`);
        gradient.addColorStop(1, `hsla(${hue}, 76%, 34%, 0.34)`);
        ctx.fillStyle = gradient;
        ctx.strokeStyle = `hsla(${hue}, 82%, 68%, 0.76)`;
        ctx.fill(ip);
        ctx.stroke(ip);
    }

    // Keep island identifiers readable over projected/baked textures.
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.font = `600 ${Math.max(10 / k, 0.018)}px system-ui, sans-serif`;
    for (let id = 0; id < uvEd.islandCount; id++) {
        const b = id * 4;
        const width = uvEd.islandBoxes[b + 2] - uvEd.islandBoxes[b];
        const height = uvEd.islandBoxes[b + 3] - uvEd.islandBoxes[b + 1];
        if (width * k < 34 || height * k < 18) continue;
        const x = (uvEd.islandBoxes[b] + uvEd.islandBoxes[b + 2]) / 2;
        const y = 1 - (uvEd.islandBoxes[b + 1] + uvEd.islandBoxes[b + 3]) / 2;
        ctx.lineWidth = 3 / k;
        ctx.strokeStyle = 'rgba(5, 8, 14, 0.82)';
        ctx.strokeText(`Island ${id + 1}`, x, y);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.92)';
        ctx.fillText(`Island ${id + 1}`, x, y);
    }

    // hovered island
    if (uvEd.hover >= 0 && !uvEd.selection.has(uvEd.hover)) {
        const hp = islandPath(uvEd.hover);
        ctx.fillStyle = 'rgba(96, 150, 255, 0.22)';
        ctx.fill(hp);
        ctx.lineWidth = 1.4 / k;
        ctx.strokeStyle = 'rgba(180, 230, 255, 0.95)';
        ctx.stroke(hp);
    }

    // selected islands
    for (const id of uvEd.selection) {
        const sp = islandPath(id);
        ctx.fillStyle = 'rgba(45, 212, 191, 0.20)';
        ctx.fill(sp);
        ctx.lineWidth = 1.8 / k;
        ctx.strokeStyle = 'rgba(94, 234, 212, 0.95)';
        ctx.stroke(sp);
    }

    // selection bounding box (transform pivot frame)
    if (uvEd.selection.size) {
        const bb = selectionBox();
        if (bb) {
            ctx.setLineDash([5 / k, 4 / k]);
            ctx.lineWidth = 1 / k;
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
            ctx.strokeRect(bb[0], 1 - bb[3], bb[2] - bb[0], bb[3] - bb[1]);
            ctx.setLineDash([]);
        }
    }

    // Rectangle drawn while B box-select is active. Coordinates remain in UV
    // space, so it stays aligned when the view is zoomed or panned.
    if (uvEd.boxSelect) {
        const { start, current } = uvEd.boxSelect;
        const u0 = Math.min(start.u, current.u), u1 = Math.max(start.u, current.u);
        const v0 = Math.min(start.v, current.v), v1 = Math.max(start.v, current.v);
        ctx.setLineDash([5 / k, 3 / k]);
        ctx.lineWidth = 1.2 / k;
        ctx.fillStyle = 'rgba(79, 140, 255, 0.13)';
        ctx.strokeStyle = 'rgba(132, 181, 255, 0.98)';
        ctx.fillRect(u0, 1 - v1, u1 - u0, v1 - v0);
        ctx.strokeRect(u0, 1 - v1, u1 - u0, v1 - v0);
        ctx.setLineDash([]);
    }

    // UV square
    ctx.lineWidth = 1.6 / k;
    ctx.strokeStyle = 'rgba(126, 146, 186, 0.6)';
    ctx.strokeRect(0, 0, 1, 1);

    ctx.setTransform(1, 0, 0, 1, 0, 0);
    setUvMeta();
}

// Pointer-move / transform redraws are coalesced to one frame. Initial draw
// and explicit resets still paint immediately.
function drawUVLayout() { requestDraw(); }
drawUVLayout.now = drawUVLayoutNow;

/* ---------- selection helpers ---------- */

function selectedVerts() {
    const isl = state.islands, out = [];
    if (!isl) return out;
    for (let v = 0; v < isl.length; v++) {
        if (uvEd.selection.has(isl[v])) out.push(v);
    }
    return out;
}

function selectionBox() {
    const verts = selectedVerts();
    if (!verts.length) return null;
    const uv = state.uv;
    let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
    for (const v of verts) {
        const u = uv[v * 2], w = uv[v * 2 + 1];
        if (u < u0) u0 = u; if (u > u1) u1 = u;
        if (w < v0) v0 = w; if (w > v1) v1 = w;
    }
    return [u0, v0, u1, v1];
}

function pointInTri(uv, f, u, v) {
    const ax = uv[f[0] * 2], ay = uv[f[0] * 2 + 1];
    const bx = uv[f[1] * 2], by = uv[f[1] * 2 + 1];
    const cx = uv[f[2] * 2], cy = uv[f[2] * 2 + 1];
    const d1 = (u - bx) * (ay - by) - (ax - bx) * (v - by);
    const d2 = (u - cx) * (by - cy) - (bx - cx) * (v - cy);
    const d3 = (u - ax) * (cy - ay) - (cx - ax) * (v - ay);
    const hasNeg = d1 < 0 || d2 < 0 || d3 < 0;
    const hasPos = d1 > 0 || d2 > 0 || d3 > 0;
    return !(hasNeg && hasPos);
}

function islandAt(p) {
    if (!state.uv || !uvEd.islandBoxes) return -1;
    const boxes = uvEd.islandBoxes;
    const nIsl = uvEd.islandCount;
    const { offsets, order } = uvEd.islandFaces;
    const faces = state.faces, uv = state.uv;
    for (let id = nIsl - 1; id >= 0; id--) {
        const b = id * 4;
        if (p.u < boxes[b] || p.u > boxes[b + 2]) continue;
        if (p.v < boxes[b + 1] || p.v > boxes[b + 3]) continue;
        for (let k = offsets[id]; k < offsets[id + 1]; k++) {
            if (pointInTri(uv, faces[order[k]], p.u, p.v)) return id;
        }
    }
    return -1;
}

function selectIslandsInBox(start, current, additive) {
    const u0 = Math.min(start.u, current.u), u1 = Math.max(start.u, current.u);
    const v0 = Math.min(start.v, current.v), v1 = Math.max(start.v, current.v);
    if (!additive) uvEd.selection.clear();
    for (let id = 0; id < uvEd.islandCount; id++) {
        const b = id * 4;
        // An island is selected when any portion of its bounds intersects the
        // rectangle. This is the expected behaviour for tightly packed charts.
        if (uvEd.islandBoxes[b] <= u1 && uvEd.islandBoxes[b + 2] >= u0
            && uvEd.islandBoxes[b + 1] <= v1 && uvEd.islandBoxes[b + 3] >= v0) {
            uvEd.selection.add(id);
        }
    }
}

function nudgeSelection(du, dv) {
    if (!uvEd.selection.size) {
        toast('Select one or more islands first', 'info', 1800);
        return;
    }
    const verts = selectedVerts();
    if (!verts.length) return;
    pushUndo();
    for (const vertex of verts) {
        state.uv[vertex * 2] += du;
        state.uv[vertex * 2 + 1] += dv;
        islandDirty.add(state.islands[vertex]);
    }
    uvEd.version++;
    refreshIslandBoxes();
    syncUVToMesh();
    drawUVLayout.now();
    setUvStatus(`nudge  Δu ${du.toFixed(3)}  Δv ${dv.toFixed(3)} · ${uvEd.selection.size} island${uvEd.selection.size === 1 ? '' : 's'}`);
}

/* ---------- transforms ---------- */

function pushUndo() {
    if (!state.uv) return;
    uvEd.undo.push(state.uv.slice());
    if (uvEd.undo.length > 24) uvEd.undo.shift();
}

function undoUv() {
    if (!uvEd.undo.length) { toast('Nothing to undo', 'info', 2000); return; }
    state.uv = uvEd.undo.pop();
    syncProjectionState();
    uvEd.version++;
    refreshIslandBoxes();
    syncUVToMesh();
    drawUVLayout.now();
    updateStats();
    refreshButtons();
    setUvStatus('Undo');
}

function beginTransform(mode, anchor, autoCommit) {
    if (!state.uv || !uvEd.selection.size) {
        toast('Select an island first', 'info', 2200);
        return false;
    }
    const verts = selectedVerts();
    const orig = new Float64Array(verts.length * 2);
    for (let i = 0; i < verts.length; i++) {
        orig[i * 2] = state.uv[verts[i] * 2];
        orig[i * 2 + 1] = state.uv[verts[i] * 2 + 1];
    }
    let u0 = Infinity, v0 = Infinity, u1 = -Infinity, v1 = -Infinity;
    for (let i = 0; i < verts.length; i++) {
        u0 = Math.min(u0, orig[i * 2]); u1 = Math.max(u1, orig[i * 2]);
        v0 = Math.min(v0, orig[i * 2 + 1]); v1 = Math.max(v1, orig[i * 2 + 1]);
    }
    uvEd.transform = {
        mode, verts, orig,
        pivot: { u: (u0 + u1) / 2, v: (v0 + v1) / 2 },
        anchor,
        autoCommit,
        moved: false,
        snap: false,
        label: '',
    };
    return true;
}

function applyTransform(cur, snap) {
    const t = uvEd.transform;
    if (!t) return;
    if (snap !== undefined) t.snap = snap;
    if (!t.moved) { pushUndo(); t.moved = true; }

    const uv = state.uv;
    const { verts, orig, pivot } = t;
    let label = '';

    if (t.mode === 'move') {
        let du = cur.u - t.anchor.u, dv = cur.v - t.anchor.v;
        if (t.snap) { du = Math.round(du * 20) / 20; dv = Math.round(dv * 20) / 20; }
        for (let i = 0; i < verts.length; i++) {
            uv[verts[i] * 2] = orig[i * 2] + du;
            uv[verts[i] * 2 + 1] = orig[i * 2 + 1] + dv;
        }
        label = `move  Δu ${du.toFixed(3)}  Δv ${dv.toFixed(3)}`;
    } else if (t.mode === 'scale') {
        const d0 = Math.hypot(t.anchor.u - pivot.u, t.anchor.v - pivot.v);
        const d1 = Math.hypot(cur.u - pivot.u, cur.v - pivot.v);
        let f = d0 > 1e-5 ? d1 / d0 : 1;
        if (t.snap) f = Math.max(0.05, Math.round(f * 10) / 10);
        for (let i = 0; i < verts.length; i++) {
            uv[verts[i] * 2] = pivot.u + (orig[i * 2] - pivot.u) * f;
            uv[verts[i] * 2 + 1] = pivot.v + (orig[i * 2 + 1] - pivot.v) * f;
        }
        label = `scale  ×${f.toFixed(3)}`;
    } else {
        let a = Math.atan2(cur.v - pivot.v, cur.u - pivot.u)
              - Math.atan2(t.anchor.v - pivot.v, t.anchor.u - pivot.u);
        if (t.snap) a = Math.round(a / (Math.PI / 12)) * (Math.PI / 12);
        const c = Math.cos(a), s = Math.sin(a);
        for (let i = 0; i < verts.length; i++) {
            const du = orig[i * 2] - pivot.u, dv = orig[i * 2 + 1] - pivot.v;
            uv[verts[i] * 2] = pivot.u + du * c - dv * s;
            uv[verts[i] * 2 + 1] = pivot.v + du * s + dv * c;
        }
        label = `rotate  ${(a * 180 / Math.PI).toFixed(1)}°`;
    }

    t.label = label;
    uvEd.version++;
    for (const v of verts) islandDirty.add(state.islands[v]);
    syncUVToMesh();
    drawUVLayout.now();   // synchronous during active transforms — no rAF frame latency
    setUvStatus(`${label} · ${verts.length} verts · Ctrl to snap · Esc to cancel`);
}

function endTransform(restore) {
    const t = uvEd.transform;
    if (!t) return;
    if (restore && t.moved) {
        for (let i = 0; i < t.verts.length; i++) {
            state.uv[t.verts[i] * 2] = t.orig[i * 2];
            state.uv[t.verts[i] * 2 + 1] = t.orig[i * 2 + 1];
        }
        uvEd.undo.pop();
        uvEd.version++;
        syncUVToMesh();
    }
    // UV hit-testing uses these bounds as a broad phase. Refresh them after
    // every completed transform, or moved islands become impossible to pick.
    refreshIslandBoxes();
    uvEd.hover = -1;
    uvEd.transform = null;
    drawUVLayout.now();
    setUvStatus(UV_HINT);
}

/* ---------- canvas interaction ---------- */

function eventUV(e) {
    const r = ui.uvCanvas.getBoundingClientRect();
    const x = (e.clientX - r.left) / Math.max(r.width, 1);
    const y = (e.clientY - r.top) / Math.max(r.height, 1);
    const { zoom, panX, panY } = uvEd.view;
    return { u: (x - panX) / zoom, v: 1 - (y - panY) / zoom };
}

function zoomAt(p, factor) {
    const v = uvEd.view;
    const z = Math.min(60, Math.max(0.3, v.zoom * factor));
    const d = v.zoom - z;
    v.panX += p.u * d;
    v.panY += (1 - p.v) * d;
    v.zoom = z;
    drawUVLayout.now();
}

function resetView() {
    uvEd.view.zoom = 1;
    uvEd.view.panX = 0;
    uvEd.view.panY = 0;
    drawUVLayout.now();
}

function setUvTool(tool) {
    uvEd.tool = tool;
    ui.uvTools.querySelectorAll('button[data-tool]').forEach((b) =>
        b.classList.toggle('active', b.dataset.tool === tool));
    ui.uvCanvas.style.cursor = tool === 'select' ? 'crosshair' : 'move';
    const map = {
        select: 'Drag islands to move them · Shift+click to add · A select all',
        rotate: 'Drag to rotate the selected islands around their center · Ctrl to snap 15°',
        scale: 'Drag to scale the selected islands around their center · Ctrl to snap',
    };
    setUvStatus(map[tool] || UV_HINT);
}

function uvPointerDown(e) {
    if (!state.uv) return;
    ui.uvCanvas.focus();
    ui.uvCanvas.setPointerCapture(e.pointerId);

    if (e.button === 1 || (e.button === 0 && e.altKey)) {
        uvEd.panning = { x: e.clientX, y: e.clientY, panX: uvEd.view.panX, panY: uvEd.view.panY };
        return;
    }
    if (e.button !== 0) return;

    // a click confirms a modal transform started with G / R / S
    if (uvEd.transform && !uvEd.transform.autoCommit) { endTransform(false); return; }

    const p = eventUV(e);
    if (uvEd.boxSelectArmed) {
        uvEd.boxSelect = { start: p, current: p, additive: e.shiftKey };
        ui.uvCanvas.style.cursor = 'crosshair';
        drawUVLayout.now();
        return;
    }
    const id = islandAt(p);

    if (id < 0) {
        if (!e.shiftKey && uvEd.selection.size) {
            uvEd.selection.clear();
            uvEd.hover = -1;
            drawUVLayout.now();
            setUvStatus(UV_HINT);
        }
        return;
    }

    if (e.shiftKey) {
        if (uvEd.selection.has(id)) uvEd.selection.delete(id);
        else uvEd.selection.add(id);
    } else if (!uvEd.selection.has(id)) {
        uvEd.selection.clear();
        uvEd.selection.add(id);
    }

    drawUVLayout.now();
    if (!uvEd.selection.size) { setUvStatus(UV_HINT); return; }
    const mode = uvEd.tool === 'select' ? 'move' : uvEd.tool;
    beginTransform(mode, p, true);
}

function uvPointerMove(e) {
    if (!state.uv) return;

    if (uvEd.panning) {
        const r = ui.uvCanvas.getBoundingClientRect();
        uvEd.view.panX = uvEd.panning.panX + (e.clientX - uvEd.panning.x) / Math.max(r.width, 1);
        uvEd.view.panY = uvEd.panning.panY + (e.clientY - uvEd.panning.y) / Math.max(r.height, 1);
        drawUVLayout.now();
        return;
    }

    if (uvEd.boxSelect) {
        uvEd.boxSelect.current = eventUV(e);
        drawUVLayout.now();
        return;
    }

    const p = eventUV(e);
    if (uvEd.transform) { applyTransform(p, e.ctrlKey || e.metaKey); return; }

    const id = islandAt(p);
    if (id !== uvEd.hover) {
        uvEd.hover = id;
        drawUVLayout.now();
        if (id >= 0) {
            const cnt = uvEd.islandFaces.offsets[id + 1] - uvEd.islandFaces.offsets[id];
            setUvStatus(`Island ${id + 1}/${uvEd.islandCount} · ${cnt} tris · drag to move`);
        }
    }
}

function uvPointerUp(e) {
    if (uvEd.panning) { uvEd.panning = null; return; }
    if (uvEd.boxSelect) {
        const box = uvEd.boxSelect;
        box.current = eventUV(e);
        uvEd.boxSelect = null;
        uvEd.boxSelectArmed = false;
        selectIslandsInBox(box.start, box.current, box.additive);
        ui.uvCanvas.style.cursor = uvEd.tool === 'select' ? 'crosshair' : 'move';
        drawUVLayout.now();
        setUvStatus(`${uvEd.selection.size} island${uvEd.selection.size === 1 ? '' : 's'} selected · arrow keys nudge`);
        return;
    }
    if (uvEd.transform && uvEd.transform.autoCommit) endTransform(false);
}

function selectAllIslands() {
    if (!state.uv) return;
    uvEd.selection.clear();
    for (let i = 0; i < uvEd.islandCount; i++) uvEd.selection.add(i);
    drawUVLayout.now();
    setUvStatus(`All ${uvEd.islandCount} islands selected`);
}

function clearSelection() {
    uvEd.selection.clear();
    uvEd.hover = -1;
    drawUVLayout.now();
    setUvStatus(UV_HINT);
}

function resetUvLayout() {
    if (!state.uvInitial) return;
    pushUndo();
    state.uv = state.uvInitial.slice();
    state.projection = null;
    uvEd.version++;
    refreshIslandBoxes();
    syncUVToMesh();
    drawUVLayout.now();
    updateStats();
    refreshButtons();
    toast('Island layout restored to the automatic pack', 'ok');
}

function uvKeyDown(e) {
    if (!state.uv) return;
    const k = e.key.toLowerCase();

    if (uvEd.transform && !uvEd.transform.autoCommit) {
        if (k === 'escape') { endTransform(true); e.preventDefault(); return; }
        if (k === 'enter') { endTransform(false); e.preventDefault(); return; }
    }
    if ((e.ctrlKey || e.metaKey) && k === 'z') { undoUv(); e.preventDefault(); return; }

    if (k === 'b') {
        uvEd.boxSelectArmed = true;
        ui.uvCanvas.style.cursor = 'crosshair';
        setUvStatus('Box select armed · drag a rectangle over the islands · Shift adds to selection');
        e.preventDefault();
        return;
    }

    const nudge = e.shiftKey ? 0.01 : 0.001;
    if (e.key === 'ArrowLeft') { nudgeSelection(-nudge, 0); e.preventDefault(); return; }
    if (e.key === 'ArrowRight') { nudgeSelection(nudge, 0); e.preventDefault(); return; }
    if (e.key === 'ArrowUp') { nudgeSelection(0, nudge); e.preventDefault(); return; }
    if (e.key === 'ArrowDown') { nudgeSelection(0, -nudge); e.preventDefault(); return; }

    const startMode = (mode) => {
        if (!uvEd.selection.size) { toast('Select an island first', 'info', 2200); return; }
        const anchor = uvEd.lastPointer || { u: 0.5, v: 0.5 };
        beginTransform(mode, anchor, false);
        setUvStatus(`${mode} · move the mouse · click or Enter to apply · Esc to cancel`);
    };

    if (k === 'g') { startMode('move'); e.preventDefault(); }
    else if (k === 'r') { startMode('rotate'); e.preventDefault(); }
    else if (k === 's') { startMode('scale'); e.preventDefault(); }
    else if (k === 'a') { selectAllIslands(); e.preventDefault(); }
    else if (k === 'escape') {
        if (uvEd.boxSelectArmed || uvEd.boxSelect) {
            uvEd.boxSelectArmed = false;
            uvEd.boxSelect = null;
            ui.uvCanvas.style.cursor = uvEd.tool === 'select' ? 'crosshair' : 'move';
            drawUVLayout.now();
            setUvStatus(UV_HINT);
        } else clearSelection();
    }
    else if (k === 'home') { resetView(); e.preventDefault(); }
}


/* ============================================================
   Helpers
   ============================================================ */
const fmtSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
};

const stripDataURL = (u) => (u.startsWith('data:') ? u.split(',')[1] : u);

/* flat typed UV array -> [[u,v], ...] for JSON / OBJ export */
function uvPairs() {
    if (!state.uv) return null;
    const n = state.uv.length / 2;
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = [state.uv[i * 2], state.uv[i * 2 + 1]];
    return out;
}

const setBusy = (on, text = 'Processing…') => {
    state.busy = on;
    ui.viewportLoading.classList.toggle('hidden', !on);
    ui.loadingText.textContent = text;
};

function updateStats() {
    ui.stV.textContent = state.vertices ? state.vertices.length.toLocaleString() : '—';
    ui.stF.textContent = state.faces ? state.faces.length.toLocaleString() : '—';
    ui.stMethod.textContent = state.uv
        ? (state.projection ? `Projection ${state.projection.axis}` : state.method)
        : '—';
    ui.btnExport.disabled = !state.uv;
}

function refreshButtons() {
    ui.btnUnwrap.disabled = !state.file || state.busy;
    ui.btnApply.disabled = !state.uv || !state.textureDataUrl || state.busy;
    ui.btnProject.disabled = !state.vertices || !state.textureDataUrl || !state.uvInitial || state.busy;
    ui.btnRestoreUv.disabled = !state.projection || state.busy;
}

/* ============================================================
   Model flow
   ============================================================ */
async function unwrap() {
    if (!state.file || state.busy) return;
    state.method = ui.methodSelect.value;
    setBusy(true, 'Unwrapping UVs…');
    ui.btnUnwrap.disabled = true;
    const t0 = performance.now();

    try {
        const fd = new FormData();
        fd.append('file', state.file);
        fd.append('method', state.method);
        if (state.method === 'SAM2') {
            fd.append('part_detection_thresh', ui.partDetection.value);
            fd.append('sam2_pred_iou_thresh', ui.sam2Iou.value);
            fd.append('sam2_stability_score_thresh', ui.sam2Stability.value);
        }

        const res = await fetch('/api/unwrap', { method: 'POST', body: fd });
        const data = await res.json();
        if (!res.ok || !data.success) {
            throw new Error(data.message || `Server error (${res.status})`);
        }

        state.vertices = data.mesh_vertices;
        state.faces = data.mesh_faces;

        // flat typed arrays keep the editor fast and let three.js take a
        // BufferAttribute directly
        const pairs = data.uv || [];
        const flat = new Float64Array(pairs.length * 2);
        for (let i = 0; i < pairs.length; i++) {
            flat[i * 2] = pairs[i][0];
            flat[i * 2 + 1] = pairs[i][1];
        }
        state.uv = flat;
        state.uvInitial = flat.slice();
        state.projection = null;
        state.islands = Array.isArray(data.islands)
            ? Int32Array.from(data.islands)
            : new Int32Array(state.vertices.length);
        const st = data.status || {};
        state.semanticIds = Array.isArray(st.semantic_vertex_ids)
            && st.semantic_vertex_ids.length === state.vertices.length
            ? Int32Array.from(st.semantic_vertex_ids)
            : null;
        uvEd.undo.length = 0;

        updateSemanticLegend(st);
        if (st.valid === false) {
            toast(`UV result may be inaccurate: ${st.message || 'unknown reason'}`, 'error', 6000);
        }

        ui.stTime.textContent = `${((performance.now() - t0) / 1000).toFixed(2)}s`;
        updateStats();
        buildMeshObject();
        rebuildIslandData();
        uvEd.version++;
        setUvTool('select');
        resetView();
        drawUVLayout.now();
        setViewMode(state.uv ? 'islands' : 'shaded');
        ui.viewportEmpty.classList.add('hidden');

        const nIslands = st.islands ?? uvEd.islandCount;
        const eff = st.packing_efficiency;
        if (state.method === 'SAM2' && st.semantic_model) {
            const detected = (st.semantic_parts_detected || []).join(', ') || 'none';
            const groups = st.islands ?? uvEd.islandCount;
            const charts = st.uv_charts ?? groups;
            const source = st.semantic_model_source || 'Grounding DINO';
            toast(`Semantic parts: ${detected} — ${groups} editable groups, ${charts} UV charts (${source})`,
                'ok', 9000);
        } else if (state.method === 'SAM2' && (st.semantic_detector_note || st.sam2_note || st.sam2_mode)) {
            const reason = st.semantic_detector_note || st.sam2_note || st.sam2_mode;
            toast(`Semantic detector unavailable; geometric fallback used: ${reason}`,
                'info', 9000);
        } else if (state.method === 'SAM2' && st.sam2_used === false) {
            toast(`Topology-adaptive unwrap completed — grouped ${st.input_components} disconnected pieces into ${nIslands} UV islands`,
                'ok', 7000);
        } else {
            toast(`UV unwrapping completed — ${nIslands} islands`
                + (eff ? `, ${Math.round(eff * 100)}% atlas fill` : ''), 'ok');
        }
    } catch (err) {
        toast(err.message || 'Unwrapping failed', 'error', 6000);
    } finally {
        setBusy(false);
        refreshButtons();
    }
}

function setModelFile(file) {
    state.file = file;
    ui.modelEmpty.classList.add('hidden');
    ui.modelFile.classList.remove('hidden');
    ui.modelName.textContent = file.name;
    ui.modelSize.textContent = fmtSize(file.size);
    // reset previous results
    state.vertices = null; state.faces = null; state.uv = null;
    state.uvInitial = null; state.islands = null;
    state.projection = null;
    state.semanticIds = null;
    state.bakedUrl = null; state.bakedImageEl = null;
    updateSemanticLegend(null);
    uvEd.selection.clear(); uvEd.hover = -1; uvEd.islandCount = 0;
    uvEd.undo.length = 0;
    ui.btnDownloadTex.classList.add('hidden');
    ui.stMethod.textContent = '—';
    ui.stTime.textContent = '—';
    drawUVLayout();
    updateStats();
    unwrap(); // auto-run with the currently selected method
}

function clearModelFile() {
    state.file = null;
    ui.modelInput.value = '';
    ui.modelFile.classList.add('hidden');
    ui.modelEmpty.classList.remove('hidden');
    state.vertices = null; state.faces = null; state.uv = null;
    state.uvInitial = null; state.islands = null;
    state.projection = null;
    state.semanticIds = null;
    updateSemanticLegend(null);
    uvEd.selection.clear(); uvEd.hover = -1; uvEd.islandCount = 0;
    uvEd.undo.length = 0;
    clearMesh();
    ui.viewportEmpty.classList.remove('hidden');
    ui.btnDownloadTex.classList.add('hidden');
    ui.stTime.textContent = '—';
    ui.stMethod.textContent = '—';
    drawUVLayout.now();
    updateStats();
    refreshButtons();
}

/* ============================================================
   Texture flow
   ============================================================ */
function setTextureFile(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        state.textureDataUrl = e.target.result;
        state.bakedUrl = null;
        state.bakedImageEl = null;
        ui.btnDownloadTex.classList.add('hidden');

        const img = new Image();
        img.onload = () => {
            state.textureImageEl = img;
            ui.textureThumb.src = state.textureDataUrl;
            ui.textureEmpty.classList.add('hidden');
            ui.textureFile.classList.remove('hidden');
            ui.textureName.textContent = file.name;
            ui.textureSize.textContent = fmtSize(file.size);
            ui.projectionOptions.hidden = false;
            drawUVLayout.now();
            if (state.uv) setViewMode('textured'); // instant client-side preview
            refreshButtons();
        };
        img.src = state.textureDataUrl;
    };
    reader.readAsDataURL(file);
}

function clearTextureFile() {
    state.textureDataUrl = null;
    state.textureImageEl = null;
    state.bakedUrl = null;
    state.bakedImageEl = null;
    ui.textureInput.value = '';
    ui.textureFile.classList.add('hidden');
    ui.textureEmpty.classList.remove('hidden');
    ui.textureThumb.src = '';
    ui.projectionOptions.hidden = true;
    ui.btnDownloadTex.classList.add('hidden');
    drawUVLayout.now();
    if (state.viewMode === 'textured') setViewMode('checkered');
    refreshButtons();
}

function projectImageToUVs() {
    if (!state.vertices || !state.textureDataUrl || !state.uvInitial) return;
    const axis = ui.projectionAxis.value;
    // Front maps X/Y, side maps Z/Y, and top maps X/Z.
    const axes = axis === 'x' ? [2, 1] : axis === 'y' ? [0, 2] : [0, 1];
    const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
    for (const vertex of state.vertices) for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], vertex[c]);
        hi[c] = Math.max(hi[c], vertex[c]);
    }
    const spanU = Math.max(hi[axes[0]] - lo[axes[0]], Number.EPSILON);
    const spanV = Math.max(hi[axes[1]] - lo[axes[1]], Number.EPSILON);
    const projected = new Float64Array(state.vertices.length * 2);
    for (let i = 0; i < state.vertices.length; i++) {
        projected[2 * i] = (state.vertices[i][axes[0]] - lo[axes[0]]) / spanU;
        projected[2 * i + 1] = (state.vertices[i][axes[1]] - lo[axes[1]]) / spanV;
    }

    pushUndo();
    state.uv = projected;
    state.projection = { axis: axis.toUpperCase() };
    uvEd.version++;
    refreshIslandBoxes();
    clearIslandCache();
    syncUVToMesh();
    drawUVLayout.now();
    setViewMode('textured');
    updateStats();
    refreshButtons();
    toast(`Image projected from ${state.projection.axis}`, 'ok');
}

function restoreAutomaticUVs() {
    if (!state.uvInitial) return;
    pushUndo();
    state.uv = state.uvInitial.slice();
    state.projection = null;
    uvEd.version++;
    refreshIslandBoxes();
    clearIslandCache();
    syncUVToMesh();
    drawUVLayout.now();
    updateStats();
    refreshButtons();
    toast('Automatic UV unwrap restored', 'ok');
}

function syncProjectionState() {
    if (!state.projection || !state.uv || !state.uvInitial) return;
    for (let i = 0; i < state.uv.length; i++) if (state.uv[i] !== state.uvInitial[i]) return;
    state.projection = null;
}

async function applyTexture() {
    if (!state.uv || !state.textureDataUrl || state.busy) return;
    setBusy(true, 'Applying texture…');
    ui.btnApply.disabled = true;

    try {
        const pairs = uvPairs();
        const res = await fetch('/api/apply-texture', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mesh: {
                    vertices: state.vertices,
                    faces: state.faces,
                    uv: pairs,
                },
                texture_image: stripDataURL(state.textureDataUrl),
                uv_coordinates: pairs,
            }),
        });
        const data = await res.json();
        if (!res.ok || !data.success) {
            throw new Error(data.message || `Server error (${res.status})`);
        }

        state.bakedUrl = data.texture_url;
        const img = new Image();
        img.onload = () => {
            state.bakedImageEl = img;
            drawUVLayout.now();
        };
        img.src = state.bakedUrl;

        ui.btnDownloadTex.href = state.bakedUrl;
        ui.btnDownloadTex.classList.remove('hidden');
        toast('Texture applied — baked texture ready in UV layout', 'ok');
    } catch (err) {
        toast(err.message || 'Texture application failed', 'error', 6000);
    } finally {
        setBusy(false);
        refreshButtons();
    }
}

/* ============================================================
   OBJ export (client-side)
   ============================================================ */
function exportOBJ() {
    if (!state.vertices) return;
    const lines = ['# Exported by SmartUnwrapper'];
    for (const v of state.vertices)
        lines.push(`v ${v[0].toFixed(6)} ${v[1].toFixed(6)} ${v[2].toFixed(6)}`);
    if (state.uv)
        for (let i = 0; i < state.uv.length; i += 2)
            lines.push(`vt ${state.uv[i].toFixed(6)} ${state.uv[i + 1].toFixed(6)}`);
    for (const f of state.faces) {
        if (state.uv)
            lines.push(`f ${f[0] + 1}/${f[0] + 1} ${f[1] + 1}/${f[1] + 1} ${f[2] + 1}/${f[2] + 1}`);
        else
            lines.push(`f ${f[0] + 1} ${f[1] + 1} ${f[2] + 1}`);
    }
    const blob = new Blob([lines.join('\n') + '\n'], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'unwrapped.obj';
    a.click();
    URL.revokeObjectURL(a.href);
    toast('OBJ exported', 'ok');
}

/* ============================================================
   File routing (click, change, drag & drop)
   ============================================================ */
function routeFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (MODEL_EXTS.includes(ext)) setModelFile(file);
    else if (IMAGE_EXTS.includes(ext)) setTextureFile(file);
    else toast(`Unsupported file type: .${ext}`, 'error');
}

/* ============================================================
   Health pill
   ============================================================ */
async function checkHealth() {
    try {
        const d = await (await fetch('/api/health')).json();
        ui.healthPill.className = 'pill ok';
        ui.healthText.textContent = `API online · v${d.version}`;
    } catch {
        ui.healthPill.className = 'pill err';
        ui.healthText.textContent = 'API offline';
    }
}

async function loadTestSampleFromUrl() {
    const sampleName = new URLSearchParams(window.location.search).get('testSample');
    if (!sampleName) return;
    if (sampleName !== 'HammerChar') {
        toast('Unknown local test sample', 'error');
        return;
    }
    try {
        const response = await fetch(`/api/test/sample/${sampleName}.fbx`);
        if (!response.ok) throw new Error(`Sample request failed (${response.status})`);
        const file = new File([await response.blob()], `${sampleName}.fbx`, {
            type: 'application/octet-stream',
        });
        routeFile(file);
    } catch (err) {
        toast(err.message || 'Could not load local test sample', 'error', 6000);
    }
}

/* ============================================================
   Wire-up
   ============================================================ */
function bind() {
    // click-to-browse
    ui.modelDrop.addEventListener('click', (e) => {
        if (e.target.closest('.icon-btn')) return;
        ui.modelInput.click();
    });
    ui.textureDrop.addEventListener('click', (e) => {
        if (e.target.closest('.icon-btn')) return;
        ui.textureInput.click();
    });

    ui.modelInput.addEventListener('change', (e) => {
        if (e.target.files[0]) routeFile(e.target.files[0]);
    });
    ui.textureInput.addEventListener('change', (e) => {
        if (e.target.files[0]) routeFile(e.target.files[0]);
    });

    ui.modelClear.addEventListener('click', (e) => { e.stopPropagation(); clearModelFile(); });
    ui.textureClear.addEventListener('click', (e) => { e.stopPropagation(); clearTextureFile(); });

    // window-level drag & drop
    for (const ev of ['dragenter', 'dragover']) {
        window.addEventListener(ev, (e) => {
            e.preventDefault();
            document.body.classList.add('dragging');
        });
    }
    window.addEventListener('dragleave', (e) => {
        if (!e.relatedTarget) document.body.classList.remove('dragging');
    });
    window.addEventListener('drop', (e) => {
        e.preventDefault();
        document.body.classList.remove('dragging');
        const file = [...e.dataTransfer.files][0];
        if (file) routeFile(file);
    });

    ui.btnUnwrap.addEventListener('click', unwrap);
    ui.btnProject.addEventListener('click', projectImageToUVs);
    ui.btnRestoreUv.addEventListener('click', restoreAutomaticUVs);
    ui.btnApply.addEventListener('click', applyTexture);
    ui.btnExport.addEventListener('click', exportOBJ);

    const syncSam2Controls = () => {
        ui.sam2Options.hidden = ui.methodSelect.value !== 'SAM2';
        ui.partDetectionValue.value = Number(ui.partDetection.value).toFixed(2);
        ui.sam2IouValue.value = Number(ui.sam2Iou.value).toFixed(2);
        ui.sam2StabilityValue.value = Number(ui.sam2Stability.value).toFixed(2);
    };
    ui.methodSelect.addEventListener('change', () => {
        syncSam2Controls();
        refreshButtons();
    });
    [ui.partDetection, ui.sam2Iou, ui.sam2Stability].forEach((input) => {
        input.addEventListener('input', syncSam2Controls);
    });
    syncSam2Controls();

    ui.viewModes.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-mode]');
        if (btn) setViewMode(btn.dataset.mode);
    });

    /* ---- UV editor ---- */
    ui.uvCanvas.addEventListener('pointerdown', uvPointerDown);
    ui.uvCanvas.addEventListener('pointermove', (e) => {
        uvEd.lastPointer = eventUV(e);
        uvPointerMove(e);
    });
    ui.uvCanvas.addEventListener('pointerup', uvPointerUp);
    ui.uvCanvas.addEventListener('pointercancel', uvPointerUp);
    ui.uvCanvas.addEventListener('contextmenu', (e) => e.preventDefault());
    ui.uvCanvas.addEventListener('keydown', uvKeyDown);
    ui.uvCanvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        if (!state.uv) return;
        zoomAt(eventUV(e), Math.exp(-e.deltaY * 0.0016));
    }, { passive: false });

    ui.uvTools.addEventListener('click', (e) => {
        const tool = e.target.closest('button[data-tool]');
        if (tool) { setUvTool(tool.dataset.tool); return; }
        if (e.target.closest('#uv-fit')) { resetView(); return; }
        if (e.target.closest('#uv-reset')) { resetUvLayout(); }
    });

    setInterval(checkHealth, 30000);
}

initThree();
bind();
checkHealth();
drawUVLayout();
updateStats();
refreshButtons();
loadTestSampleFromUrl();

if (state.threeFailed) {
    ui.viewportEmpty.innerHTML = '<p>3D preview unavailable (three.js CDN unreachable)</p><p class="hint">You can still unwrap and inspect the UV layout</p>';
}

/* Exposed for debugging / automated checks in the browser console:
   __smart.state.uv, __smart.uvEd.selection, __smart.uvPairs() ... */
window.__smart = { state, uvEd, uvPairs, drawUVLayout };
