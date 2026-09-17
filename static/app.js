/* XLB Workbench - plain browser frontend.  All field data shown in the UI comes
 * from the workbench HTTP API; local drawing is limited to geometry and mesh
 * previews. */

(function () {
  'use strict';

  const FACE_INFO = {
    xmin: { label: 'X 最小面', short: 'xmin', axis: '−X' },
    xmax: { label: 'X 最大面', short: 'xmax', axis: '+X' },
    ymin: { label: 'Y 最小面', short: 'ymin', axis: '−Y' },
    ymax: { label: 'Y 最大面', short: 'ymax', axis: '+Y' },
    zmin: { label: 'Z 最小面', short: 'zmin', axis: '−Z' },
    zmax: { label: 'Z 最大面', short: 'zmax', axis: '+Z' },
    cad: { label: 'CAD 表面', short: 'cad', axis: 'CAD' }
  };
  const FACES = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'];
  // Synthetic box faces are ordered z-, z+, y-, x+, y+, x-.
  const BOX_FACE_GROUPS = ['zmin', 'zmax', 'ymin', 'xmax', 'ymax', 'xmin'];
  const FIELD_INFO = {
    temperature: { label: '温度', title: '温度分布', unit: 'K', color: 'temperature' },
    speed: { label: '速度', title: '速度ノルム', unit: 'm/s', color: 'speed' },
    velocity_x: { label: '速度 X', title: '速度 X 成分', unit: 'm/s', color: 'speed' },
    velocity_y: { label: '速度 Y', title: '速度 Y 成分', unit: 'm/s', color: 'speed' },
    velocity_z: { label: '速度 Z', title: '速度 Z 成分', unit: 'm/s', color: 'speed' },
    pressure: { label: '圧力', title: '圧力分布', unit: 'Pa', color: 'pressure' },
    density: { label: '材料密度', title: '材料密度分布', unit: 'kg/m³', color: 'density' },
    eddy_viscosity: { label: '渦粘性', title: '渦粘性係数', unit: 'm²/s', color: 'eddy_viscosity' },
    vorticity: { label: '渦度', title: '渦度ノルム', unit: 's⁻¹', color: 'vorticity' },
    vorticity_x: { label: '渦度 X', title: '渦度 X 成分', unit: 's⁻¹', color: 'vorticity' },
    vorticity_y: { label: '渦度 Y', title: '渦度 Y 成分', unit: 's⁻¹', color: 'vorticity' },
    vorticity_z: { label: '渦度 Z', title: '渦度 Z 成分', unit: 's⁻¹', color: 'vorticity' },
    heat_flux: { label: '伝導熱流束', title: '伝導熱流束ノルム', unit: 'W/m²', color: 'heat_flux' },
    heat_flux_x: { label: '伝導熱流束 X', title: '伝導熱流束 X 成分', unit: 'W/m²', color: 'heat_flux' },
    heat_flux_y: { label: '伝導熱流束 Y', title: '伝導熱流束 Y 成分', unit: 'W/m²', color: 'heat_flux' },
    heat_flux_z: { label: '伝導熱流束 Z', title: '伝導熱流束 Z 成分', unit: 'W/m²', color: 'heat_flux' },
    conductivity: { label: '熱伝導率', title: '熱伝導率分布', unit: 'W/(m·K)', color: 'conductivity' },
    dynamic_viscosity: { label: '粘度', title: '粘度分布', unit: 'Pa·s', color: 'dynamic_viscosity' },
    heat_capacity: { label: '比熱容量', title: '比熱容量分布', unit: 'J/(kg·K)', color: 'heat_capacity' }
  };
  const RESULT_FIELD_ORDER = [
    'temperature', 'speed', 'velocity_x', 'velocity_y', 'velocity_z', 'pressure', 'density',
    'eddy_viscosity', 'vorticity', 'vorticity_x', 'vorticity_y', 'vorticity_z',
    'heat_flux', 'heat_flux_x', 'heat_flux_y', 'heat_flux_z', 'conductivity',
    'dynamic_viscosity', 'heat_capacity'
  ];
  const DEFAULT_GRAVITY = {
    enabled: false,
    mode: 'uniform',
    vector: [0, 0, -9.80665],
    reference_temperature: 293.15
  };
  const DEFAULT_MONITOR = {
    tolerance: 1.0e-5,
    consecutive_samples: 5
  };
  const MONITOR_RESIDUAL_KIND = 'sample_change_linf_v1';
  const MONITOR_COMPONENTS = [
    { key: 'velocity_residual', changeKey: 'velocity_change_max', label: '速度', unit: 'm/s', color: '#4fb9e9' },
    { key: 'pressure_residual', changeKey: 'pressure_change_max', label: '圧力', unit: 'Pa', color: '#eab765' },
    { key: 'temperature_residual', changeKey: 'temperature_change_max', label: '温度', unit: 'K', color: '#5cd29d' }
  ];
  const DEFAULT_PROJECT = {
    schema_version: 1,
    name: '無題のモデル',
    geometry: { kind: 'box', size: [0.06, 0.02, 0.02], asset_id: null, role: 'fluid' },
    materials: [{
      id: 'water', name: 'Water',
      density: { kind: 'constant', value: 998 },
      viscosity: { kind: 'constant', value: 0.001 },
      heat_capacity: { kind: 'constant', value: 4182 },
      conductivity: { kind: 'constant', value: 0.6 }
    }],
    physics: { flow: true, thermal: true, material_id: 'water', initial_temperature: 293.15, gravity: { ...DEFAULT_GRAVITY, vector: [...DEFAULT_GRAVITY.vector] }, turbulence: { model: 'laminar', smagorinsky_constant: 0.17, turbulent_prandtl: 0.9 } },
    boundaries: [
      { id: 'inlet', name: 'Inlet', face: 'xmin', flow: { type: 'velocity', velocity: [0.01, 0, 0] }, thermal: { type: 'temperature', value: 293.15 } },
      { id: 'outlet', name: 'Outlet', face: 'xmax', flow: { type: 'pressure', value: 0 }, thermal: { type: 'adiabatic' } },
      { id: 'heated-wall', name: 'Heated wall', face: 'zmax', flow: { type: 'wall' }, thermal: { type: 'heat_flux', value: 1000 } }
    ],
    mesh: { cells: [30, 10, 10] },
    study: { steps: 200, output_interval: 20, snapshot_interval: 20, dt: 0.01, device: 'cpu', gpu_batch_cells: 65536, monitor: { ...DEFAULT_MONITOR } }
  };

  const state = {
    project: null,
    projectId: null,
    projects: [],
    runs: [],
    selectedNode: 'geometry',
    dirty: false,
    mesh: null,
    run: null,
    runTimer: null,
    framesTimer: null,
    slice: null,
    resultControls: null,
    sliceRequest: 0,
    sliceLoading: false,
    frames: [],
    resultFields: [],
    framesRequest: 0,
    framesLoading: false,
    framesEndpointUnavailable: false,
    selectedFrameStep: null,
    playbackTimer: null,
    playbackActive: false,
    followLatest: false,
    // Restart metadata is kept separately from the editable project.  A
    // restart always uses the source run's checkpoint and must not mark or
    // save the current project.
    restartCache: new Map(),
    restartAvailability: null,
    restartAvailabilityKey: null,
    restartAvailabilityLoading: false,
    restartAvailabilityRequest: 0,
    restartSubmitRequest: 0,
    restartAdditionalSteps: 200,
    restartSubmitting: false,
    restartNotice: '',
    resultLineage: null,
    materialPresets: [],
    materialPresetsLoading: false,
    materialPresetsAttempted: false,
    materialPresetsRequest: 0,
    materialPresetsError: '',
    materialPresetSelectedId: '',
    materialPresetAppliedMaterialRef: null,
    materialPresetAppliedId: '',
    materialPresetAppliedSignature: '',
    materialPresetEdited: false,
    materialTablePages: {},
    materialCsvInputMode: null,
    materialCsvRequest: 0,
    materialCsvPreview: null,
    materialCsvLoading: false,
    history: [],
    logs: [],
    validation: [],
    activeDock: 'log',
    view: 'geometry',
    showGrid: true,
    orbit: { yaw: -0.58, pitch: 0.34, zoom: 1 },
    drag: null,
    cadFile: null,
    cadAssetMeta: null,
    selectedSurface: null,
    projectedFaces: [],
    serverOnline: null,
    limits: null
  };

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));
  const clone = (value) => JSON.parse(JSON.stringify(value));
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const numberOr = (value, fallback = 0) => {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  // History rows intentionally use null for baselines and inactive physics.
  // Keep that distinction all the way through the UI; Number(null) would turn
  // an unavailable value into a misleading zero on a log plot.
  const nullableNumber = (value, fallback = null) => {
    if (value === null || value === undefined || (typeof value === 'string' && value.trim() === '')) return fallback;
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };
  const nullableBoolean = (value, fallback = null) => {
    if (value === null || value === undefined || (typeof value === 'string' && value.trim() === '')) return fallback;
    if (typeof value === 'boolean') return value;
    if (typeof value === 'number') return value !== 0;
    const normalized = String(value).trim().toLowerCase();
    if (normalized === 'true' || normalized === '1' || normalized === 'yes') return true;
    if (normalized === 'false' || normalized === '0' || normalized === 'no') return false;
    return fallback;
  };
  function normalizeMonitor(monitor) {
    const source = monitor && typeof monitor === 'object' && !Array.isArray(monitor) ? monitor : {};
    const tolerance = nullableNumber(source.tolerance, DEFAULT_MONITOR.tolerance);
    const consecutive = nullableNumber(source.consecutive_samples, DEFAULT_MONITOR.consecutive_samples);
    return {
      ...source,
      tolerance: tolerance === null ? DEFAULT_MONITOR.tolerance : tolerance,
      consecutive_samples: consecutive === null ? DEFAULT_MONITOR.consecutive_samples : Math.round(consecutive)
    };
  }
  function hasOwn(object, key) {
    return Boolean(object && Object.prototype.hasOwnProperty.call(object, key));
  }
  const intOr = (value, fallback = 1) => Math.max(1, Math.round(numberOr(value, fallback)));
  const uid = (prefix = 'bc') => {
    if (window.crypto && typeof window.crypto.randomUUID === 'function') return `${prefix}-${window.crypto.randomUUID().slice(0, 8)}`;
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  };
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (character) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[character]));
  const fmt = (value, digits = 3) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    return n.toLocaleString('en-US', { maximumFractionDigits: digits });
  };
  const fmtSci = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n)) return '—';
    if (n === 0) return '0';
    if (Math.abs(n) >= 0.01 && Math.abs(n) < 1000) return n.toPrecision(4);
    return n.toExponential(3).replace('e+', 'e');
  };
  const localTime = () => new Date().toLocaleTimeString('ja-JP', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  async function request(path, options = {}) {
    const headers = { ...(options.headers || {}) };
    if (options.body && typeof options.body !== 'string' && !(options.body instanceof Blob) && !(options.body instanceof ArrayBuffer)) {
      headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, { ...options, headers });
    const contentType = response.headers.get('content-type') || '';
    let payload;
    if (contentType.includes('application/json')) {
      payload = await response.json().catch(() => ({}));
    } else {
      payload = await response.text().catch(() => '');
    }
    if (!response.ok) {
      const detail = typeof payload === 'object' ? payload.error : payload;
      throw new Error(detail || `HTTP ${response.status}`);
    }
    return payload;
  }

  function currentMaterial() {
    const materials = state.project && Array.isArray(state.project.materials) ? state.project.materials : [];
    if (!materials.length) return null;
    const selected = state.project.physics && state.project.physics.material_id;
    return materials.find((material) => material.id === selected) || materials[0];
  }

  function normalizeProject(input) {
    const project = clone(input || DEFAULT_PROJECT);
    project.schema_version = 1;
    project.name = String(project.name || '無題のモデル');
    project.geometry = project.geometry || clone(DEFAULT_PROJECT.geometry);
    project.geometry.kind = project.geometry.kind === 'cad' ? 'cad' : 'box';
    project.geometry.role = project.geometry.role === 'obstacle' ? 'obstacle' : 'fluid';
    project.geometry.size = Array.isArray(project.geometry.size) && project.geometry.size.length === 3
      ? project.geometry.size.map((value, index) => numberOr(value, DEFAULT_PROJECT.geometry.size[index]))
      : clone(DEFAULT_PROJECT.geometry.size);
    project.geometry.asset_id = project.geometry.asset_id || null;
    if (Array.isArray(project.geometry.solids)) project.geometry.solids = project.geometry.solids.map(normalizeSolid).filter(Boolean);
    if (project.geometry.solid_material_id !== undefined) project.geometry.solid_material_id = project.geometry.solid_material_id || null;
    project.materials = Array.isArray(project.materials) && project.materials.length ? project.materials : clone(DEFAULT_PROJECT.materials);
    project.materials = project.materials.map((material, index) => normalizeMaterial(material, index));
    const sourcePhysics = project.physics || {};
    const hasTurbulence = Object.prototype.hasOwnProperty.call(sourcePhysics, 'turbulence');
    const hasGravity = Object.prototype.hasOwnProperty.call(sourcePhysics, 'gravity');
    project.physics = { ...clone(DEFAULT_PROJECT.physics), ...sourcePhysics };
    project.physics.flow = project.physics.flow !== false;
    project.physics.thermal = project.physics.thermal !== false;
    project.physics.material_id = project.materials.some((material) => material.id === project.physics.material_id)
      ? project.physics.material_id : project.materials[0].id;
    project.physics.initial_temperature = numberOr(project.physics.initial_temperature, 293.15);
    if (hasTurbulence) project.physics.turbulence = normalizeTurbulence(project.physics.turbulence);
    else delete project.physics.turbulence;
    if (hasGravity) project.physics.gravity = normalizeGravity(project.physics.gravity);
    else delete project.physics.gravity;
    project.boundaries = Array.isArray(project.boundaries) ? project.boundaries.map(normalizeBoundary).filter(Boolean) : [];
    project.mesh = { ...clone(DEFAULT_PROJECT.mesh), ...(project.mesh || {}) };
    project.mesh.cells = Array.isArray(project.mesh.cells) && project.mesh.cells.length === 3
      ? project.mesh.cells.map((value, index) => intOr(value, DEFAULT_PROJECT.mesh.cells[index])) : clone(DEFAULT_PROJECT.mesh.cells);
    const sourceStudy = project.study || {};
    const hasSnapshotInterval = Object.prototype.hasOwnProperty.call(sourceStudy, 'snapshot_interval');
    const hasMonitor = Object.prototype.hasOwnProperty.call(sourceStudy, 'monitor');
    project.study = { ...clone(DEFAULT_PROJECT.study), ...sourceStudy };
    project.study.steps = intOr(project.study.steps, 200);
    project.study.output_interval = intOr(project.study.output_interval, 20);
    // New projects use the default interval.  Existing projects that predate
    // frame snapshots omit the key and retain the compatible disabled value.
    project.study.snapshot_interval = hasSnapshotInterval
      ? Math.max(0, Math.round(numberOr(project.study.snapshot_interval, 0)))
      : (input ? 0 : DEFAULT_PROJECT.study.snapshot_interval);
    project.study.dt = numberOr(project.study.dt, 0.001);
    project.study.device = ['cpu', 'cuda:0', 'cuda:0-ram'].includes(String(project.study.device || '')) ? String(project.study.device) : 'cpu';
    const gpuBatchCells = Number(project.study.gpu_batch_cells);
    project.study.gpu_batch_cells = Number.isInteger(gpuBatchCells) && gpuBatchCells >= 1024 && gpuBatchCells <= 1048576 ? gpuBatchCells : 65536;
    // Monitor settings are optional in the project schema.  New projects keep
    // the default descriptor, while an older project that has never exposed
    // monitor settings remains byte-compatible when it is saved unchanged.
    if (hasMonitor) project.study.monitor = normalizeMonitor(sourceStudy.monitor);
    else if (input) delete project.study.monitor;
    return project;
  }

  function normalizeMaterial(material, index) {
    const base = clone(DEFAULT_PROJECT.materials[0]);
    const normalized = { ...base, ...(material || {}) };
    normalized.id = String(normalized.id || (index === 0 ? 'water' : uid('mat')));
    normalized.name = String(normalized.name || `Material ${index + 1}`);
    ['density', 'viscosity', 'heat_capacity', 'conductivity'].forEach((key) => {
      const property = normalized[key] || base[key];
      if (property.kind === 'table' && Array.isArray(property.points)) {
        normalized[key] = { kind: 'table', points: property.points.map((point) => [numberOr(point[0], 293.15), numberOr(point[1], 0)]) };
      } else {
        normalized[key] = { kind: 'constant', value: numberOr(property.value, base[key].value) };
      }
    });
    return normalized;
  }

  function normalizeTurbulence(turbulence) {
    const source = turbulence || {};
    return {
      model: source.model === 'smagorinsky' ? 'smagorinsky' : 'laminar',
      smagorinsky_constant: numberOr(source.smagorinsky_constant, 0.17),
      turbulent_prandtl: numberOr(source.turbulent_prandtl, 0.9)
    };
  }

  function normalizeGravity(gravity) {
    const source = gravity && typeof gravity === 'object' && !Array.isArray(gravity) ? gravity : {};
    const vector = Array.isArray(source.vector) && source.vector.length === 3
      ? source.vector.map((value, index) => numberOr(value, DEFAULT_GRAVITY.vector[index]))
      : [...DEFAULT_GRAVITY.vector];
    return {
      ...source,
      enabled: typeof source.enabled === 'boolean' ? source.enabled : DEFAULT_GRAVITY.enabled,
      mode: source.mode === 'buoyancy' ? 'buoyancy' : DEFAULT_GRAVITY.mode,
      vector,
      reference_temperature: numberOr(source.reference_temperature, DEFAULT_GRAVITY.reference_temperature)
    };
  }

  function ensureGravity(project = state.project) {
    if (!project?.physics) return null;
    if (!Object.prototype.hasOwnProperty.call(project.physics, 'gravity')) project.physics.gravity = normalizeGravity(null);
    else if (!project.physics.gravity || typeof project.physics.gravity !== 'object' || Array.isArray(project.physics.gravity)) project.physics.gravity = normalizeGravity(project.physics.gravity);
    return project.physics.gravity;
  }

  function normalizeSolid(solid, index = 0) {
    if (!solid) return null;
    const item = clone(solid);
    item.id = String(item.id || uid(`solid${index}`));
    item.name = String(item.name || `Solid ${index + 1}`);
    item.origin = Array.isArray(item.origin) && item.origin.length === 3 ? item.origin.map((value) => numberOr(value, 0)) : [0, 0, 0];
    item.size = Array.isArray(item.size) && item.size.length === 3 ? item.size.map((value) => numberOr(value, 0.01)) : [0.01, 0.01, 0.01];
    item.material_id = String(item.material_id || state.project?.materials?.[0]?.id || 'water');
    return item;
  }

  function isCadFace(face) { return typeof face === 'string' && (face === 'cad' || /^cad:[A-Za-z0-9_.:-]+$/.test(face)); }
  function isKnownFace(face) { return Boolean(FACE_INFO[face]) || isCadFace(face); }
  function surfaceGroups() {
    const metadata = state.cadAssetMeta || state.mesh?.preview || state.mesh || {};
    const groups = metadata.surface_groups || metadata.surfaceGroups || [];
    if (Array.isArray(groups)) return groups;
    if (groups && typeof groups === 'object') return Object.entries(groups).map(([id, value]) => ({ id, ...(value || {}) }));
    return [];
  }
  function surfaceGroupId(group) { return String(group?.id || group?.patch_id || group?.name || 'surface'); }
  function surfaceGroupInfo(groupId) { return surfaceGroups().find((group) => surfaceGroupId(group) === String(groupId)) || null; }
  function faceLabel(face) {
    if (FACE_INFO[face]) return FACE_INFO[face].label;
    if (isCadFace(face)) {
      const groupId = String(face).slice(4);
      const group = surfaceGroupInfo(groupId);
      return group?.name || groupId || 'CAD 表面';
    }
    return String(face || '境界');
  }

  function normalizeBoundary(boundary) {
    if (!boundary) return null;
    const normalized = clone(boundary);
    normalized.id = String(normalized.id || uid('bc'));
    normalized.name = String(normalized.name || faceLabel(normalized.face) || 'Boundary');
    normalized.face = isKnownFace(normalized.face) ? normalized.face : 'xmin';
    normalized.flow = normalized.flow || { type: 'wall' };
    if (normalized.face === 'cad') normalized.flow = { type: 'wall' };
    normalized.flow.type = ['velocity', 'pressure', 'wall'].includes(normalized.flow.type) ? normalized.flow.type : 'wall';
    if (normalized.flow.type === 'velocity') {
      normalized.flow.velocity = Array.isArray(normalized.flow.velocity) && normalized.flow.velocity.length === 3
        ? normalized.flow.velocity.map((value) => numberOr(value, 0)) : [0, 0, 0];
    }
    if (normalized.flow.type === 'pressure') normalized.flow.value = numberOr(normalized.flow.value, 0);
    normalized.thermal = normalized.thermal || { type: 'adiabatic' };
    normalized.thermal.type = ['temperature', 'heat_flux', 'convection', 'adiabatic'].includes(normalized.thermal.type)
      ? normalized.thermal.type : 'adiabatic';
    if (normalized.thermal.type === 'temperature') normalized.thermal.value = numberOr(normalized.thermal.value, 293.15);
    if (normalized.thermal.type === 'heat_flux') normalized.thermal.value = numberOr(normalized.thermal.value, 0);
    if (normalized.thermal.type === 'convection') {
      normalized.thermal.h = numberOr(normalized.thermal.h, 10);
      normalized.thermal.ambient_temperature = numberOr(normalized.thermal.ambient_temperature, 293.15);
    }
    return normalized;
  }

  function setByPath(root, path, value) {
    const parts = path.split('.');
    let target = root;
    for (let index = 0; index < parts.length - 1; index += 1) {
      const key = parts[index];
      if (target[key] === undefined || target[key] === null) target[key] = /^\d+$/.test(parts[index + 1]) ? [] : {};
      target = target[key];
    }
    target[parts[parts.length - 1]] = value;
  }

  function boundaryForFace(face) {
    return state.project.boundaries.find((boundary) => boundary.face === face) || null;
  }

  function ensureBoundary(face) {
    let boundary = boundaryForFace(face);
    if (!boundary) {
      boundary = normalizeBoundary({ id: uid('bc'), name: faceLabel(face) || 'Boundary', face, flow: { type: 'wall' }, thermal: { type: 'adiabatic' } });
      state.project.boundaries.push(boundary);
      state.dirty = true;
      renderTree();
    }
    return boundary;
  }

  function projectLabel() {
    return state.project ? state.project.name : '無題のモデル';
  }

  function setProject(project, id = null, runs = []) {
    stopPlayback();
    window.clearTimeout(state.framesTimer);
    state.project = normalizeProject(project);
    state.projectId = id;
    state.run = null;
    state.runs = Array.isArray(runs) ? runs.map((run) => normalizeRun(run)) : [];
    state.dirty = false;
    state.mesh = null;
    state.slice = null; state.resultControls = null;
    state.frames = [];
    state.resultFields = [];
    state.framesLoading = false;
    state.framesEndpointUnavailable = false;
    state.selectedFrameStep = null;
    state.followLatest = false;
    state.restartCache = new Map();
    state.restartAvailability = null;
    state.restartAvailabilityKey = null;
    state.restartAvailabilityLoading = false;
    state.restartAvailabilityRequest += 1;
    state.restartSubmitRequest += 1;
    state.restartAdditionalSteps = 200;
    state.restartSubmitting = false;
    state.restartNotice = '';
    state.resultLineage = null;
    state.materialPresetSelectedId = '';
    state.materialPresetAppliedMaterialRef = null;
    state.materialPresetAppliedId = '';
    state.materialPresetAppliedSignature = '';
    state.materialPresetEdited = false;
    state.materialTablePages = {};
    invalidateMaterialCsvPreview(false);
    state.cadAssetMeta = null;
    state.selectedSurface = null;
    state.projectedFaces = [];
    state.history = [];
    state.selectedNode = 'geometry';
    // A newly opened/imported model is a model-definition context.  Reset the
    // canvas tab as well as the tree selection so a previous result view does
    // not leave the user looking at an empty result panel.
    state.view = 'geometry';
    updateHeader();
    renderTree();
    renderInspector();
    drawAll();
    switchView('geometry');
    renderHistory();
    if (state.project.geometry.kind === 'cad' && state.project.geometry.asset_id) loadAssetMeta(state.project.geometry.asset_id);
  }

  async function loadAssetMeta(assetId) {
    try {
      const payload = await request(`/api/assets/${encodeURIComponent(assetId)}`);
      const asset = payload.asset || payload;
      if (state.project?.geometry?.asset_id === assetId) { state.cadAssetMeta = asset; renderTree(); drawGeometry(); renderInspector(); }
    } catch (error) {
      pushLog(`CAD アセット情報を取得できません: ${error.message}`, 'warn');
    }
  }

  function markDirty() {
    state.dirty = true;
    updateHeader();
    updateValidationSummary();
  }

  function updateHeader() {
    const title = $('#projectTitle');
    const summaryName = $('#modelSummaryName');
    const summaryType = $('#modelSummaryType');
    const dirty = $('#dirtyBadge');
    const summaryState = $('#modelSummaryState');
    if (!title || !state.project) return;
    title.textContent = projectLabel();
    summaryName.textContent = projectLabel();
    summaryType.textContent = state.projectId ? `保存済み / ${state.projectId.slice(0, 8)}` : '未保存プロジェクト';
    dirty.classList.toggle('hidden', !state.dirty);
    summaryState.textContent = state.dirty ? '編集' : '同期済み';
    summaryState.classList.toggle('saved', !state.dirty);
    const cellCount = state.project.mesh.cells.reduce((acc, value) => acc * value, 1);
    const estimate = state.meshEstimateKey === JSON.stringify(serializableProject()) ? state.meshEstimate : null;
    $('#statusCellCount').textContent = state.mesh ? fmt(state.mesh.total_cells || cellCount, 0) : estimate?.total_cells ? fmt(estimate.total_cells, 0) : state.project.mesh.target_cells !== undefined ? `目標 ${fmt(state.project.mesh.target_cells, 0)}` : fmt(cellCount, 0);
  }

  function updateConnection(online, message) {
    const badge = $('#connectionBadge');
    if (!badge) return;
    badge.className = `status-badge ${online ? 'online' : 'offline'}`;
    badge.innerHTML = `<span class="status-dot"></span>${esc(message || (online ? 'API 接続中' : 'API 未接続'))}`;
    state.serverOnline = online;
  }

  function pushLog(message, level = 'info') {
    state.logs.push({ at: localTime(), message: String(message), level });
    if (state.logs.length > 160) state.logs.shift();
    renderLogs();
  }

  function toast(message, level = 'info') {
    const region = $('#toastRegion');
    if (!region) return;
    const element = document.createElement('div');
    element.className = `toast ${level}`;
    element.textContent = message;
    region.appendChild(element);
    window.setTimeout(() => element.remove(), 4600);
  }

  function setRunState(label, running = false) {
    const element = $('#ribbonRunState');
    if (!element) return;
    element.classList.toggle('running', running);
    element.innerHTML = `<span class="pulse-dot"></span><span>${esc(label)}</span>`;
    const stop = $('[data-action="stop-simulation"]');
    if (stop) stop.disabled = !running;
  }

  async function loadInitial() {
    setProject(DEFAULT_PROJECT);
    pushLog('ワークベンチを初期化しています。');
    try {
      const project = await request('/api/default');
      project.mesh.target_cells = project.mesh.cells.reduce((a, b) => a * b, 1);
      setProject(project);
      updateConnection(true, 'API 接続中');
      pushLog('サーバーから標準プロジェクトを読み込みました。');
    } catch (error) {
      updateConnection(false, 'API 未接続');
      pushLog(`標準プロジェクトを取得できません: ${error.message}`, 'warn');
      toast('API に接続できないため、ローカルの初期値を表示しています。', 'warn');
    }
    try {
      await refreshProjects();
    } catch (error) {
      pushLog(`プロジェクト一覧を取得できません: ${error.message}`, 'warn');
    }
    try {
      state.limits = (await request('/api/health')).limits || null;
      if (state.selectedNode === 'mesh') renderInspector();
    } catch (_) { /* Limits are informational; server validation remains authoritative. */ }
  }

  async function refreshProjects() {
    const payload = await request('/api/projects');
    state.projects = Array.isArray(payload) ? payload : (payload.projects || []);
    return state.projects;
  }

  function treeButton(node, symbol, name, meta = '', extraClass = '', alert = false) {
    const active = state.selectedNode === node ? ' active' : '';
    return `<button type="button" role="treeitem" aria-selected="${state.selectedNode === node}" class="tree-item${active} ${extraClass}" data-action="select-node" data-node="${esc(node)}"><span class="tree-symbol">${symbol}</span><span class="tree-name">${esc(name)}</span>${meta ? `<span class="tree-meta">${esc(meta)}</span>` : ''}${alert ? '<span class="tree-alert" title="要確認">!</span>' : ''}</button>`;
  }

  function renderTree() {
    const tree = $('#modelTree');
    if (!tree || !state.project) return;
    const geometry = state.project.geometry;
    const material = currentMaterial();
    const hasMesh = Boolean(state.mesh);
    const boundaries = state.project.boundaries || [];
    const boundaryByFace = Object.fromEntries(boundaries.map((boundary) => [boundary.face, boundary]));
    const geometryMeta = geometry.kind === 'cad' ? 'CAD' : `${fmt(geometry.size[0] * 1000, 0)} mm`;
    const materialMeta = material ? material.name : '未定義';
    const turbulence = state.project.physics.turbulence;
    const gravity = normalizeGravity(state.project.physics.gravity);
    const physicsMeta = [state.project.physics.flow ? '流れ' : '', state.project.physics.thermal ? '熱' : '', turbulence?.model === 'smagorinsky' ? 'LES' : '', gravity.enabled ? (gravity.mode === 'buoyancy' ? '浮力' : '重力') : ''].filter(Boolean).join(' + ') || '無効';
    const meshMeta = hasMesh ? `${fmt(state.mesh.fluid_cells || 0, 0)} fluid` : state.project.mesh.target_cells !== undefined ? `目標 ${fmt(state.project.mesh.target_cells, 0)}` : `${state.project.mesh.cells.join(' × ')}`;
    const boundaryStatus = `${boundaries.length} 定義`;
    const cadGroups = geometry.kind === 'cad' ? surfaceGroups() : [];
    const cadBoundaries = boundaries.filter((boundary) => isCadFace(boundary.face) && boundary.face !== 'cad' && !cadGroups.some((group) => `cad:${surfaceGroupId(group)}` === boundary.face));
    const solidCount = Array.isArray(geometry.solids) ? geometry.solids.length : 0;
    const cadSolid = geometry.kind === 'cad' && Boolean(geometry.solid_material_id);
    const solidDomainMeta = state.run?.solid_cells ? `solid ${fmt(state.run.solid_cells, 0)}` : cadSolid ? 'CAD CHT' : solidCount ? `${solidCount} solid` : '';
    const resultMeta = state.run ? `${String(state.run.status || 'queued')}${solidDomainMeta ? ` · ${solidDomainMeta}` : ''}` : (solidDomainMeta || '未実行');
    tree.innerHTML = [
      '<div class="tree-group">',
      '<div class="tree-group-label"><span class="tree-caret">⌄</span>モデル定義</div>',
      treeButton('geometry', '◇', 'ジオメトリ', geometryMeta, '', geometry.kind === 'cad' && !geometry.asset_id),
      solidCount ? treeButton('geometry', '▧', `固体領域 (${solidCount})`, 'CHT', 'tree-subitem') : '',
      cadSolid ? treeButton('geometry', '▧', 'CAD 固体領域', 'CHT', 'tree-subitem') : '',
      treeButton('material', '●', '材料', materialMeta),
      treeButton('physics', '◌', '物理場', physicsMeta),
      '</div>',
      '<div class="tree-group">',
      '<div class="tree-group-label"><span class="tree-caret">⌄</span>境界条件 <span class="tree-meta">' + boundaryStatus + '</span></div>',
      ...FACES.map((face) => {
        const boundary = boundaryByFace[face];
        const label = boundary ? boundary.name : FACE_INFO[face].label;
        const meta = boundary ? `${boundary.flow.type} / ${boundary.thermal.type}` : '既定';
        return treeButton(`face:${face}`, boundary ? '◉' : '○', label, meta, '', false);
      }),
      ...cadGroups.map((group) => {
        const id = surfaceGroupId(group); const face = `cad:${id}`; const boundary = boundaryByFace[face];
        return treeButton(`face:${face}`, boundary ? '◉' : '◇', boundary?.name || group.name || id, boundary ? `${boundary.flow.type} / ${boundary.thermal.type}` : `${group.triangle_count ?? group.triangleCount ?? '—'} triangles`);
      }),
      ...cadBoundaries.map((boundary) => treeButton(`face:${boundary.face}`, '◉', boundary.name, `${boundary.face} / ${boundary.thermal.type}`)),
      boundaries.find((boundary) => boundary.face === 'cad') ? treeButton('face:cad', '◉', boundaries.find((boundary) => boundary.face === 'cad').name, 'cad') : '',
      '<button type="button" class="tree-item tree-add" data-action="add-boundary"><span class="tree-symbol">＋</span><span class="tree-name">境界を追加</span></button>',
      '</div>',
      '<div class="tree-group">',
      '<div class="tree-group-label"><span class="tree-caret">⌄</span>計算</div>',
      treeButton('mesh', '▦', 'メッシュ', meshMeta, '', !hasMesh),
      treeButton('study', '◫', 'スタディ', `${state.project.study.steps} steps`),
      '</div>',
      '<div class="tree-group">',
      '<div class="tree-group-label"><span class="tree-caret">⌄</span>結果</div>',
      treeButton('results', '◒', '結果データ', resultMeta),
      '</div>'
    ].join('');
    $('#modelSummaryState').textContent = state.run && ['queued', 'running', 'stopping'].includes(state.run.status) ? '計算中' : (state.dirty ? '編集' : '同期済み');
  }

  function renderInspector() {
    const inspector = $('#inspector');
    if (!inspector || !state.project) return;
    let title = 'ジオメトリ';
    let type = 'GEOMETRY';
    let content = '';
    if (state.selectedNode === 'geometry') {
      title = 'ジオメトリ'; type = 'GEOMETRY'; content = geometryInspector();
    } else if (state.selectedNode === 'material') {
      title = '材料'; type = 'MATERIAL'; content = materialInspector();
    } else if (state.selectedNode === 'physics') {
      title = '物理場'; type = 'PHYSICS'; content = physicsInspector();
    } else if (state.selectedNode.startsWith('face:')) {
      const face = state.selectedNode.slice(5);
      title = faceLabel(face) || '境界条件'; type = 'BOUNDARY'; content = boundaryInspector(face);
    } else if (state.selectedNode === 'mesh') {
      title = 'メッシュ'; type = 'MESH'; content = meshInspector();
    } else if (state.selectedNode === 'study') {
      title = 'スタディ'; type = 'STUDY'; content = studyInspector();
    } else if (state.selectedNode === 'results') {
      title = '結果データ'; type = 'RESULTS'; content = resultsInspector();
    }
    $('#inspectorTitle').textContent = title;
    $('#inspectorType').textContent = type;
    inspector.innerHTML = content;
    updateValidationSummary();
  }

  function geometryInspector() {
    const geometry = state.project.geometry;
    const boxSelected = geometry.kind === 'box';
    const dimensionNote = state.project.mesh.target_cells !== undefined ? '目標総セル数から等方格子を自動計算（メッシュで確認）' : boxSelected ? `等方セル条件: ${fmt(geometry.size[0] / state.project.mesh.cells[0], 6)} m` : 'CAD 境界から流体領域を構築';
    return `
      <div class="property-section">
        <div class="section-title"><span>形状タイプ</span><small>GEOMETRY</small></div>
        <div class="segmented" role="group" aria-label="形状タイプ">
          <button type="button" class="${boxSelected ? 'active' : ''}" data-action="set-geometry-kind" data-kind="box" aria-pressed="${boxSelected}">ボックス</button>
          <button type="button" class="${!boxSelected ? 'active' : ''}" data-action="set-geometry-kind" data-kind="cad" aria-pressed="${!boxSelected}">CAD</button>
        </div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>計算領域</span><small>DOMAIN</small></div>
        <div class="form-field"><label for="geometryRole">モデルの役割</label><select id="geometryRole" data-bind="geometry.role"><option value="fluid" ${geometry.role === 'fluid' ? 'selected' : ''}>流体領域</option><option value="obstacle" ${geometry.role === 'obstacle' ? 'selected' : ''}>障害物 / 固体</option></select></div>
        ${boxSelected ? `
          <div class="section-title"><span>領域サイズ</span><small>SI / m</small></div>
          <div class="field-grid three">
            ${['x', 'y', 'z'].map((axis, index) => `<div class="form-field"><label for="size${axis}">${axis.toUpperCase()}</label><div class="input-with-unit"><input id="size${axis}" class="numeric" type="number" min="0.000001" step="any" data-bind="geometry.size.${index}" value="${esc(geometry.size[index])}"><span class="input-unit">m</span></div></div>`).join('')}
          </div>
          <div class="info-note">${esc(dimensionNote)}<br>メッシュのセル幅が各軸で一致する必要があります。</div>
          ${geometry.role === 'fluid' ? solidRegionsEditor(geometry) : ''}
        ` : `
          ${geometry.asset_id ? cadAssetCard(geometry) : '<div class="warn-note">CAD アセットが未指定です。STL / OBJ / STEP / IGES / BREP を読み込んでください。</div>'}
          <div class="button-row"><button type="button" class="button secondary compact" data-action="import-cad">◇ CAD を読み込む</button></div>
          ${geometry.role === 'obstacle' ? cadSolidMaterialEditor(geometry) : ''}
          ${surfaceSelector()}
          <div class="info-note">CAD は表面メッシュとして読み込み、流体マスクをサーバーで生成します。</div>
        `}
      </div>
      <div class="property-section">
        <div class="section-title"><span>プロジェクト名</span><small>IDENTITY</small></div>
        <div class="form-field"><label for="projectName">表示名</label><input id="projectName" type="text" maxlength="120" data-bind="name" value="${esc(state.project.name)}"></div>
      </div>`;
  }

  function solidRegionsEditor(geometry) {
    const solids = Array.isArray(geometry.solids) ? geometry.solids : [];
    const cards = solids.map((solid, index) => {
      const options = state.project.materials.map((material) => `<option value="${esc(material.id)}" ${material.id === solid.material_id ? 'selected' : ''}>${esc(material.name)}</option>`).join('');
      return `<div class="solid-card"><div class="solid-card-head"><strong>固体領域 ${index + 1}</strong><button type="button" class="icon-delete" data-action="remove-solid" data-solid-index="${index}" aria-label="固体領域を削除">×</button></div><div class="form-field"><label for="solidName${index}">表示名</label><input id="solidName${index}" type="text" maxlength="80" data-solid-field="name" data-solid-index="${index}" value="${esc(solid.name)}"></div><div class="form-field"><label for="solidMaterial${index}">材料</label><select id="solidMaterial${index}" data-solid-field="material_id" data-solid-index="${index}">${options}</select></div><div class="form-field"><label>原点 [m]</label><div class="vector-grid">${['x', 'y', 'z'].map((axis, component) => `<div class="form-field"><span class="vector-label">${axis}</span><input class="numeric" type="number" step="any" data-solid-field="origin" data-solid-vector="${component}" data-solid-index="${index}" value="${esc(solid.origin[component])}" aria-label="${solid.name} 原点 ${axis}"></div>`).join('')}</div></div><div class="form-field"><label>サイズ [m]</label><div class="vector-grid">${['x', 'y', 'z'].map((axis, component) => `<div class="form-field"><span class="vector-label">${axis}</span><input class="numeric" type="number" min="0.000001" step="any" data-solid-field="size" data-solid-vector="${component}" data-solid-index="${index}" value="${esc(solid.size[component])}" aria-label="${solid.name} サイズ ${axis}"></div>`).join('')}</div></div></div>`;
    }).join('');
    return `<div class="property-section solids-section"><div class="section-title"><span>固体領域 / CHT</span><small>${solids.length} SOLID${solids.length === 1 ? '' : 'S'}</small></div>${cards || '<div class="info-note">流体領域内に固体を追加すると、固体の温度場を連成します。</div>'}<div class="button-row" style="margin-top:8px"><button type="button" class="button secondary compact" data-action="add-solid">＋ 固体領域を追加</button></div></div>`;
  }

  function cadSolidMaterialEditor(geometry) {
    const options = state.project.materials.map((material) => `<option value="${esc(material.id)}" ${material.id === geometry.solid_material_id ? 'selected' : ''}>${esc(material.name)}</option>`).join('');
    return `<div class="property-section"><div class="section-title"><span>固体材料 / CHT</span><small>CAD OBSTACLE</small></div><div class="form-field"><label for="solidMaterialSelect">CAD の固体材料</label><select id="solidMaterialSelect" data-bind="geometry.solid_material_id"><option value="" ${!geometry.solid_material_id ? 'selected' : ''}>指定なし（流体のみ）</option>${options}</select></div><small class="muted">材料を指定すると CAD 内部の温度場を流体と連成します。</small></div>`;
  }

  function surfaceSelector() {
    const groups = surfaceGroups();
    if (!groups.length) return '<div class="property-section"><div class="section-title"><span>CAD 表面</span><small>SURFACE PATCH</small></div><div class="info-note">表面グループ情報が読み込まれると、面ごとの境界条件を選択できます。</div></div>';
    const options = groups.map((group) => { const id = surfaceGroupId(group); return `<option value="${esc(id)}" ${state.selectedSurface === id ? 'selected' : ''}>${esc(group.name || id)} · ${esc(group.triangle_count ?? group.triangleCount ?? '—')} triangles</option>`; }).join('');
    return `<div class="property-section"><div class="section-title"><span>CAD 表面を選択</span><small>${groups.length} PATCHES</small></div><div class="form-field"><label for="surfaceSelect">ハイライトする面</label><select id="surfaceSelect" data-action="select-surface"><option value="" ${!state.selectedSurface ? 'selected' : ''}>全表面</option>${options}</select></div><div class="info-note">選択した面は 3D ビューで黄色く表示されます。境界条件は面ごとに定義できます。</div></div>`;
  }

  function cadAssetCard(geometry) {
    const asset = geometry.asset_id;
    const assetMeta = state.cadAssetMeta && (state.cadAssetMeta.id === asset || state.cadAssetMeta.asset_id === asset) ? state.cadAssetMeta : (typeof asset === 'object' ? asset : null);
    const label = assetMeta?.original_name || assetMeta?.name || (typeof asset === 'string' ? asset : geometry.asset_id);
    const bounds = extractBounds(assetMeta?.bounds || assetMeta?.bounding_box || assetMeta?.bounds_m);
    const boundsText = bounds ? `min [${bounds.min.map((value) => fmt(value, 6)).join(', ')}] m · max [${bounds.max.map((value) => fmt(value, 6)).join(', ')}] m` : 'サーバーに保存済み';
    return `<div class="asset-card"><div class="asset-card-head"><strong>${esc(label)}</strong><span class="asset-kind">CAD</span></div><span>${esc(boundsText)}</span><span>asset_id: ${esc(assetMeta?.id || assetMeta?.asset_id || asset)}</span></div>`;
  }

  const MATERIAL_KEYS = ['density', 'viscosity', 'heat_capacity', 'conductivity'];
  const MATERIAL_CSV_MAX_BYTES = 1024 * 1024;

  function normalizeMaterialPresetProperty(value, key) {
    if (!value || typeof value !== 'object') return null;
    if (value.kind === 'table') {
      if (!Array.isArray(value.points) || value.points.length < 2) return null;
      const points = value.points.map((point) => [Number(point?.[0]), Number(point?.[1])]);
      if (points.some((point) => !Number.isFinite(point[0]) || point[0] <= 0 || !Number.isFinite(point[1]) || point[1] <= 0)) return null;
      if (points.some((point, index) => index > 0 && point[0] <= points[index - 1][0])) return null;
      return { kind: 'table', points };
    }
    if (value.kind !== 'constant') return null;
    const scalar = Number(value.value);
    return Number.isFinite(scalar) && scalar > 0 ? { kind: 'constant', value: scalar } : null;
  }

  function normalizeMaterialPreset(value) {
    if (!value || typeof value !== 'object' || !value.id) return null;
    const materialSource = value.material && typeof value.material === 'object' ? value.material : {};
    const range = Array.isArray(value.temperature_range_k) && value.temperature_range_k.length >= 2
      ? [Number(value.temperature_range_k[0]), Number(value.temperature_range_k[1])]
      : null;
    const normalizedRange = range && range.every((item) => Number.isFinite(item) && item > 0) && range[1] >= range[0] ? range : null;
    if (value.temperature_range_k !== null && value.temperature_range_k !== undefined && !normalizedRange) return null;
    const pressure = Number(value.pressure_pa);
    const sources = Array.isArray(value.sources) ? value.sources.map((source) => {
      if (!source || typeof source !== 'object') return null;
      const url = materialSourceUrl(source.url);
      return url ? { title: String(source.title || url), url } : null;
    }).filter(Boolean) : [];
    const properties = Object.fromEntries(MATERIAL_KEYS.map((key) => [key, normalizeMaterialPresetProperty(materialSource[key], key)]));
    if (Object.values(properties).some((property) => !property)) return null;
    return {
      id: String(value.id),
      label: String(value.label || value.id),
      description: String(value.description || ''),
      temperature_range_k: normalizedRange,
      pressure_pa: Number.isFinite(pressure) ? pressure : null,
      sources,
      material: {
        name: String(materialSource.name || value.label || value.id),
        ...properties
      }
    };
  }

  function materialSourceUrl(value) {
    try {
      const url = new URL(String(value || ''), window.location.origin);
      return ['http:', 'https:'].includes(url.protocol) ? url.href : null;
    } catch (_) {
      return null;
    }
  }

  function ensureMaterialPresets() {
    if (state.materialPresetsAttempted || state.materialPresetsLoading) return;
    state.materialPresetsLoading = true;
    const requestId = state.materialPresetsRequest = (state.materialPresetsRequest || 0) + 1;
    request('/api/materials/presets').then((payload) => {
      if (state.materialPresetsRequest !== requestId) return;
      const rawPresets = Array.isArray(payload?.presets) ? payload.presets : [];
      const normalized = rawPresets.map(normalizeMaterialPreset);
      state.materialPresets = normalized.every(Boolean) ? normalized : [];
      state.materialPresetsError = rawPresets.length && normalized.every(Boolean) ? '' : '材料プリセットが見つかりませんでした。';
      if (rawPresets.length && !normalized.every(Boolean)) state.materialPresetsError = 'プリセットの形式を読み込めませんでした。';
      state.materialPresetsAttempted = true;
      state.materialPresetsLoading = false;
      if (state.selectedNode === 'material') renderInspector();
    }).catch((error) => {
      if (state.materialPresetsRequest !== requestId) return;
      state.materialPresets = [];
      state.materialPresetsError = `プリセットを取得できません: ${error.message}`;
      state.materialPresetsAttempted = true;
      state.materialPresetsLoading = false;
      pushLog(state.materialPresetsError, 'warn');
      if (state.selectedNode === 'material') renderInspector();
    });
  }

  function selectedMaterialPreset() {
    return state.materialPresets.find((preset) => preset.id === state.materialPresetSelectedId) || null;
  }

  function materialPresetDetailsMarkup(preset, material) {
    if (!preset) return '<div class="info-note">プリセットを選択すると説明、適用範囲、出典を表示します。</div>';
    const range = Array.isArray(preset.temperature_range_k)
      ? `${fmt(preset.temperature_range_k[0], 2)}–${fmt(preset.temperature_range_k[1], 2)} K`
      : '指定なし';
    const pressure = preset.pressure_pa === null ? '指定なし' : `${fmtSci(preset.pressure_pa)} Pa`;
    const sources = preset.sources.length
      ? `<div class="preset-sources"><span>出典</span>${preset.sources.map((source) => `<a href="${esc(source.url)}" target="_blank" rel="noopener noreferrer">${esc(source.title)}</a>`).join('')}</div>`
      : '<div class="muted">出典情報はありません。</div>';
    const appliedHere = state.materialPresetAppliedMaterialRef === material && state.materialPresetAppliedId === preset.id;
    const edited = appliedHere && state.materialPresetEdited;
    const applied = appliedHere && !edited;
    const attribution = edited
      ? '<div class="warn-note">現在の材料はプリセット適用後に編集されています。表示中の出典はプリセット値の参照情報です。</div>'
      : applied
        ? '<div class="info-note">現在の材料へプリセット値を適用済みです。表示中の出典はプリセット値の参照情報です。</div>'
        : '';
    return `<div class="preset-details"><strong>${esc(preset.label)}</strong>${preset.description ? `<p>${esc(preset.description)}</p>` : ''}<span>温度範囲: ${esc(range)}</span><span>基準圧力: ${esc(pressure)}</span>${sources}</div>${attribution}`;
  }

  function materialPresetIdBase(value) {
    const base = String(value || 'material').toLowerCase().replace(/[^a-z0-9_-]+/g, '-').replace(/^-+|-+$/g, '');
    return base || 'material';
  }

  function uniqueMaterialId(value) {
    const used = new Set((state.project?.materials || []).map((material) => String(material.id)));
    const base = materialPresetIdBase(value);
    let candidate = base;
    let suffix = 2;
    while (used.has(candidate)) candidate = `${base}-${suffix++}`;
    return candidate;
  }

  function materialPropertiesSignature(material) {
    return JSON.stringify(MATERIAL_KEYS.map((key) => material?.[key] || null));
  }

  function markMaterialManuallyEdited(material) {
    if (state.materialPresetAppliedMaterialRef === material) state.materialPresetEdited = true;
  }

  function selectMaterialPreset(value) {
    state.materialPresetSelectedId = String(value || '');
    renderInspector();
  }

  function applyMaterialPreset(addNew = false) {
    const preset = selectedMaterialPreset();
    if (!preset || !state.project) return;
    if (addNew) {
      const id = uniqueMaterialId(preset.id || preset.material.name);
      const material = normalizeMaterial({ id, ...clone(preset.material) }, state.project.materials.length);
      state.project.materials.push(material);
      state.project.physics.material_id = material.id;
      state.materialPresetAppliedMaterialRef = material;
      state.materialPresetAppliedId = preset.id;
      state.materialPresetAppliedSignature = materialPropertiesSignature(material);
      state.materialPresetEdited = false;
      state.materialTablePages = {};
      markDirty();
      renderTree();
      renderInspector();
      scheduleMeshEstimate();
      toast(`材料「${material.name}」を追加しました。`);
      return;
    }
    const material = currentMaterial();
    if (!material) return;
    material.name = preset.material.name;
    MATERIAL_KEYS.forEach((key) => { material[key] = clone(preset.material[key]); });
    state.materialPresetAppliedMaterialRef = material;
    state.materialPresetAppliedId = preset.id;
    state.materialPresetAppliedSignature = materialPropertiesSignature(material);
    state.materialPresetEdited = false;
    state.materialTablePages = {};
    markDirty();
    renderTree();
    renderInspector();
    scheduleMeshEstimate();
    toast(`材料「${material.name}」へプリセットを適用しました。`);
  }

  function invalidateMaterialCsvPreview(close = true) {
    state.materialCsvRequest += 1;
    state.materialCsvInputMode = null;
    state.materialCsvLoading = false;
    state.materialCsvPreview = null;
    if (close) closeDialog('materialCsvDialog');
  }

  function materialCsvContextCurrent(context) {
    const material = currentMaterial();
    return Boolean(context && state.project === context.projectRef && state.projectId === context.projectId && material === context.materialRef && material.id === context.materialId);
  }

  function openMaterialCsvDialog() {
    const dialog = $('#materialCsvDialog');
    if (!dialog) return;
    renderMaterialCsvPreview();
    if (typeof dialog.showModal === 'function' && !dialog.open) dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  function startMaterialCsvImport(propertyKey = null) {
    const material = currentMaterial();
    if (!material || (propertyKey && !MATERIAL_KEYS.includes(propertyKey))) return;
    const input = $('#materialCsvFile');
    if (!input) return;
    invalidateMaterialCsvPreview();
    const requestId = state.materialCsvRequest = (state.materialCsvRequest || 0) + 1;
    state.materialCsvInputMode = {
      requestId,
      projectRef: state.project,
      projectId: state.projectId,
      materialRef: material,
      materialId: String(material.id),
      propertyKey: propertyKey || null,
      filename: ''
    };
    input.value = '';
    input.click();
  }

  async function readUtf8CsvFile(file) {
    const buffer = await file.arrayBuffer();
    const decoder = new TextDecoder('utf-8', { fatal: true });
    return decoder.decode(buffer).replace(/^\uFEFF/, '');
  }

  function showMaterialCsvError(context, filename, message) {
    if (!materialCsvContextCurrent(context) || state.materialCsvRequest !== context.requestId) return;
    state.materialCsvLoading = false;
    state.materialCsvPreview = { status: 'error', filename, context, error: String(message) };
    openMaterialCsvDialog();
  }

  async function handleMaterialCsvFile(file) {
    const context = state.materialCsvInputMode;
    if (!context || !file) return;
    context.filename = String(file.name || 'material.csv');
    if (file.size > MATERIAL_CSV_MAX_BYTES) {
      showMaterialCsvError(context, context.filename, 'CSV ファイルは 1 MiB 以下にしてください。');
      return;
    }
    state.materialCsvLoading = true;
    state.materialCsvPreview = { status: 'loading', filename: context.filename, context };
    openMaterialCsvDialog();
    try {
      const text = await readUtf8CsvFile(file);
      if (!materialCsvContextCurrent(context) || state.materialCsvRequest !== context.requestId) return;
      const body = { text };
      if (context.propertyKey) body.property_key = context.propertyKey;
      const payload = await request('/api/materials/csv', { method: 'POST', body });
      if (!materialCsvContextCurrent(context) || state.materialCsvRequest !== context.requestId) return;
      const properties = payload?.properties && typeof payload.properties === 'object' ? payload.properties : {};
      const included = Object.keys(properties).filter((key) => MATERIAL_KEYS.includes(key));
      if (!included.length) throw new Error('CSV から物性値を読み込めませんでした。');
      state.materialCsvLoading = false;
      state.materialCsvPreview = { status: 'ready', filename: context.filename, context, properties, rows: Number(payload.rows) || 0, temperature_unit: payload.temperature_unit === 'C' ? 'C' : 'K', warnings: Array.isArray(payload.warnings) ? payload.warnings : [] };
      openMaterialCsvDialog();
    } catch (error) {
      showMaterialCsvError(context, context.filename, error.message);
    }
  }

  function csvPreviewRows(properties) {
    const keys = Object.keys(properties).filter((key) => MATERIAL_KEYS.includes(key));
    const first = keys.length ? (Array.isArray(properties[keys[0]]?.points) ? properties[keys[0]].points : []) : [];
    const rowCount = Math.min(5, Math.max(first.length, ...keys.map((key) => Array.isArray(properties[key]?.points) ? properties[key].points.length : 0)));
    return Array.from({ length: rowCount }, (_, index) => {
      const row = keys.map((key) => properties[key]?.points?.[index] || []);
      const temperature = row.find((point) => Number.isFinite(Number(point?.[0])))?.[0];
      return { temperature, values: row.map((point) => point?.[1]) };
    });
  }

  function materialCsvTemperatureRange(properties) {
    const temperatures = Object.values(properties || {}).flatMap((property) => Array.isArray(property?.points) ? property.points.map((point) => Number(point?.[0])).filter((value) => Number.isFinite(value)) : []);
    if (!temperatures.length) return '—';
    return `${fmt(Math.min(...temperatures), 2)}–${fmt(Math.max(...temperatures), 2)} K`;
  }

  function validMaterialTableProperty(value) {
    if (!value || typeof value !== 'object' || value.kind !== 'table' || !Array.isArray(value.points) || value.points.length < 2) return false;
    return value.points.every((point, index) => {
      const temperature = Number(point?.[0]);
      const propertyValue = Number(point?.[1]);
      const previous = index ? Number(value.points[index - 1]?.[0]) : null;
      return Number.isFinite(temperature) && temperature > 0 && Number.isFinite(propertyValue) && propertyValue > 0 && (previous === null || temperature > previous);
    });
  }

  function renderMaterialCsvPreview() {
    const output = $('#materialCsvPreview');
    const apply = $('[data-action="apply-material-csv"]');
    if (!output) return;
    const preview = state.materialCsvPreview;
    if (!preview) {
      output.innerHTML = '<div class="info-note">CSV ファイルを選択すると、適用前に内容を確認できます。</div>';
      if (apply) apply.disabled = true;
      return;
    }
    const current = materialCsvContextCurrent(preview.context);
    if (!current) {
      output.innerHTML = '<div class="warn-note">対象のプロジェクトまたは材料が変更されました。このプレビューは適用できません。ダイアログを閉じて再度選択してください。</div>';
      if (apply) apply.disabled = true;
      return;
    }
    if (preview.status === 'loading') {
      output.innerHTML = `<div class="asset-card"><div class="asset-card-head"><strong>${esc(preview.filename)}</strong><span class="asset-kind">CSV</span></div><span>UTF-8 CSV を解析しています…</span></div>`;
      if (apply) apply.disabled = true;
      return;
    }
    if (preview.status === 'error') {
      output.innerHTML = `<div class="error-note" role="alert"><strong>${esc(preview.filename)}</strong><br>${esc(preview.error)}</div>`;
      if (apply) apply.disabled = true;
      return;
    }
    const keys = Object.keys(preview.properties || {}).filter((key) => MATERIAL_KEYS.includes(key));
    const rows = csvPreviewRows(preview.properties);
    const header = ['温度 [K]', ...keys.map((key) => `${MATERIAL_LABELS[key]} [${MATERIAL_UNITS[key]}]`)].map((value) => `<span>${esc(value)}</span>`).join('');
    const body = rows.map((row) => `<div class="table-row"><span>${esc(fmt(row.temperature, 3))}</span>${row.values.map((value) => `<span>${esc(fmtSci(value))}</span>`).join('')}</div>`).join('');
    const warnings = preview.warnings?.length ? `<div class="warn-note">${preview.warnings.map((warning) => esc(warning)).join('<br>')}</div>` : '';
    output.innerHTML = `<div class="asset-card"><div class="asset-card-head"><strong>${esc(preview.filename)}</strong><span class="asset-kind">${fmt(preview.rows, 0)} ROWS</span></div><span>含まれる物性: ${esc(keys.map((key) => MATERIAL_LABELS[key]).join('、'))}</span><span>温度範囲: ${esc(materialCsvTemperatureRange(preview.properties))}（入力温度: ${preview.temperature_unit === 'C' ? '°C' : 'K'} / 内部値: K）</span></div><div class="table-editor csv-preview-table"><div class="table-head">${header}</div>${body || '<div class="info-note">表示できる行がありません。</div>'}</div><div class="info-note">適用時は含まれる物性だけを現在の材料へ反映します。温度は K または °C、粘度は Pa·s（SI）で入力してください。</div>${warnings}`;
    if (apply) apply.disabled = !keys.length;
  }

  function applyMaterialCsv() {
    const preview = state.materialCsvPreview;
    if (!preview || preview.status !== 'ready') return;
    if (!materialCsvContextCurrent(preview.context)) {
      renderMaterialCsvPreview();
      return;
    }
    const material = currentMaterial();
    const keys = Object.keys(preview.properties || {}).filter((key) => MATERIAL_KEYS.includes(key));
    if (!material || !keys.length) return;
    if (keys.some((key) => !validMaterialTableProperty(preview.properties[key]))) {
      state.materialCsvPreview = { ...preview, status: 'error', error: 'CSV 応答の物性テーブルが不正です。元の材料は変更していません。' };
      renderMaterialCsvPreview();
      return;
    }
    keys.forEach((key) => { material[key] = clone(preview.properties[key]); });
    state.materialPresetAppliedMaterialRef = null;
    state.materialPresetAppliedId = '';
    state.materialPresetAppliedSignature = '';
    state.materialPresetEdited = false;
    state.materialTablePages = {};
    markDirty();
    invalidateMaterialCsvPreview();
    renderTree();
    renderInspector();
    scheduleMeshEstimate();
    toast(`CSV の ${keys.length} 種類の物性を適用しました。`);
  }

  function materialInspector() {
    const material = currentMaterial();
    if (!material) return '<div class="error-note">材料がありません。プロジェクト JSON を確認してください。</div>';
    ensureMaterialPresets();
    const materialOptions = state.project.materials.map((item) => `<option value="${esc(item.id)}" ${item.id === material.id ? 'selected' : ''}>${esc(item.name)}</option>`).join('');
    const selectedPreset = selectedMaterialPreset();
    const presetOptions = state.materialPresets.map((preset) => `<option value="${esc(preset.id)}" ${preset.id === state.materialPresetSelectedId ? 'selected' : ''}>${esc(preset.label)}</option>`).join('');
    const presetStatus = state.materialPresetsLoading
      ? '<div class="info-note">材料プリセットを読み込んでいます…</div>'
      : state.materialPresetsError
        ? `<div class="warn-note">${esc(state.materialPresetsError)}</div>`
        : materialPresetDetailsMarkup(selectedPreset, material);
    const presetDisabled = state.materialPresetsLoading || Boolean(state.materialPresetsError) || !state.materialPresets.length;
    return `
      <div class="property-section">
        <div class="section-title"><span>材料セット</span><small>${state.project.materials.length} MATERIAL${state.project.materials.length === 1 ? '' : 'S'}</small></div>
        <div class="form-field"><label for="materialSelect">使用する材料</label><select id="materialSelect" data-action="select-material">${materialOptions}</select></div>
        <div class="form-field"><label for="materialName">材料名</label><input id="materialName" type="text" maxlength="80" data-material-name value="${esc(material.name)}"></div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>材料プリセット</span><small>CATALOG</small></div>
        <div class="form-field"><label for="materialPreset">プリセット</label><select id="materialPreset" data-material-preset ${presetDisabled ? 'disabled' : ''}><option value="" ${selectedPreset ? '' : 'selected'}>プリセットを選択…</option>${presetOptions}</select></div>
        ${presetStatus}
        <div class="button-row"><button type="button" class="button secondary compact" data-action="apply-material-preset" ${selectedPreset ? '' : 'disabled'}>選択中の材料へ適用</button><button type="button" class="button ghost compact" data-action="add-material-preset" ${selectedPreset ? '' : 'disabled'}>新しい材料として追加</button></div>
        <div class="muted" style="font-size:10px;margin-top:7px">空気は乾燥空気です。温度依存密度は熱計算と有効化した浮力に使用します。流れは基準密度を使う非圧縮近似です（圧力依存・湿度・相変化なし）。</div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>物性値</span><small>CONSTANT / TABLE</small></div>
        ${materialPropertyCard(material, 'density', '密度', 'ρ', 'kg/m³', '温度依存なし')}
        ${materialPropertyCard(material, 'viscosity', '粘度（粘性係数）', 'μ', 'Pa·s', '温度依存なし')}
        ${materialPropertyCard(material, 'heat_capacity', '比熱容量', 'Cp', 'J/(kg·K)', '温度依存なし')}
        ${materialPropertyCard(material, 'conductivity', '熱伝導率', 'k', 'W/(m·K)', '温度依存なし')}
      </div>
      <div class="property-section">
        <div class="section-title"><span>CSV 読み込み</span><small>UTF-8 / SI</small></div>
        <div class="button-row"><button type="button" class="button secondary compact" data-action="import-material-csv">材料 CSV をプレビュー</button></div>
        <div class="info-note">wide CSV は temperature_K または temperature_C と、密度・粘度・比熱容量・熱伝導率の任意の列を指定できます。2 列形式は per-property ボタンから読み込みます。温度は K / °C、粘度は Pa·s です。</div>
        <div class="button-row"><a class="text-button" href="/static/examples/material-properties.csv" target="_blank" rel="noopener">wide CSV 例</a><a class="text-button" href="/static/examples/material-property.csv" target="_blank" rel="noopener">2 列 CSV 例</a></div>
      </div>
      <div class="info-note">テーブル補間は線形、端点の外側はクランプされます。使用単位は SI です。</div>`;
  }

  const MATERIAL_LABELS = { density: '密度', viscosity: '粘度（粘性係数）', heat_capacity: '比熱容量', conductivity: '熱伝導率' };
  const MATERIAL_SYMBOLS = { density: 'ρ', viscosity: 'μ', heat_capacity: 'Cp', conductivity: 'k' };
  const MATERIAL_UNITS = { density: 'kg/m³', viscosity: 'Pa·s', heat_capacity: 'J/(kg·K)', conductivity: 'W/(m·K)' };

  function materialPropertyCard(material, key, label, symbol, unit, fallbackText) {
    const property = material[key];
    const table = property.kind === 'table';
    return `<div class="property-card" data-property-card="${key}">
      <div class="property-card-head"><div class="property-name"><span class="property-symbol">${symbol}</span><span>${label}</span></div><select data-material-mode="${key}" aria-label="${label} の定義"><option value="constant" ${!table ? 'selected' : ''}>定数</option><option value="table" ${table ? 'selected' : ''}>テーブル</option></select></div>
      ${table ? materialTable(property, key, unit) : `<div class="input-with-unit"><input class="numeric" type="number" step="any" data-material-value="${key}" value="${esc(property.value)}" aria-label="${label}"><span class="input-unit">${unit}</span></div><small class="muted">${fallbackText}</small>`}
      <div class="button-row" style="margin-top:7px"><button type="button" class="button ghost compact" data-action="import-material-property-csv" data-property="${key}">CSV（2列）</button></div>
    </div>`;
  }

  function materialTablePageKey(key) {
    return `${String(currentMaterial()?.id || 'material')}:${key}`;
  }

  function changeMaterialTablePage(key, direction) {
    const material = currentMaterial();
    const points = material?.[key]?.kind === 'table' && Array.isArray(material[key].points) ? material[key].points : [];
    const pageCount = Math.max(1, Math.ceil(points.length / 100));
    const pageKey = materialTablePageKey(key);
    const current = Math.max(0, Math.min(pageCount - 1, Math.floor(numberOr(state.materialTablePages[pageKey], 0))));
    state.materialTablePages[pageKey] = Math.max(0, Math.min(pageCount - 1, current + Math.sign(Number(direction) || 0)));
    renderInspector();
  }

  function materialTable(property, key, unit) {
    const points = Array.isArray(property.points) && property.points.length ? property.points : [[273.15, 0], [373.15, 0]];
    const pageKey = materialTablePageKey(key);
    const pageCount = Math.max(1, Math.ceil(points.length / 100));
    const page = Math.max(0, Math.min(pageCount - 1, Math.floor(numberOr(state.materialTablePages[pageKey], 0))));
    state.materialTablePages[pageKey] = page;
    const start = page * 100;
    const visible = points.slice(start, start + 100);
    const rows = visible.map((point, offset) => {
      const index = start + offset;
      return `<div class="table-row"><input class="numeric" type="number" step="any" data-table-property="${key}" data-table-index="${index}" data-table-column="0" value="${esc(point[0])}" aria-label="${MATERIAL_LABELS[key]} 温度 ${index + 1}"><input class="numeric" type="number" step="any" data-table-property="${key}" data-table-index="${index}" data-table-column="1" value="${esc(point[1])}" aria-label="${MATERIAL_LABELS[key]} 値 ${index + 1}"><button type="button" class="icon-delete" data-action="remove-material-row" data-property="${key}" data-index="${index}" aria-label="行を削除">×</button></div>`;
    }).join('');
    const pager = pageCount > 1 ? `<div class="button-row" style="padding:7px 8px;border-top:1px solid var(--line-soft);justify-content:space-between"><button type="button" class="button ghost compact" data-action="material-table-page" data-property="${key}" data-page-direction="-1" ${page <= 0 ? 'disabled' : ''}>‹ 前へ</button><span class="muted" style="font-size:10px">${page + 1} / ${pageCount}（全 ${points.length} 点）</span><button type="button" class="button ghost compact" data-action="material-table-page" data-property="${key}" data-page-direction="1" ${page >= pageCount - 1 ? 'disabled' : ''}>次へ ›</button></div>` : '';
    return `<div class="table-editor"><div class="table-head"><span>T [K]</span><span>value [${unit}]</span><span></span></div>${rows}<button type="button" class="table-add" data-action="add-material-row" data-property="${key}">＋ 温度点を追加</button>${pager}</div>`;
  }

  function physicsInspector() {
    const physics = state.project.physics;
    const turbulence = normalizeTurbulence(physics.turbulence);
    const gravity = normalizeGravity(physics.gravity);
    const gravityEnabled = gravity.enabled === true;
    const gravityBuoyancy = gravity.mode === 'buoyancy';
    const gravityInactive = gravityEnabled && !physics.flow;
    const materials = state.project.materials.map((material) => `<option value="${esc(material.id)}" ${material.id === physics.material_id ? 'selected' : ''}>${esc(material.name)}</option>`).join('');
    return `
      <div class="property-section">
        <div class="section-title"><span>物理インターフェース</span><small>PHYSICS</small></div>
        <div class="check-row"><div class="check-copy"><strong>非圧縮流れ${turbulence.model === 'smagorinsky' ? '' : '（層流）'}</strong><span>速度・圧力を解く</span></div><label class="switch"><input type="checkbox" data-bind="physics.flow" ${physics.flow ? 'checked' : ''} aria-label="非圧縮流れ"><span class="switch-track"></span></label></div>
        <div class="check-row"><div class="check-copy"><strong>熱輸送</strong><span>温度場を解く</span></div><label class="switch"><input type="checkbox" data-bind="physics.thermal" ${physics.thermal ? 'checked' : ''} aria-label="熱輸送"><span class="switch-track"></span></label></div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>流体の計算法</span><small>FLOW METHOD</small></div>
        <div class="form-field"><label for="flowMethod">計算法</label><input id="flowMethod" type="text" value="格子ボルツマン法（LBM）" readonly></div>
        <div class="info-note">${physics.flow ? 'XLB の D3Q27 格子・BGK 衝突モデルで速度と圧力を計算します。低マッハ数の非圧縮近似に対応し、層流・LES のどちらにも LBM を使用します。' : '流れは無効です。有効にすると格子ボルツマン法（LBM）で計算します。'}</div>
        ${physics.thermal ? '<div class="info-note">温度場は有限体積法で計算します。流れが有効な場合は LBM の速度場と連成します。</div>' : ''}
      </div>
      <div class="property-section">
        <div class="section-title"><span>流体モデル</span><small>FLOW CLOSURE</small></div>
        <div class="form-field"><label for="turbulenceModel">モデル</label><select id="turbulenceModel" data-turbulence-field="model"><option value="laminar" ${turbulence.model === 'laminar' ? 'selected' : ''}>層流</option><option value="smagorinsky" ${turbulence.model === 'smagorinsky' ? 'selected' : ''}>Smagorinsky LES</option></select></div>
        ${turbulence.model === 'smagorinsky' ? `<div class="field-grid"><div class="form-field"><label for="smagorinskyConstant">Smagorinsky 定数 Cs</label><input id="smagorinskyConstant" class="numeric" type="number" min="0" step="0.01" data-turbulence-field="smagorinsky_constant" value="${esc(turbulence.smagorinsky_constant)}"></div><div class="form-field"><label for="turbulentPrandtl">乱流 Prandtl 数</label><input id="turbulentPrandtl" class="numeric" type="number" min="0.01" step="0.01" data-turbulence-field="turbulent_prandtl" value="${esc(turbulence.turbulent_prandtl)}"></div></div><div class="info-note">Smagorinsky の渦粘性を速度場へ適用します。結果ビューから渦粘性係数も取得できます。</div>` : '<div class="info-note">層流モデルを使用しています。乱流の寄与はありません。</div>'}
      </div>
      <div class="property-section">
        <div class="section-title"><span>重力・浮力</span><small>BODY FORCE</small></div>
        <div class="check-row"><div class="check-copy"><strong>重力を有効化</strong><span>体積力または温度差による浮力を計算</span></div><label class="switch"><input type="checkbox" data-bind="physics.gravity.enabled" ${gravityEnabled ? 'checked' : ''} aria-label="重力を有効化"><span class="switch-track"></span></label></div>
        ${gravityEnabled ? `
          <div class="form-field"><label for="gravityMode">モデル</label><select id="gravityMode" data-bind="physics.gravity.mode"><option value="uniform" ${!gravityBuoyancy ? 'selected' : ''}>重力（一定加速度）</option><option value="buoyancy" ${gravityBuoyancy ? 'selected' : ''}>浮力（温度依存密度）</option></select></div>
          <div class="form-field"><label>重力加速度ベクトル</label><div class="vector-grid">${['x', 'y', 'z'].map((axis, index) => `<div class="form-field"><span class="vector-label">${axis}</span><div class="input-with-unit"><input class="numeric" type="number" step="any" data-bind="physics.gravity.vector.${index}" value="${esc(gravity.vector[index])}" aria-label="重力加速度 ${axis}"><span class="input-unit">m/s²</span></div></div>`).join('')}</div></div>
          ${gravityBuoyancy ? '<div class="form-field"><label for="gravityReferenceTemperature">基準温度</label><div class="input-with-unit"><input id="gravityReferenceTemperature" class="numeric" type="number" min="0.000001" step="any" data-bind="physics.gravity.reference_temperature" value="' + esc(gravity.reference_temperature) + '"><span class="input-unit">K</span></div><small>浮力は ρ(T) / ρ(Tref) − 1 から計算します。</small></div>' : ''}
          ${gravityBuoyancy ? '<div class="info-note">浮力モードでは基準静水圧を除いた圧力を表示します。定密度材料では浮力は 0 です。暖かく軽い流体は重力と逆向きに加速します。</div>' : '<div class="info-note">一定加速度として重力の全量を適用し、圧力は通常のゲージ圧です。</div>'}
          <div class="info-note">低速近似でも必ず重力駆動になるわけではありません。閉じた一定密度流体では静水圧と釣り合います。</div>
          ${gravityInactive ? '<div class="warn-note">流れが無効のため、設定は保存されますが重力・浮力は計算に適用されません。</div>' : ''}
        ` : '<div class="info-note">重力・浮力は無効です。</div>'}
      </div>
      <div class="property-section">
        <div class="section-title"><span>初期条件</span><small>INITIAL</small></div>
        <div class="form-field"><label for="physicsMaterial">材料</label><select id="physicsMaterial" data-bind="physics.material_id">${materials}</select></div>
        <div class="form-field"><label for="initialTemperature">初期温度</label><div class="input-with-unit"><input id="initialTemperature" class="numeric" type="number" min="0" step="any" data-bind="physics.initial_temperature" value="${esc(physics.initial_temperature)}"><span class="input-unit">K</span></div></div>
      </div>
      ${state.project.geometry.solids?.length || state.project.geometry.solid_material_id ? '<div class="info-note">CHT: 流体と固体の温度場を連成します。固体領域の物性は材料設定から参照されます。</div>' : `<div class="info-note">粘度・比熱・熱伝導率は温度で更新します。${gravityEnabled && gravityBuoyancy ? '流れの基準密度は浮力の基準温度から求め、局所的な密度の温度変化を浮力に反映します。' : '流れの密度は初期温度の基準値を使います。'}</div>`}`;
  }

  function boundaryInspector(face) {
    const existing = boundaryForFace(face);
    const defaultBoundary = !existing;
    const boundary = existing || normalizeBoundary({ id: uid('preview'), name: faceLabel(face) || 'Boundary', face, flow: { type: 'wall' }, thermal: { type: 'adiabatic' } });
    const flow = boundary.flow || { type: 'wall' };
    const thermal = boundary.thermal || { type: 'adiabatic' };
    const flowType = ['velocity', 'pressure', 'wall'].includes(flow.type) ? flow.type : 'wall';
    const thermalType = ['temperature', 'heat_flux', 'convection', 'adiabatic'].includes(thermal.type) ? thermal.type : 'adiabatic';
    const cadFace = isCadFace(boundary.face);
    // The unspecific `cad` selector is the fallback wall for all remaining
    // surfaces.  A named `cad:<patch_id>` selector is an individual surface
    // and may use the same velocity/pressure controls as a box face.
    const cadFallback = boundary.face === 'cad';
    const faceChoices = [...FACES.map((value) => ({ value, label: `${FACE_INFO[value].label} (${FACE_INFO[value].short})` })), ...surfaceGroups().map((group) => { const id = surfaceGroupId(group); return { value: `cad:${id}`, label: `${group.name || id} (CAD)` }; })];
    if (face && !faceChoices.some((item) => item.value === face)) faceChoices.push({ value: face, label: `${faceLabel(face)} (CAD)` });
    return `
      ${defaultBoundary ? `<div class="info-note">${esc(faceLabel(face))} は既定条件です。カスタム条件を有効にするとプロジェクトに保存されます。</div><div class="button-row" style="margin-top:9px"><button type="button" class="button secondary compact" data-action="enable-boundary" data-face="${esc(face)}">＋ カスタム境界を有効化</button></div>` : `
        <div class="property-section">
          <div class="section-title"><span>境界の識別</span><small>${esc(boundary.id.slice(0, 12))}</small></div>
          <div class="form-field"><label for="boundaryName">表示名</label><input id="boundaryName" type="text" maxlength="80" data-boundary-field="name" value="${esc(boundary.name)}"></div>
          <div class="form-field"><label for="boundaryFace">対象面</label><select id="boundaryFace" data-boundary-field="face">${faceChoices.map((item) => `<option value="${esc(item.value)}" ${boundary.face === item.value ? 'selected' : ''}>${esc(item.label)}</option>`).join('')}</select></div>
        </div>
        <div class="boundary-card">
          <div class="subhead"><span>Flow / 流れ</span><span>${esc(FACE_INFO[boundary.face]?.axis || (isCadFace(boundary.face) ? 'CAD' : ''))}</span></div>
          <div class="form-field"><label for="flowType">条件タイプ</label><select id="flowType" data-boundary-field="flow.type"><option value="velocity" ${flowType === 'velocity' ? 'selected' : ''} ${cadFallback ? 'disabled' : ''}>速度指定</option><option value="pressure" ${flowType === 'pressure' ? 'selected' : ''} ${cadFallback ? 'disabled' : ''}>圧力指定</option><option value="wall" ${flowType === 'wall' ? 'selected' : ''}>壁（no-slip）</option></select>${cadFallback ? '<small>CAD 全体の既定面は壁（no-slip）として扱います。個別の CAD 面では速度・圧力も指定できます。</small>' : cadFace ? '<small>個別 CAD 面の境界条件です。</small>' : ''}</div>
          ${flowType === 'velocity' ? `<div class="form-field"><label>速度ベクトル [m/s]</label><div class="vector-grid">${['x', 'y', 'z'].map((axis, index) => `<div class="form-field"><span class="vector-label">${axis}</span><input class="numeric" type="number" step="any" data-boundary-vector="${index}" value="${esc(flow.velocity?.[index] ?? 0)}" aria-label="速度 ${axis}"></div>`).join('')}</div></div>` : ''}
          ${flowType === 'pressure' ? `<div class="form-field"><label for="flowPressure">静圧</label><div class="input-with-unit"><input id="flowPressure" class="numeric" type="number" step="any" data-boundary-field="flow.value" value="${esc(flow.value ?? 0)}"><span class="input-unit">Pa</span></div></div>` : ''}
        </div>
        <div class="boundary-card">
          <div class="subhead"><span>Thermal / 熱</span><span>${state.project.physics.thermal ? 'ACTIVE' : 'OFF'}</span></div>
          <div class="form-field"><label for="thermalType">条件タイプ</label><select id="thermalType" data-boundary-field="thermal.type"><option value="temperature" ${thermalType === 'temperature' ? 'selected' : ''}>温度指定</option><option value="heat_flux" ${thermalType === 'heat_flux' ? 'selected' : ''}>熱流束</option><option value="convection" ${thermalType === 'convection' ? 'selected' : ''}>対流</option><option value="adiabatic" ${thermalType === 'adiabatic' ? 'selected' : ''}>断熱</option></select></div>
          ${thermalType === 'temperature' ? `<div class="form-field"><label for="boundaryTemperature">表面温度</label><div class="input-with-unit"><input id="boundaryTemperature" class="numeric" type="number" step="any" data-boundary-field="thermal.value" value="${esc(thermal.value ?? 293.15)}"><span class="input-unit">K</span></div></div>` : ''}
          ${thermalType === 'heat_flux' ? `<div class="form-field"><label for="boundaryHeatFlux">内向き熱流束</label><div class="input-with-unit"><input id="boundaryHeatFlux" class="numeric" type="number" step="any" data-boundary-field="thermal.value" value="${esc(thermal.value ?? 0)}"><span class="input-unit">W/m²</span></div><small>正の値は領域へ流入する向きです。</small></div>` : ''}
          ${thermalType === 'convection' ? `<div class="field-grid"><div class="form-field"><label for="boundaryH">熱伝達係数 h</label><div class="input-with-unit"><input id="boundaryH" class="numeric" type="number" min="0" step="any" data-boundary-field="thermal.h" value="${esc(thermal.h ?? 10)}"><span class="input-unit">W/m²K</span></div></div><div class="form-field"><label for="ambientTemperature">周囲温度</label><div class="input-with-unit"><input id="ambientTemperature" class="numeric" type="number" min="0" step="any" data-boundary-field="thermal.ambient_temperature" value="${esc(thermal.ambient_temperature ?? 293.15)}"><span class="input-unit">K</span></div></div></div>` : ''}
        </div>
        <div class="button-row"><button type="button" class="button ghost compact" data-action="remove-boundary" data-face="${esc(boundary.face)}">境界を削除</button></div>
      `}`;
  }

  function meshInspector() {
    const mesh = state.project.mesh;
    const target = mesh.target_cells ?? mesh.cells.reduce((a, b) => a * b, 1);
    scheduleMeshEstimate();
    return `
      <div class="property-section">
        <div class="section-title"><span>目標総セル数</span><small>ISOTROPIC GRID</small></div>
        ${state.limits ? `<div class="info-note">上限: 総セル数 ${Number(state.limits.max_total_cells).toLocaleString()} / 各軸 ${Number(state.limits.max_axis_cells).toLocaleString()}<br>メッシュ受付上限です。計算には別途 RAM・GPU メモリが必要です。</div>` : ''}
        <div class="form-field"><label for="meshTarget">目標総セル数</label><input id="meshTarget" class="numeric" type="number" min="1" ${state.limits ? `max="${state.limits.max_total_cells}"` : ''} step="1" data-bind="mesh.target_cells" value="${esc(target)}"></div>
        <div class="info-note">縦横比に応じて各軸の分割数を自動計算します。整数分割のため実際の総数は目標値と異なる場合があります。総数には固体・領域外セルも含まれます。</div>
        ${mesh.target_cells === undefined ? '<div class="info-note">既存の各軸分割数を保持しています。目標総セル数を入力すると自動計算に切り替わります。</div>' : ''}
        <div id="meshEstimate">${meshEstimateMarkup()}</div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>メッシュ状態</span><small>${state.mesh ? 'BUILT' : 'NOT BUILT'}</small></div>
        ${state.mesh ? `<div class="asset-card"><div class="asset-card-head"><strong>プレビュー生成済み</strong><span class="asset-kind">${esc(state.mesh.shape ? state.mesh.shape.join(' × ') : 'grid')}</span></div><span>総セル数: ${esc(fmt(state.mesh.total_cells, 0))}</span><span>fluid cells: ${esc(state.mesh.fluid_cells ?? '—')}</span>${state.mesh.solid_cells !== undefined ? `<span>solid cells: ${esc(state.mesh.solid_cells)}</span>` : ''}<span>spacing: ${esc((state.mesh.spacing || []).map((value) => fmt(value, 5)).join(' / '))} m</span></div>` : '<div class="info-note">設定を確認してから「メッシュ生成」を実行してください。</div>'}
        <div class="button-row"><button type="button" class="button primary compact" data-action="build-mesh">${state.mesh ? '↻ 再生成' : '▦ メッシュ生成'}</button></div>
      </div>`;
  }

  function meshEstimateMarkup() {
    const estimate = state.meshEstimate;
    if (!estimate || estimate.pending) return '<div class="info-note">分割数を計算中…</div>';
    if (estimate.error) {
      const message = estimate.error.includes('exact isotropic mesh is impossible')
        ? '領域寸法を保つ等方格子を、現在の各軸・総セル数上限内では作成できません。寸法比と上限を確認してください。'
        : estimate.error.includes('CAD target mesh cannot satisfy')
          ? 'CAD形状を捉える等方格子が、現在の各軸・総セル数上限に収まりません。'
          : estimate.error;
      return `<div class="warn-note">${esc(message)}</div>`;
    }
    const difference = estimate.target_cells ? estimate.total_cells - estimate.target_cells : 0;
    const warnings = (estimate.warnings || []).filter(w => !w.startsWith('target total cell count')).map(w => w.includes('evenly centred exterior padding') ? 'CAD外周に、等方格子への丸めによる余白を均等に追加します。' : w);
    return `<div class="asset-card"><strong>実際の総セル数: ${esc(fmt(estimate.total_cells, 0))}</strong><span>X × Y × Z: ${esc(estimate.shape.join(' × '))}</span><span>等方セル幅: ${esc(Number(estimate.spacing[0]).toPrecision(6))} m</span>${difference ? `<span>目標との差: ${difference > 0 ? '+' : ''}${esc(fmt(difference, 0))} セル</span>` : ''}${warnings.map(w => `<span>${esc(w)}</span>`).join('')}</div>`;
  }

  function scheduleMeshEstimate() {
    const payload = serializableProject();
    const key = JSON.stringify(payload);
    if (state.meshEstimateKey === key) return;
    state.meshEstimateKey = key;
    state.meshEstimate = { pending: true };
    clearTimeout(state.meshEstimateTimer);
    const target = $('#meshEstimate');
    if (target) target.innerHTML = meshEstimateMarkup();
    state.meshEstimateTimer = setTimeout(async () => {
      let estimate;
      try { estimate = await request('/api/mesh/estimate', { method: 'POST', body: payload }); }
      catch (error) { estimate = { error: error.message }; }
      if (state.meshEstimateKey !== key || JSON.stringify(serializableProject()) !== key) return;
      state.meshEstimate = estimate;
      const output = $('#meshEstimate');
      if (output) output.innerHTML = meshEstimateMarkup();
      const stability = $('#flowTimeStepInfo');
      if (stability) stability.innerHTML = flowTimeStepMarkup();
      updateHeader();
      updateValidationSummary();
    }, 300);
  }

  function studyInspector() {
    const study = state.project.study;
    const monitor = normalizeMonitor(study.monitor);
    scheduleMeshEstimate();
    const outputInvalid = study.output_interval > study.steps;
    const gpuRamMode = study.device === 'cuda:0-ram';
    return `
      <div class="property-section">
        <div class="section-title"><span>時間積分</span><small>TIME STEPPING</small></div>
        <div class="field-grid"><div class="form-field"><label for="studySteps">ステップ数</label><input id="studySteps" class="numeric" type="number" min="1" step="1" data-bind="study.steps" value="${esc(study.steps)}"></div><div class="form-field"><label for="studyInterval">出力間隔</label><input id="studyInterval" class="numeric" type="number" min="1" step="1" data-bind="study.output_interval" value="${esc(study.output_interval)}"></div></div>
        <div class="form-field"><label for="studyDt">時間刻み Δt</label><div class="input-with-unit"><input id="studyDt" class="numeric" type="number" min="0.000000001" step="any" data-bind="study.dt" value="${esc(study.dt)}"><span class="input-unit">s</span></div></div>
        <div id="flowTimeStepInfo">${flowTimeStepMarkup()}</div>
        ${outputInvalid ? '<div class="warn-note">出力間隔はステップ数以下にしてください。</div>' : ''}
      </div>
      <div class="property-section monitor-settings">
        <div class="section-title"><span>残差モニター</span><small>STEADY STATE</small></div>
        <div class="info-note monitor-definition"><strong>定常性の目安（前回記録からの相対変化）</strong><br>出力間隔ごとに速度・圧力・温度の変化を記録します。すべての有効な場が閾値以下の状態を、連続サンプル数だけ確認します。</div>
        <div class="field-grid"><div class="form-field"><label for="studyMonitorTolerance">許容値</label><div class="input-with-unit"><input id="studyMonitorTolerance" class="numeric" type="number" min="0.000000000001" step="any" data-bind="study.monitor.tolerance" value="${esc(monitor.tolerance)}"><span class="input-unit">相対</span></div></div><div class="form-field"><label for="studyMonitorSamples">必要な連続サンプル</label><div class="input-with-unit"><input id="studyMonitorSamples" class="numeric" type="number" min="1" max="100" step="1" data-bind="study.monitor.consecutive_samples" value="${esc(monitor.consecutive_samples)}"><span class="input-unit">点</span></div></div></div>
        <small class="muted monitor-settings-note">既定値: ${fmtSci(DEFAULT_MONITOR.tolerance)} / ${DEFAULT_MONITOR.consecutive_samples} 点。出力間隔がモニターのサンプル間隔です。自動停止は行いません。</small>
        <div class="warn-note monitor-transient-note">過渡計算や LES では物理的な揺らぎが残るため、定常性の目安が継続して満たされない場合があります。</div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>結果スナップショット</span><small>TIME SERIES</small></div>
        <div class="form-field"><label for="studySnapshotInterval">保存間隔</label><div class="input-with-unit"><input id="studySnapshotInterval" class="numeric" type="number" min="0" step="1" data-bind="study.snapshot_interval" value="${esc(study.snapshot_interval)}"><span class="input-unit">step</span></div><small>0 = 無効。間隔ごとに保存した場の結果を時系列再生できます。</small></div>
        <div class="info-note">保存間隔を短くすると結果ファイルの容量と転送量が増えます。目安: 1 frame は 6 float32/セル ≈ 24 bytes/セル（300 万セルで約 72 MB）。全ステップ保存は 1 を指定してください。</div>
      </div>
      <div class="property-section">
        <div class="section-title"><span>実行デバイス</span><small>EXECUTION</small></div>
        <div class="form-field"><label for="studyDevice">デバイス</label><select id="studyDevice" data-bind="study.device"><option value="cpu" ${study.device === 'cpu' ? 'selected' : ''}>CPU（標準）</option><option value="cuda:0" ${study.device === 'cuda:0' ? 'selected' : ''}>CUDA GPU 0</option><option value="cuda:0-ram" ${gpuRamMode ? 'selected' : ''}>GPU＋RAM（大規模・低速）</option></select><small>利用できないデバイスはサーバー側でエラーになります。</small></div>
        ${gpuRamMode ? `<div class="form-field"><label for="gpuBatchCells">GPU バッチセル数</label><div class="input-with-unit"><input id="gpuBatchCells" class="numeric" type="number" min="1024" max="1048576" step="1" data-bind="study.gpu_batch_cells" value="${esc(study.gpu_batch_cells)}"><span class="input-unit">cells</span></div><small>1,024–1,048,576 cells。大きい値は GPU 転送回数を減らしますが、VRAM 使用量が増えます。</small></div><div class="info-note">全体の格子は RAM に保持し、XLB の衝突計算だけを GPU へ分割します。移流・熱・LES は CPU で処理します。GPU＋RAM は VRAM と RAM を単純合算する方式ではなく、RAM 容量にも制限されます。</div>` : ''}
      </div>
      <div class="info-note">計算開始前に現在のプロジェクトを保存し、保存時点の JSON をスナップショットして実行します。</div>`;
  }

  function flowTimeStepMarkup() {
    if (!state.project.physics.flow) return '';
    const estimate = state.meshEstimate;
    if (!estimate || estimate.pending) return '<div class="info-note">格子幅と境界速度から時間刻みの上限を確認中…</div>';
    if (estimate.error) return `<div class="warn-note">時間刻みの確認: ${esc(estimate.error)}</div>`;
    const stability = estimate.flow_stability;
    const machLimit = Number(stability?.dt_max_exclusive);
    const forceLimit = Number(stability?.force_dt_max);
    const hasMachLimit = Number.isFinite(machLimit) && machLimit > 0;
    const hasForceLimit = Number.isFinite(forceLimit) && forceLimit > 0;
    if (!hasMachLimit && !hasForceLimit) return '<div class="info-note">非ゼロの速度境界がないため、CFLから推奨時間刻みを算出できません。圧力駆動などの場合は想定する流速に基づく確認が必要です。</div>';
    const exceeded = hasMachLimit && Number(stability.boundary_mach) >= Number(stability.mach_limit);
    const forceExceeded = hasForceLimit && Number(state.project.study.dt) > forceLimit;
    const recommendationBasis = hasMachLimit && hasForceLimit ? 'CFL・重力から' : hasForceLimit ? '重力から' : 'CFLから';
    const recommendationHint = hasMachLimit && hasForceLimit
      ? '速度境界と重力・浮力の制約から算出。設定値は自動変更しません。'
      : hasForceLimit ? '重力・浮力による格子速度増分と圧力差の制約から算出。設定値は自動変更しません。'
        : '境界速度から算出。設定値は自動変更しません。';
    const recommendation = Number(stability.recommended_dt) > 0
      ? `<div class="form-field"><label for="recommendedDt">${recommendationBasis}推奨 Δt${hasMachLimit ? `（目標 ${esc(stability.cfl_target)}）` : ''}</label><div class="input-with-unit"><input id="recommendedDt" class="numeric" type="text" readonly value="${esc(Number(stability.recommended_dt).toExponential(6))}"><span class="input-unit">s</span></div><small>${recommendationHint}</small></div>` : '';
    const forceBasis = stability.force_dt_basis === 'uniform gravity'
      ? '一定重力'
      : stability.force_dt_basis === '10% buoyancy density contrast' ? '浮力（密度差 10% の上限）' : '体積力';
    const cflLines = hasMachLimit
      ? `現在の CFL: ${esc(Number(stability.boundary_cfl).toPrecision(4))}（目標 ${esc(stability.cfl_target)}）<br>CFL = Δt × (|uₓ|/Δx + |uᵧ|/Δy + |u_z|/Δz)<br>LBM 格子マッハ数: ${esc(Number(stability.boundary_mach).toPrecision(4))}（${esc(stability.mach_limit)} 未満が必要）<br>LBMによる Δt 上限: ${esc(machLimit.toPrecision(6))} s 未満。`
      : '非ゼロの速度境界がないため、CFL・マッハ数の上限は算出されていません。';
    const forceLine = hasForceLimit
      ? `<br>重力・浮力による Δt 上限: ${esc(forceLimit.toPrecision(6))} s 以下（${esc(forceBasis)}、加速と重力による圧力差から見積。|a|Δt²/Δx ≤ 0.05）。`
      : '';
    const errorLead = exceeded || forceExceeded ? '<strong>時間刻みが大きすぎます。</strong><br>' : '';
    return `${recommendation}<div class="${exceeded || forceExceeded ? 'error-note' : 'info-note'}">${errorLead}${cflLines}${forceLine}<br>推奨値は目標CFL、LBM上限、重力の格子速度増分上限に余裕を設けた最小値です。計算中の内部速度増加、熱拡散・境界の熱損失、物性による緩和時間の制約は別途確認が必要です。<br>現在の計算時間: ${esc(Number(state.project.study.steps * state.project.study.dt).toPrecision(6))} s。Δt を小さくすると、同じステップ数で進む時間も短くなります。</div>`;
  }

  function resultsInspector() {
    const run = state.run;
    ensureRestartAvailability(run);
    const shape = run?.shape || state.mesh?.shape || state.project.mesh.cells;
    const fields = availableResultFields(run);
    const requestedField = state.resultControls?.field || state.slice?.field || fields[0]?.id || 'temperature';
    const field = fields.some((item) => item.id === requestedField) ? requestedField : (fields[0]?.id || 'temperature');
    const axis = state.resultControls?.axis || state.slice?.axis || 'y';
    const axisLength = shape[['x', 'y', 'z'].indexOf(axis)] || 1;
    const index = clamp(numberOr(state.resultControls?.index ?? state.slice?.index, Math.floor((axisLength - 1) / 2)), 0, Math.max(0, axisLength - 1));
    const resultStatus = run ? runStatusLabel(run.status) : '未実行';
    const runs = availableRuns();
    const frames = state.frames.slice();
    const frameStep = state.selectedFrameStep !== null && state.selectedFrameStep !== undefined
      ? state.selectedFrameStep : state.resultControls?.frameStep ?? null;
    const activeRun = ['queued', 'running', 'stopping'].includes(run?.status);
    const canLoadSlice = Boolean(run && (['completed', 'stopped'].includes(run?.status) || frames.length));
    const frameOptions = frames.map((frame) => `<option value="${esc(frame.step)}" ${frameStep !== null && frameStep !== undefined && Number(frameStep) === frame.step ? 'selected' : ''}>step ${esc(frame.step)}${frame.time !== null ? ` · t ${esc(fmt(frame.time, 6))} s` : ''}</option>`).join('');
    const frameValue = frameStep === null || frameStep === undefined ? '' : String(frameStep);
    const latest = frames.length ? frames[frames.length - 1] : null;
    const warnings = runWarnings(run);
    const solidCells = Number(run?.solid_cells || run?.diagnostics?.solid_cells || 0);
    const sliceHasSolid = maskContainsSolid(state.slice?.solid_mask);
    const coupledDomain = solidCells > 0 || sliceHasSolid;
    const fieldNote = field.startsWith('heat_flux') ? '<br>q = −k∇T（移流・LES 分を含まない伝導成分）' : '';
    const pressureNote = field === 'pressure' ? runPressureNote(run) : '';
    const fieldMeta = fields.find((item) => item.id === field);
    const domainNote = sliceHasSolid
      ? (fieldUsesThermalMask(field, fieldMeta) ? '流体・固体の値を含む断面です。' : '固体領域は非表示です。流体フィールドのみ表示しています。')
      : '';
    return `
      <div class="property-section">
        <div class="section-title"><span>実行を選択</span><small>${runs.length} RUN${runs.length === 1 ? '' : 'S'}</small></div>
        <div class="form-field"><label for="runSelect">保存済み実行</label><select id="runSelect" data-action="select-run"><option value="" disabled ${run ? '' : 'selected'}>実行を選択してください</option>${runs.map((item) => `<option value="${esc(item.id)}" ${run?.id === item.id ? 'selected' : ''}>${esc(runStatusLabel(item.status))} · ${esc(String(item.id).slice(0, 12))}${item.created_at ? ` · ${esc(new Date(item.created_at).toLocaleString('ja-JP'))}` : ''}</option>`).join('')}</select></div>
        <div class="section-title"><span>実行状態</span><small>${esc(String(run?.id || '—').slice(0, 14))}</small></div>
        <div class="asset-card"><div class="asset-card-head"><strong>${esc(resultStatus)}</strong><span class="asset-kind">${run ? esc(`${Math.round((run.progress || 0) * 100)}%`) : '—'}</span></div><span>${run ? `steps: ${run.restart_from && run.start_step !== undefined && run.end_step !== undefined ? `${esc(run.start_step)} → ${esc(run.end_step)}` : esc(run.steps ?? '—')} / ${esc(run.converged === true ? 'converged' : run.converged === false ? 'not converged' : 'pending')}` : '計算を開始すると結果を取得できます。'}</span><span>${coupledDomain ? `CHT · solid cells ${fmt(solidCells, 0)}` : 'thermal domain · fluid'}</span></div>
        ${run ? `<div class="muted" style="font-size:10px">この結果は run の入力スナップショットから生成されています。</div>` : ''}
        ${run?.restart_from ? `<div class="info-note" style="margin-top:8px">追加計算: run ${esc(String(run.restart_from).slice(0, 14))} の step ${esc(run.start_step ?? '—')} から step ${esc(run.end_step ?? run.steps ?? '—')} まで</div>` : ''}
        ${run?.status === 'failed' ? `<div class="error-note" role="alert" style="margin-top:8px;overflow-wrap:anywhere"><strong>計算の停止理由</strong><br>${String(run.error || '').includes('boundary Mach number') ? '境界速度に対して時間刻みが大きすぎます。「スタディ」で Δt を小さくしてください。<br>' : ''}${esc(run.error || run.message || '詳細は実行の error.log を確認してください。')}</div>` : ''}
        ${warnings.length ? `<div class="warn-note" style="margin-top:8px"><strong>計算時の注意</strong><br>${warnings.slice(0, 4).map((warning) => esc(warning)).join('<br>')}</div>` : ''}
      </div>
      ${monitorInspectorMarkup(run)}
      ${restartInspectorMarkup(run)}
      <div class="property-section">
        <div class="section-title"><span>時系列</span><small>${frames.length ? `${frames.length} SAVED STEPS` : 'NO SAVED STEPS'}</small></div>
        <div class="form-field"><label for="resultFrame">保存ステップ</label><select id="resultFrame" data-result-frame ${frames.length ? '' : 'disabled'}><option value="" ${frameValue === '' ? 'selected' : ''}>最終結果${activeRun ? '（保存済みなし）' : ''}</option>${frameOptions}</select></div>
        <div class="frame-toolbar"><button type="button" class="button secondary compact" data-action="toggle-result-playback" ${frames.length < 2 ? 'disabled' : ''}>${state.playbackActive ? '■ 停止' : '▶ 再生'}</button><label class="frame-follow"><input type="checkbox" data-result-follow ${state.followLatest ? 'checked' : ''} ${activeRun ? '' : 'disabled'}>最新に追従</label><span class="frame-status">${frames.length ? `最新 step ${esc(latest.step)}${latest.time !== null ? ` · t ${esc(fmt(latest.time, 6))} s` : ''}` : (state.framesLoading ? '保存ステップを確認中…' : '保存されたステップはありません')}</span></div>
        ${activeRun && !frames.length ? '<div class="info-note">計算中は保存済みスナップショットが到着するとここから表示できます。保存間隔 0 の場合は完了後の最終結果だけを表示します。</div>' : frames.length ? '<div class="info-note">保存ステップを選択して再生できます。スナップショットの読み込みは選択した 1 枚だけです。</div>' : '<div class="info-note">保存ステップがありません。保存間隔 0 の実行では最終結果のみ取得できます。</div>'}
      </div>
      <div class="property-section">
        <div class="section-title"><span>スライス表示</span><small>ACTUAL FIELD</small></div>
        <div class="form-field"><label for="resultField">フィールド</label><select id="resultField" data-result-field ${run ? '' : 'disabled'}>${fields.map((item) => `<option value="${esc(item.id)}" ${field === item.id ? 'selected' : ''}>${esc(item.label)}${item.unit ? ` [${esc(item.unit)}]` : ''}${item.mask ? ` · ${esc(formatFieldMask(item.mask))}` : ''}</option>`).join('')}</select></div>
        <div class="field-grid"><div class="form-field"><label for="resultAxis">断面</label><select id="resultAxis" data-result-axis><option value="x" ${axis === 'x' ? 'selected' : ''}>YZ（X 断面）</option><option value="y" ${axis === 'y' ? 'selected' : ''}>XZ（Y 断面）</option><option value="z" ${axis === 'z' ? 'selected' : ''}>XY（Z 断面）</option></select></div><div class="form-field"><label for="resultIndex">インデックス / ${Math.max(0, axisLength - 1)}</label><input id="resultIndex" class="numeric" type="number" min="0" max="${Math.max(0, axisLength - 1)}" step="1" data-result-index value="${esc(index)}"></div></div>
        ${pressureNote ? `<div class="info-note">${esc(pressureNote)}</div>` : ''}
        <div class="button-row"><button type="button" class="button primary compact" data-action="load-slice" ${canLoadSlice ? '' : 'disabled'}>${state.sliceLoading ? '取得中…' : '↻ スライス取得'}</button></div>
        ${state.slice ? `<div class="info-note">min ${fmtSci(state.slice.min)} / max ${fmtSci(state.slice.max)} ${esc(state.slice.unit || '')}${fieldNote}${domainNote ? `<br>${domainNote}` : ''}</div>` : '<div class="info-note">完了した計算のフィールドだけを表示します。</div>'}
      </div>
      <div class="property-section"><div class="section-title"><span>ファイル</span><small>RUN ARTIFACTS</small></div><div class="button-row"><button type="button" class="button ghost compact" data-action="download-history" ${run ? '' : 'disabled'}>history.csv</button><button type="button" class="button ghost compact" data-action="download-vtk" ${run ? '' : 'disabled'}>fields.vtk</button><button type="button" class="button ghost compact" data-action="download-npz" ${run ? '' : 'disabled'}>fields.npz</button><button type="button" class="button ghost compact" data-action="download-input" ${run ? '' : 'disabled'}>input.json</button></div></div>`;
  }

  function normalizeFrame(frame) {
    if (frame === null || frame === undefined) return null;
    const source = typeof frame === 'object' ? frame : { step: frame };
    const rawStep = source.step ?? source.iteration ?? source.time_step;
    const step = Number(rawStep);
    if (!Number.isFinite(step)) return null;
    const rawTime = source.time ?? source.time_s ?? source.t;
    const time = Number(rawTime);
    return { ...source, step: Math.max(0, Math.round(step)), time: Number.isFinite(time) ? time : null };
  }

  function normalizeResultField(field) {
    if (field === null || field === undefined) return null;
    const source = typeof field === 'object' ? field : { id: field };
    const id = String(source.id ?? source.field ?? '');
    if (!id) return null;
    const fallback = FIELD_INFO[id] || { label: id, title: id, unit: '', color: 'temperature' };
    // Known quantities use one stable UI vocabulary even when an older API
    // advertises a slightly different label.  Units and masks remain API data.
    return { ...source, id, label: String(fallback.label || source.label || id), title: String(fallback.title || source.title || source.label || id), unit: String(source.unit ?? fallback.unit ?? ''), mask: source.mask ?? null };
  }

  function availableResultFields(run = state.run) {
    if (state.resultFields.length) return state.resultFields;
    const ids = ['temperature', 'speed', 'pressure'];
    if (runSupportsEddyViscosity(run)) ids.push('eddy_viscosity');
    return ids.map((id) => normalizeResultField(id)).filter(Boolean);
  }

  function formatFieldMask(mask) {
    const raw = Array.isArray(mask) ? mask.join(' / ') : mask && typeof mask === 'object' ? (mask.label || mask.name || mask.kind || 'mask') : String(mask ?? '');
    const normalized = raw.toLowerCase();
    if (normalized.includes('thermal') || normalized.includes('solid')) return '流体・固体';
    if (normalized.includes('fluid')) return '流体';
    return raw;
  }

  function fieldUsesThermalMask(field, metadata = null) {
    const mask = metadata?.mask;
    const maskText = Array.isArray(mask) ? mask.join(' ') : mask && typeof mask === 'object' ? JSON.stringify(mask) : String(mask || '');
    if (/thermal|solid/i.test(maskText)) return true;
    if (/fluid/i.test(maskText)) return false;
    return ['temperature', 'density', 'heat_flux', 'heat_flux_x', 'heat_flux_y', 'heat_flux_z', 'conductivity', 'heat_capacity'].includes(field);
  }

  function availableRuns() {
    const runs = Array.isArray(state.runs) ? state.runs.slice() : [];
    if (state.run?.id && !runs.some((item) => item.id === state.run.id)) runs.unshift(state.run);
    return runs.filter((item) => item && item.id);
  }

  function restartRunIsActive(run = state.run) {
    return ['queued', 'running', 'stopping'].includes(run?.status);
  }

  function restartCacheKey(run = state.run) {
    if (!run?.id) return null;
    return `${String(run.id)}:${String(run.status || '')}`;
  }

  function normalizeRestartAvailability(payload) {
    const source = payload && typeof payload === 'object' ? payload : {};
    const stepValue = Number(source.step);
    const dtValue = Number(source.dt);
    const timeValue = Number(source.time);
    return {
      ...source,
      available: source.available === true,
      reason: source.reason ? String(source.reason) : '',
      step: Number.isFinite(stepValue) ? Math.max(0, Math.round(stepValue)) : null,
      dt: Number.isFinite(dtValue) && dtValue > 0 ? dtValue : null,
      time: Number.isFinite(timeValue) ? timeValue : null
    };
  }

  function restartMetrics(run = state.run, availability = state.restartAvailability, additionalSteps = state.restartAdditionalSteps) {
    // Only the restart endpoint knows which checkpoint is actually present.
    // Do not preview a continuation from the editable project's current dt or
    // from a requested end step while availability is still unknown.
    const authoritative = availability?.available === true;
    const sourceStepValue = Number(authoritative ? availability.step : NaN);
    const sourceStep = Number.isFinite(sourceStepValue) ? Math.max(0, Math.round(sourceStepValue)) : null;
    const dtValue = Number(authoritative ? availability.dt : NaN);
    const dt = Number.isFinite(dtValue) && dtValue > 0 ? dtValue : null;
    const explicitTime = Number(authoritative ? availability.time : NaN);
    const sourceTime = Number.isFinite(explicitTime)
      ? explicitTime
      : (sourceStep !== null && dt !== null ? sourceStep * dt : null);
    const stepValue = Number(additionalSteps);
    const steps = Number.isInteger(stepValue) && stepValue > 0 ? stepValue : null;
    const duration = steps !== null && dt !== null ? steps * dt : null;
    return { sourceStep, dt, sourceTime, steps, duration, endTime: sourceTime !== null && duration !== null ? sourceTime + duration : null };
  }

  function restartSourceLabel(metrics) {
    if (metrics.sourceStep === null && metrics.sourceTime === null) return 'チェックポイント情報を確認中…';
    const step = metrics.sourceStep === null ? '—' : fmt(metrics.sourceStep, 0);
    const time = metrics.sourceTime === null ? '—' : `${fmtSci(metrics.sourceTime)} s`;
    return `チェックポイント: step ${step} · t ${time}`;
  }

  function restartDurationLabel(metrics) {
    return metrics.duration === null ? '追加時間: —' : `追加時間: ${fmtSci(metrics.duration)} s`;
  }

  function restartEndTimeLabel(metrics) {
    return metrics.endTime === null ? '終了時刻: —' : `終了時刻: t ${fmtSci(metrics.endTime)} s`;
  }

  function updateRestartAdditionalInput(element) {
    const raw = String(element?.value ?? '').trim();
    const value = Number(raw);
    state.restartAdditionalSteps = raw && Number.isFinite(value) ? value : 0;
    const key = restartCacheKey(state.run);
    const availability = key && key === state.restartAvailabilityKey ? state.restartAvailability : null;
    const metrics = restartMetrics(state.run, availability, state.restartAdditionalSteps);
    const duration = $('[data-restart-duration]');
    const endTime = $('[data-restart-endtime]');
    if (duration) duration.textContent = restartDurationLabel(metrics);
    if (endTime) endTime.textContent = restartEndTimeLabel(metrics);
    const button = $('[data-action="restart-simulation"]');
    if (button) {
      const canRestart = Boolean(availability?.available) && !restartRunIsActive() && metrics.steps !== null;
      button.disabled = !canRestart || state.restartSubmitting || state.restartAvailabilityLoading;
    }
  }

  function ensureRestartAvailability(run = state.run) {
    const key = restartCacheKey(run);
    if (!key) {
      state.restartAvailability = null;
      state.restartAvailabilityKey = null;
      state.restartAvailabilityLoading = false;
      return;
    }
    if (state.restartAvailabilityKey === key && (state.restartAvailabilityLoading || state.restartAvailability)) return;
    state.restartAvailabilityKey = key;
    state.restartAvailability = null;
    state.restartAvailabilityLoading = false;
    state.restartAdditionalSteps = 200;
    const cached = state.restartCache.get(key);
    if (cached) {
      state.restartAvailability = cached;
      return;
    }
    if (restartRunIsActive(run)) {
      const activeInfo = normalizeRestartAvailability({ available: false, reason: 'キュー待ちまたは計算中の実行は再開できません。' });
      state.restartCache.set(key, activeInfo);
      state.restartAvailability = activeInfo;
      return;
    }
    const requestId = state.restartAvailabilityRequest = (state.restartAvailabilityRequest || 0) + 1;
    state.restartAvailabilityLoading = true;
    request(`/api/runs/${encodeURIComponent(run.id)}/restart`).then((payload) => {
      if (state.restartAvailabilityRequest !== requestId || restartCacheKey(state.run) !== key) return;
      const availability = normalizeRestartAvailability(payload);
      state.restartCache.set(key, availability);
      state.restartAvailability = availability;
      state.restartAvailabilityLoading = false;
      renderInspector();
    }).catch((error) => {
      if (state.restartAvailabilityRequest !== requestId || restartCacheKey(state.run) !== key) return;
      const availability = normalizeRestartAvailability({ available: false, reason: `再開情報を取得できません: ${error.message}` });
      state.restartCache.set(key, availability);
      state.restartAvailability = availability;
      state.restartAvailabilityLoading = false;
      pushLog(availability.reason, 'warn');
      renderInspector();
    });
  }

  function restartInspectorMarkup(run) {
    const key = restartCacheKey(run);
    const availability = key && key === state.restartAvailabilityKey ? state.restartAvailability : null;
    const loading = Boolean(key && state.restartAvailabilityKey === key && state.restartAvailabilityLoading);
    const active = restartRunIsActive(run);
    const additionalValue = Number.isInteger(state.restartAdditionalSteps) && state.restartAdditionalSteps > 0 ? state.restartAdditionalSteps : '';
    const metrics = restartMetrics(run, availability, additionalValue);
    const canRestart = Boolean(run && availability?.available && !active && metrics.steps !== null && !loading && !state.restartSubmitting);
    let checkpoint = '';
    if (loading) checkpoint = '<div class="info-note">再開可能か確認中…</div>';
    else if (availability?.available) checkpoint = `<div class="asset-card"><div class="asset-card-head"><strong>再開可能</strong><span class="asset-kind">CHECKPOINT</span></div><span>${esc(restartSourceLabel(metrics))}</span><span>Δt: ${metrics.dt === null ? '—' : `${fmtSci(metrics.dt)} s`}</span></div>`;
    else if (availability?.reason) checkpoint = `<div class="warn-note" role="status">${esc(availability.reason)}</div>`;
    else checkpoint = '<div class="info-note">実行を選択すると再開可能か確認します。</div>';
    const notice = state.restartNotice || (availability?.available ? '元の実行条件（メッシュ・Δt・物性・境界）を再利用します。現在の編集内容は追加計算には使用されません。' : '');
    return `
      <div class="property-section">
        <div class="section-title"><span>リスタート</span><small>CONTINUE RUN</small></div>
        ${checkpoint}
        <div class="form-field"><label for="restartAdditionalSteps">追加ステップ数</label><div class="input-with-unit"><input id="restartAdditionalSteps" class="numeric" type="number" min="1" step="1" data-restart-additional value="${esc(additionalValue)}" ${availability?.available && !active ? '' : 'disabled'}><span class="input-unit">step</span></div></div>
        <div class="info-note"><span data-restart-duration>${esc(restartDurationLabel(metrics))}</span><br><span data-restart-endtime>${esc(restartEndTimeLabel(metrics))}</span></div>
        ${notice ? `<div class="info-note" role="status">${esc(notice)}</div>` : ''}
        <div class="button-row"><button type="button" class="button primary compact" data-action="restart-simulation" ${canRestart ? '' : 'disabled'}>${state.restartSubmitting ? '開始中…' : '追加計算を開始'}</button></div>
      </div>`;
  }

  function monitorDescriptor(run = state.run) {
    const candidates = [
      run?.input_snapshot?.study?.monitor,
      run?.input?.study?.monitor,
      run?.project?.study?.monitor,
      run?.study?.monitor,
      run?.monitor
    ];
    return candidates.find((candidate) => candidate && typeof candidate === 'object' && !Array.isArray(candidate)) || null;
  }

  function firstPositiveNumber(values, fallback) {
    for (const value of values) {
      const parsed = nullableNumber(value);
      if (parsed !== null && parsed > 0) return parsed;
    }
    return fallback;
  }

  function firstNonNegativeInteger(values, fallback) {
    for (const value of values) {
      const parsed = nullableNumber(value);
      if (parsed !== null && Number.isInteger(parsed) && parsed >= 0) return parsed;
    }
    return fallback;
  }

  function firstPositiveInteger(values, fallback) {
    for (const value of values) {
      const parsed = nullableNumber(value);
      if (parsed !== null && Number.isInteger(parsed) && parsed > 0) return parsed;
    }
    return fallback;
  }

  function monitorSettingsForRun(run = state.run) {
    const descriptor = monitorDescriptor(run);
    const summary = run?.diagnostics?.convergence_monitor;
    return {
      tolerance: firstPositiveNumber([
        run?.monitor_tolerance,
        run?.monitorTolerance,
        summary?.tolerance,
        descriptor?.tolerance
      ], DEFAULT_MONITOR.tolerance),
      requiredSamples: firstPositiveInteger([
        run?.monitor_required_samples,
        run?.monitorRequiredSamples,
        summary?.required_samples,
        descriptor?.consecutive_samples
      ], DEFAULT_MONITOR.consecutive_samples),
      configured: Boolean(descriptor || summary || nullableNumber(run?.monitor_tolerance) !== null || nullableNumber(run?.monitor_required_samples) !== null)
    };
  }

  function isNativeMonitorRow(row) {
    if (!row) return false;
    return row.residual_kind === MONITOR_RESIDUAL_KIND
      // The older history format also had temperature_residual.  Absolute
      // change columns or a velocity/pressure component identify the new
      // field-change monitor without misclassifying that legacy column.
      || ['monitor_eligible', 'monitor_tolerance', 'monitor_required_samples', 'monitor_consecutive_samples',
          'velocity_residual', 'pressure_residual', ...MONITOR_COMPONENTS.map((series) => series.changeKey)]
          .some((key) => row[key] !== null && row[key] !== undefined);
  }

  function monitorState(run = state.run) {
    const history = Array.isArray(state.history) ? state.history : [];
    const nativeRows = history.filter(isNativeMonitorRow);
    const latest = nativeRows.length ? nativeRows[nativeRows.length - 1] : null;
    const legacyRows = history.filter((row) => Number.isFinite(row.residual) && !isNativeMonitorRow(row));
    const settings = monitorSettingsForRun(run);
    const summary = run?.diagnostics?.convergence_monitor;
    const tolerance = firstPositiveNumber([latest?.monitor_tolerance, summary?.tolerance, settings.tolerance], DEFAULT_MONITOR.tolerance);
    const requiredSamples = firstPositiveInteger([latest?.monitor_required_samples, summary?.required_samples, settings.requiredSamples], DEFAULT_MONITOR.consecutive_samples);
    const consecutive = firstNonNegativeInteger([
      latest?.monitor_consecutive_samples,
      summary?.consecutive_samples,
      run?.monitor_consecutive_samples
    ], 0);
    const satisfied = nullableBoolean(latest?.monitor_satisfied, nullableBoolean(summary?.satisfied, nullableBoolean(run?.monitor_satisfied, false)));
    const eligible = nullableBoolean(latest?.monitor_eligible, nullableBoolean(summary?.eligible, null));
    const residual = nullableNumber(latest?.residual, nullableNumber(summary?.residual, null));
    return {
      history,
      nativeRows,
      latest,
      legacyRows,
      legacy: !nativeRows.length && legacyRows.length > 0,
      settings: { tolerance, requiredSamples },
      consecutive,
      satisfied,
      eligible,
      residual,
      active: ['queued', 'running', 'stopping'].includes(run?.status)
    };
  }

  function monitorStatusText(status) {
    if (!status?.active && !status?.history.length) return state.run ? 'サンプル待ち' : '未実行';
    if (status?.legacy) return '旧形式履歴（統合 residual）';
    if (!status?.nativeRows.length) return status?.active ? 'サンプル待ち' : 'モニター履歴なし';
    if (status.satisfied === true) return '定常性基準を満たしました';
    if (!status.latest) return 'サンプル待ち';
    if (status.eligible === false) return '完全なサンプル待ち';
    if (status.residual !== null && status.residual <= status.settings.tolerance) return `確認中 ${Math.min(status.consecutive, status.settings.requiredSamples)} / ${status.settings.requiredSamples}`;
    return `変動あり ${Math.min(status.consecutive, status.settings.requiredSamples)} / ${status.settings.requiredSamples}`;
  }

  function monitorStatusClass(status) {
    if (status?.satisfied === true) return 'ok';
    if (status?.legacy || status?.eligible === false || !status?.nativeRows.length) return 'neutral';
    return 'warning';
  }

  function monitorValue(value, unit = '') {
    return value === null || value === undefined || !Number.isFinite(Number(value)) ? '—' : `${fmtSci(value)}${unit ? ` ${unit}` : ''}`;
  }

  function monitorInspectorMarkup(run) {
    const status = monitorState(run);
    const settings = status.settings;
    const latest = status.latest;
    const values = MONITOR_COMPONENTS.map((series) => {
      const residual = latest ? nullableNumber(latest[series.key]) : null;
      const change = latest ? nullableNumber(latest[series.changeKey]) : null;
      return `<div class="monitor-value-card"><span>${series.label}の相対変化</span><strong>${esc(monitorValue(residual))}</strong><small>絶対変化 ${esc(monitorValue(change, series.unit))}</small></div>`;
    }).join('');
    const statusText = monitorStatusText(status);
    return `
      <div class="property-section monitor-result-section">
        <div class="section-title"><span>残差モニター</span><small>RUN MONITOR</small></div>
        <div class="monitor-status-line"><span class="monitor-status ${monitorStatusClass(status)}">${esc(statusText)}</span><span class="monitor-status-detail">${latest ? `step ${esc(fmt(latest.step, 0))}` : '記録待ち'}</span></div>
        <div class="monitor-value-grid">${values}</div>
        <div class="monitor-summary-grid"><div><span>許容値</span><strong>${esc(fmtSci(settings.tolerance))}</strong></div><div><span>必要な連続サンプル</span><strong>${esc(fmt(status.consecutive, 0))} / ${esc(fmt(settings.requiredSamples, 0))}</strong></div></div>
        <div class="info-note monitor-definition"><strong>定常性の目安（前回記録からの相対変化）</strong><br>速度・圧力・温度の有効な場に同じ閾値を適用します。保存済み run のモニター設定を表示しています。<br>過渡計算や LES の揺らぎは残ることがあります。自動停止は行いません。</div>
        <div class="button-row"><button type="button" class="button primary compact monitor-open-button" data-action="open-residual-monitor">残差モニターを開く</button></div>
      </div>`;
  }

  function runSupportsEddyViscosity(run) {
    const diagnostics = run?.diagnostics;
    if (!diagnostics || typeof diagnostics !== 'object') return false;
    const turbulenceModel = diagnostics.turbulence_model;
    if (typeof turbulenceModel === 'string' && turbulenceModel.trim()) return true;
    return Object.prototype.hasOwnProperty.call(diagnostics, 'eddy_viscosity_max_m2_s')
      && diagnostics.eddy_viscosity_max_m2_s !== null
      && diagnostics.eddy_viscosity_max_m2_s !== undefined;
  }

  function meshAdjustmentNotice(value) {
    const match = String(value).match(/^target total cell count (\d+) adjusted to nearest (?:admissible isotropic|permitted CAD) grid (\d+) \(\[([\d, ]+)\]\)$/);
    if (!match) return null;
    const shape = match[3].split(',').map(part => part.trim()).join(' × ');
    return `等方格子に合わせてセル数を調整しました。目標 ${Number(match[1]).toLocaleString()} → 実際 ${Number(match[2]).toLocaleString()} セル（${shape}）。正常な調整です。`;
  }

  function runWarnings(run) {
    if (!run) return [];
    const values = [];
    if (Array.isArray(run.warnings)) values.push(...run.warnings);
    if (Array.isArray(run.mesh_warnings)) values.push(...run.mesh_warnings);
    if (Array.isArray(run.diagnostics?.warnings)) values.push(...run.diagnostics.warnings);
    if (typeof run.diagnostics?.warning === 'string') values.push(run.diagnostics.warning);
    return values.map((value) => typeof value === 'string' ? value : JSON.stringify(value)).filter(value => value && !meshAdjustmentNotice(value));
  }

  function runPressureNote(run) {
    const diagnostics = run?.diagnostics;
    const kind = diagnostics?.pressure_kind ?? run?.pressure_kind;
    if (kind === 'reduced') return '圧力は基準静水圧を除いた値です。';
    if (kind === 'gauge') return '圧力は通常のゲージ圧です。';
    const gravity = run?.input?.physics?.gravity
      || run?.input_snapshot?.physics?.gravity
      || run?.project?.physics?.gravity
      || diagnostics?.gravity;
    if (gravity?.enabled === true && gravity.mode === 'buoyancy') return '圧力は基準静水圧を除いた値です。';
    if (gravity?.enabled === true) return '圧力は通常のゲージ圧です。';
    return '';
  }

  // The API includes a mask for every slice so a fluid-only run can keep the
  // response shape stable.  Treat it as CHT only when it contains at least
  // one solid cell; the presence of an all-false mask is not evidence of a
  // coupled solid domain.
  function maskContainsSolid(mask) {
    if (!Array.isArray(mask)) return false;
    const pending = mask.slice();
    while (pending.length) {
      const value = pending.pop();
      if (Array.isArray(value)) pending.push(...value);
      else if (value === true || Number(value) > 0) return true;
    }
    return false;
  }

  function runStatusLabel(status) {
    return ({ queued: 'キュー待ち', running: '計算中', stopping: '停止要求中', completed: '完了', stopped: '停止', failed: '失敗', interrupted: '中断', cancelled: 'キャンセル', canceled: 'キャンセル' }[status] || status || '不明');
  }

  function renderLogs() {
    const list = $('#logList');
    if (!list) return;
    list.innerHTML = state.logs.length ? state.logs.slice().reverse().map((entry) => `<div class="log-line ${esc(entry.level)}"><span class="log-time">${esc(entry.at)}</span><span class="log-level">${entry.level === 'error' ? 'ERROR' : entry.level === 'warn' ? 'WARN' : 'INFO'}</span><span class="log-message" title="${esc(entry.message)}">${esc(entry.message)}</span></div>`).join('') : '<div class="empty-state">ログはありません。</div>';
    $('#logCount').textContent = String(state.logs.length);
  }

  function validateProject(project = state.project) {
    const messages = [];
    if (!project) return messages;
    const geometry = project.geometry || {};
    const size = Array.isArray(geometry.size) ? geometry.size : [];
    const materialIds = new Set((project.materials || []).map((material) => material.id));
    if (!['box', 'cad'].includes(geometry.kind)) messages.push({ level: 'error', text: 'ジオメトリタイプが不正です。' });
    if (geometry.kind === 'box' && size.length === 3 && size.some((value) => !Number.isFinite(Number(value)) || Number(value) <= 0)) messages.push({ level: 'error', text: 'ボックスの各辺は 0 より大きい値にしてください。' });
    if (geometry.kind === 'cad' && !geometry.asset_id) messages.push({ level: 'error', text: 'CAD ジオメトリにアセットが指定されていません。' });
    if (geometry.solid_material_id && !materialIds.has(geometry.solid_material_id)) messages.push({ level: 'error', text: 'CAD 固体材料が材料一覧にありません。' });
    if (Array.isArray(geometry.solids)) geometry.solids.forEach((solid, index) => {
      if (geometry.kind !== 'box' || geometry.role !== 'fluid') messages.push({ level: 'error', text: `固体領域 ${index + 1} はボックス流体領域でのみ定義できます。` });
      if (!Array.isArray(solid.origin) || solid.origin.length !== 3 || solid.origin.some((value) => !Number.isFinite(Number(value)))) messages.push({ level: 'error', text: `固体領域 ${solid.name || index + 1} の原点を確認してください。` });
      if (!Array.isArray(solid.size) || solid.size.length !== 3 || solid.size.some((value) => !Number.isFinite(Number(value)) || Number(value) <= 0)) messages.push({ level: 'error', text: `固体領域 ${solid.name || index + 1} のサイズは正の値が必要です。` });
      if (!materialIds.has(solid.material_id)) messages.push({ level: 'error', text: `固体領域 ${solid.name || index + 1} の材料が材料一覧にありません。` });
    });
    if (!Array.isArray(project.materials) || !project.materials.length) messages.push({ level: 'error', text: '材料を 1 つ以上定義してください。' });
    (project.materials || []).forEach((material) => {
      ['density', 'viscosity', 'heat_capacity', 'conductivity'].forEach((key) => {
        const property = material[key];
        if (!property) messages.push({ level: 'error', text: `${material.name || material.id}: ${MATERIAL_LABELS[key]} がありません。` });
        else if (property.kind === 'constant' && (!Number.isFinite(Number(property.value)) || Number(property.value) <= 0)) messages.push({ level: 'error', text: `${material.name || material.id}: ${MATERIAL_LABELS[key]} は 0 より大きい数値が必要です。` });
        else if (property.kind === 'table') {
          if (!Array.isArray(property.points) || property.points.length < 2) messages.push({ level: 'error', text: `${material.name || material.id}: ${MATERIAL_LABELS[key]} テーブルには 2 点以上必要です。` });
          else if (property.points.some((point) => !Array.isArray(point) || point.length < 2 || !Number.isFinite(Number(point[0])) || Number(point[0]) <= 0 || !Number.isFinite(Number(point[1])) || Number(point[1]) <= 0)) messages.push({ level: 'error', text: `${material.name || material.id}: ${MATERIAL_LABELS[key]} テーブルは正の温度・物性値で入力してください。` });
          else if (property.points.some((point, index) => index > 0 && Number(point[0]) <= Number(property.points[index - 1][0]))) messages.push({ level: 'error', text: `${material.name || material.id}: ${MATERIAL_LABELS[key]} の温度点は昇順で入力してください。` });
        }
      });
    });
    if (project.physics.turbulence) {
      const turbulence = normalizeTurbulence(project.physics.turbulence);
      if (!['laminar', 'smagorinsky'].includes(turbulence.model)) messages.push({ level: 'error', text: '乱流モデルは層流または Smagorinsky LES を選択してください。' });
      if (turbulence.model === 'smagorinsky' && (turbulence.smagorinsky_constant <= 0 || turbulence.turbulent_prandtl <= 0)) messages.push({ level: 'error', text: 'Smagorinsky 定数と乱流 Prandtl 数は 0 より大きい値が必要です。' });
    }
    if (Object.prototype.hasOwnProperty.call(project.physics || {}, 'gravity')) {
      const gravity = project.physics.gravity;
      const allowedGravityKeys = ['enabled', 'mode', 'vector', 'reference_temperature'];
      if (!gravity || typeof gravity !== 'object' || Array.isArray(gravity)) {
        messages.push({ level: 'error', text: '重力設定はオブジェクトで指定してください。' });
      } else {
        const unknown = Object.keys(gravity).filter((key) => !allowedGravityKeys.includes(key));
        if (unknown.length) messages.push({ level: 'error', text: `重力設定に不明な項目があります: ${unknown.join(', ')}` });
        if (typeof gravity.enabled !== 'boolean') messages.push({ level: 'error', text: '重力の有効化は boolean で指定してください。' });
        if (!['uniform', 'buoyancy'].includes(gravity.mode)) messages.push({ level: 'error', text: '重力モデルは一定加速度または温度依存密度を選択してください。' });
        if (!Array.isArray(gravity.vector) || gravity.vector.length !== 3 || gravity.vector.some((value) => !Number.isFinite(Number(value)))) messages.push({ level: 'error', text: '重力加速度ベクトルは有限な 3 成分で指定してください。' });
        if (!Number.isFinite(Number(gravity.reference_temperature)) || Number(gravity.reference_temperature) <= 0) messages.push({ level: 'error', text: '浮力の基準温度は 0 より大きい有限値にしてください。' });
      }
    }
    const cells = project.mesh && Array.isArray(project.mesh.cells) ? project.mesh.cells : [];
    const autoMesh = project.mesh?.target_cells !== undefined;
    if (autoMesh && (!Number.isInteger(project.mesh.target_cells) || project.mesh.target_cells < 1 || (state.limits && project.mesh.target_cells > state.limits.max_total_cells))) messages.push({ level: 'error', text: '目標総セル数は 1 以上、総セル数の上限以下の整数にしてください。' });
    if (cells.length !== 3 || cells.some((value) => !Number.isInteger(Number(value)) || Number(value) < 1)) messages.push({ level: 'error', text: 'メッシュのセル数は各軸 1 以上の整数にしてください。' });
    if (!autoMesh && geometry.kind === 'box' && size.length === 3 && cells.length === 3 && cells.every((value) => Number(value) > 0)) {
      const spacing = size.map((value, index) => Number(value) / Number(cells[index]));
      const maxSpacing = Math.max(...spacing);
      if (maxSpacing > 0 && (Math.max(...spacing) - Math.min(...spacing)) / maxSpacing > 1e-5) messages.push({ level: 'error', text: 'ボックスのサイズとセル数から算出したセル幅が非等方です。各軸のセル幅を一致させてください。' });
    }
    const seenFaces = new Set();
    (project.boundaries || []).forEach((boundary) => {
      if (seenFaces.has(boundary.face)) messages.push({ level: 'error', text: `境界面 ${boundary.face} が重複しています。` });
      seenFaces.add(boundary.face);
      if (!isKnownFace(boundary.face)) messages.push({ level: 'error', text: `境界 ${boundary.name || boundary.id} の対象面が不正です。` });
      if (isCadFace(boundary.face) && geometry.kind !== 'cad') messages.push({ level: 'error', text: `CAD 面境界 ${boundary.name || boundary.id} は CAD ジオメトリでのみ使用できます。` });
      if (boundary.face === 'cad' && boundary.flow?.type !== 'wall') messages.push({ level: 'error', text: `CAD 全体の既定面境界 ${boundary.name || boundary.id} は壁（no-slip）を使用してください。` });
      if (boundary.flow?.type === 'velocity' && (!Array.isArray(boundary.flow.velocity) || boundary.flow.velocity.length !== 3 || boundary.flow.velocity.some((value) => !Number.isFinite(Number(value))))) messages.push({ level: 'error', text: `${boundary.name || '境界'} の速度ベクトルを確認してください。` });
      if (boundary.thermal?.type === 'convection' && (!Number.isFinite(Number(boundary.thermal.h)) || Number(boundary.thermal.h) < 0)) messages.push({ level: 'error', text: `${boundary.name || '境界'} の熱伝達係数 h を確認してください。` });
    });
    const study = project.study || {};
    const flowStability = state.meshEstimateKey === JSON.stringify(serializableProject(project)) ? state.meshEstimate?.flow_stability : null;
    if (project.physics.flow && flowStability && flowStability.boundary_mach >= flowStability.mach_limit) messages.push({ level: 'error', text: `境界速度に対して時間刻みが大きすぎます。Δt を ${Number(flowStability.dt_max_exclusive).toPrecision(6)} s 未満にしてください。` });
    if (project.physics.flow && flowStability && flowStability.force_dt_max !== null && flowStability.force_dt_max !== undefined && Number.isFinite(Number(flowStability.force_dt_max)) && Number(flowStability.force_dt_max) > 0 && Number(study.dt) > Number(flowStability.force_dt_max)) messages.push({ level: 'error', text: `重力加速度が大きすぎます。Δt を ${Number(flowStability.force_dt_max).toPrecision(6)} s 以下にしてください。` });
    if (!Number.isInteger(Number(study.steps)) || Number(study.steps) < 1) messages.push({ level: 'error', text: 'ステップ数は 1 以上の整数にしてください。' });
    if (!Number.isInteger(Number(study.output_interval)) || Number(study.output_interval) < 1 || Number(study.output_interval) > Number(study.steps)) messages.push({ level: 'error', text: '出力間隔は 1 以上、ステップ数以下にしてください。' });
    if (!Number.isInteger(Number(study.snapshot_interval)) || Number(study.snapshot_interval) < 0) messages.push({ level: 'error', text: '結果保存間隔は 0（無効）または 0 より大きい整数にしてください。' });
    if (hasOwn(study, 'monitor')) {
      const monitor = study.monitor;
      if (!monitor || typeof monitor !== 'object' || Array.isArray(monitor)) {
        messages.push({ level: 'error', text: '残差モニター設定はオブジェクトで指定してください。' });
      } else {
        const unknown = Object.keys(monitor).filter((key) => !['tolerance', 'consecutive_samples'].includes(key));
        if (unknown.length) messages.push({ level: 'error', text: `残差モニター設定に不明な項目があります: ${unknown.join(', ')}` });
        if (!Number.isFinite(Number(monitor.tolerance)) || Number(monitor.tolerance) <= 0) messages.push({ level: 'error', text: '残差モニターの許容値は 0 より大きい有限値にしてください。' });
        if (!Number.isInteger(Number(monitor.consecutive_samples)) || Number(monitor.consecutive_samples) < 1 || Number(monitor.consecutive_samples) > 100) messages.push({ level: 'error', text: '残差モニターの必要な連続サンプル数は 1〜100 の整数にしてください。' });
      }
    }
    if (!Number.isFinite(Number(study.dt)) || Number(study.dt) <= 0) messages.push({ level: 'error', text: '時間刻み Δt は 0 より大きい値にしてください。' });
    if (!['cpu', 'cuda:0', 'cuda:0-ram'].includes(String(study.device || ''))) messages.push({ level: 'error', text: '実行デバイスが不正です。' });
    if (!Number.isInteger(Number(study.gpu_batch_cells)) || Number(study.gpu_batch_cells) < 1024 || Number(study.gpu_batch_cells) > 1048576) messages.push({ level: 'error', text: 'GPU＋RAM のバッチセル数は 1,024 以上 1,048,576 以下の整数にしてください。' });
    if (!messages.length && geometry.kind === 'cad' && geometry.role === 'obstacle') messages.push({ level: 'warning', text: 'CAD 障害物の計算領域はサーバー側の CAD bounds 設定に依存します。' });
    return messages;
  }

  function updateValidationSummary() {
    const summary = $('#validationSummary');
    if (!summary) return;
    const messages = validateProject();
    state.validation = messages;
    const errors = messages.filter((message) => message.level === 'error');
    const warnings = messages.filter((message) => message.level === 'warning');
    summary.className = `validation-summary ${errors.length ? 'error' : warnings.length ? 'warning' : ''}`;
    summary.textContent = errors.length ? `${errors.length} 件のエラー` : warnings.length ? `${warnings.length} 件の注意` : '検証 OK';
    renderValidationDock();
  }

  function renderValidationDock() {
    const list = $('#validationList');
    if (!list) return;
    const messages = validateProject();
    state.validation = messages;
    if (!messages.length) {
      list.innerHTML = '<div class="validation-item ok"><span class="validation-icon">✓</span><span>現在のプロジェクト設定にエラーはありません。</span></div>';
      return;
    }
    list.innerHTML = messages.map((message) => `<div class="validation-item ${message.level}"><span class="validation-icon">${message.level === 'error' ? '!' : '△'}</span><span>${esc(message.text)}</span></div>`).join('');
  }

  function selectNode(node) {
    if (!node) return;
    state.selectedNode = node;
    if (node === 'geometry' && state.project?.geometry?.kind === 'cad') switchView('geometry');
    if (node.startsWith('face:')) {
      const face = node.slice(5);
      if (isCadFace(face)) {
        state.selectedSurface = face === 'cad' ? null : face.slice(4);
        switchView('geometry');
      } else if (FACES.includes(face)) {
        state.selectedSurface = face;
        switchView('geometry');
      }
    }
    renderTree();
    renderInspector();
    drawGeometry();
    if (node === 'results') {
      switchView('results');
      if (state.run?.id) {
        loadFrames({ silent: true, autoLoad: true }).then(() => {
          if (state.run?.id && !state.slice && (state.frames.length || ['completed', 'stopped'].includes(state.run.status))) loadSlice({ silent: true });
        });
      }
    }
  }

  function updateBoundValue(element) {
    const path = element.dataset.bind;
    if (!path || !state.project) return;
    let value;
    if (element.type === 'checkbox') value = element.checked;
    else if (element.type === 'number') value = numberOr(element.value, 0);
    else value = element.value;
    if (path.startsWith('physics.gravity.')) ensureGravity();
    setByPath(state.project, path, value);
    if (path === 'physics.material_id') {
      state.materialPresetSelectedId = '';
      state.materialPresetAppliedMaterialRef = null;
      state.materialPresetAppliedId = '';
      state.materialPresetAppliedSignature = '';
      state.materialPresetEdited = false;
      state.materialTablePages = {};
      invalidateMaterialCsvPreview();
    }
    if (path === 'name') updateHeader();
    markDirty();
    if (path.startsWith('geometry.') || path.startsWith('mesh.')) {
      state.mesh = null;
      state.slice = null; state.resultControls = null;
      if (state.selectedNode === 'mesh') scheduleMeshEstimate();
      renderTree();
      drawAll();
    }
    if (path.startsWith('physics.')) renderTree();
    if ((path === 'physics.flow' || path === 'physics.gravity.enabled' || path === 'physics.gravity.mode') && state.selectedNode === 'physics') renderInspector();
    if (path === 'physics.gravity.enabled' || path === 'physics.gravity.mode' || path.startsWith('physics.gravity.vector.') || path === 'physics.gravity.reference_temperature' || path === 'physics.material_id' || path === 'physics.initial_temperature') scheduleMeshEstimate();
    if (path === 'study.device' && state.selectedNode === 'study') renderInspector();
    if (path.startsWith('study.') && state.selectedNode === 'study') {
      scheduleMeshEstimate();
      const stability = $('#flowTimeStepInfo');
      if (stability) stability.innerHTML = flowTimeStepMarkup();
    }
  }

  function updateMaterialInput(element) {
    const material = currentMaterial();
    if (!material) return;
    if (element.dataset.materialName !== undefined) {
      material.name = element.value;
      markMaterialManuallyEdited(material);
      updateHeader();
      markDirty();
      renderTree();
      scheduleMeshEstimate();
      return;
    }
    const key = element.dataset.materialValue;
    if (key) {
      material[key] = { kind: 'constant', value: numberOr(element.value, 0) };
      markMaterialManuallyEdited(material);
      markDirty();
      scheduleMeshEstimate();
      return;
    }
    const tableKey = element.dataset.tableProperty;
    if (tableKey) {
      const index = Number(element.dataset.tableIndex);
      const column = Number(element.dataset.tableColumn);
      if (!material[tableKey] || material[tableKey].kind !== 'table') material[tableKey] = { kind: 'table', points: [] };
      if (!material[tableKey].points[index]) material[tableKey].points[index] = [273.15, 0];
      material[tableKey].points[index][column] = numberOr(element.value, 0);
      markMaterialManuallyEdited(material);
      markDirty();
      scheduleMeshEstimate();
    }
  }

  function updateBoundaryInput(element) {
    if (!state.selectedNode.startsWith('face:')) return;
    const face = state.selectedNode.slice(5);
    const boundary = boundaryForFace(face);
    if (!boundary) return;
    if (element.dataset.boundaryVector !== undefined) {
      const index = Number(element.dataset.boundaryVector);
      boundary.flow = boundary.flow || { type: 'velocity', velocity: [0, 0, 0] };
      boundary.flow.velocity = boundary.flow.velocity || [0, 0, 0];
      boundary.flow.velocity[index] = numberOr(element.value, 0);
    } else {
      const path = element.dataset.boundaryField;
      if (path) {
        let value = element.value;
        if (element.type === 'number') value = numberOr(value, 0);
        setByPath(boundary, path, value);
      }
    }
    // `cad` is the catch-all surface selector and the schema defines it as a
    // stationary wall.  Keep that invariant when a user moves an existing
    // boundary onto the fallback selector; named cad:<patch_id> selectors
    // remain fully editable below.
    if (boundary.face === 'cad') boundary.flow = { type: 'wall' };
    markDirty();
    if (element.dataset.boundaryField === 'flow.type' || element.dataset.boundaryField === 'thermal.type' || element.dataset.boundaryField === 'face') {
      Object.assign(boundary, normalizeBoundary(boundary));
      state.selectedNode = `face:${boundary.face}`;
      renderTree();
      renderInspector();
      drawAll();
    }
  }

  function handleInput(event) {
    const element = event.target;
    if (element.matches('[data-bind]')) updateBoundValue(element);
    if (element.matches('[data-material-name], [data-material-value], [data-table-property]')) updateMaterialInput(element);
    if (element.matches('[data-boundary-field], [data-boundary-vector]')) updateBoundaryInput(element);
    if (element.matches('[data-turbulence-field]')) updateTurbulenceInput(element);
    if (element.matches('[data-solid-field]')) updateSolidInput(element);
    if (element.matches('[data-restart-additional]')) updateRestartAdditionalInput(element);
    if (element.matches('[data-result-index]')) {
      const run = state.run;
      const shape = run?.shape || state.mesh?.shape || state.project.mesh.cells;
      const axis = $('[data-result-axis]')?.value || state.slice?.axis || 'z';
      const axisLength = shape[['x', 'y', 'z'].indexOf(axis)] || 1;
      const index = clamp(Math.round(numberOr(element.value, 0)), 0, Math.max(0, axisLength - 1));
      state.resultControls = { ...(state.resultControls || {}), field: $('[data-result-field]')?.value || 'temperature', axis, index, frameStep: state.selectedFrameStep };
    }
  }

  function handleChange(event) {
    const element = event.target;
    if (element.matches('[data-bind]')) {
      updateBoundValue(element);
      if (element.dataset.bind === 'geometry.role' || element.dataset.bind === 'geometry.solid_material_id') renderInspector();
      if (element.dataset.bind.startsWith('mesh.')) renderInspector();
    }
    if (element.matches('[data-material-mode], [data-material-value], [data-table-property]')) {
      if (element.matches('[data-material-mode]')) changeMaterialMode(element.dataset.materialMode, element.value);
      else updateMaterialInput(element);
    }
    if (element.matches('[data-boundary-field], [data-boundary-vector]')) updateBoundaryInput(element);
    if (element.matches('[data-turbulence-field]')) updateTurbulenceInput(element);
    if (element.matches('[data-solid-field]')) updateSolidInput(element);
    if (element.matches('[data-restart-additional]')) updateRestartAdditionalInput(element);
    if (element.matches('[data-action="select-material"]')) {
      state.project.physics.material_id = element.value;
      state.materialPresetSelectedId = '';
      state.materialPresetAppliedMaterialRef = null;
      state.materialPresetAppliedId = '';
      state.materialPresetAppliedSignature = '';
      state.materialPresetEdited = false;
      state.materialTablePages = {};
      invalidateMaterialCsvPreview();
      markDirty();
      renderTree();
      renderInspector();
    }
    if (element.matches('[data-material-preset]')) selectMaterialPreset(element.value);
    if (element.matches('[data-result-field], [data-result-axis], [data-result-index]')) {
      state.resultControls = {
        field: $('[data-result-field]')?.value || 'temperature',
        axis: $('[data-result-axis]')?.value || 'y',
        index: Number($('[data-result-index]')?.value || 0),
        frameStep: state.selectedFrameStep
      };
      refreshResultSlice();
    }
    if (element.matches('[data-result-frame]')) selectFrameStep(element.value);
    if (element.matches('[data-result-follow]')) toggleFollowLatest(element.checked);
    if (element.matches('[data-action="select-run"]') && element.value) selectRun(element.value);
    if (element.matches('[data-action="select-surface"]')) selectSurfaceGroup(element.value);
  }

  function updateTurbulenceInput(element) {
    if (!state.project) return;
    if (!state.project.physics.turbulence) state.project.physics.turbulence = normalizeTurbulence();
    const key = element.dataset.turbulenceField;
    state.project.physics.turbulence[key] = key === 'model' ? (element.value === 'smagorinsky' ? 'smagorinsky' : 'laminar') : numberOr(element.value, key === 'smagorinsky_constant' ? 0.17 : 0.9);
    markDirty();
    if (key === 'model') { renderTree(); renderInspector(); }
  }

  function updateSolidInput(element) {
    const index = Number(element.dataset.solidIndex);
    const key = element.dataset.solidField;
    const solid = state.project.geometry.solids?.[index];
    if (!solid || !key) return;
    if (element.dataset.solidVector !== undefined) {
      const component = Number(element.dataset.solidVector);
      solid[key][component] = numberOr(element.value, 0);
    } else if (key === 'material_id') solid[key] = element.value;
    else if (key === 'name') solid[key] = element.value;
    else solid[key] = numberOr(element.value, 0);
    // A solid's placement, size, or material changes the masked mesh and the
    // coupled field, so discard the preview/result until the next build/run.
    if (key !== 'name') {
      state.mesh = null;
      state.slice = null;
      state.resultControls = null;
    }
    markDirty();
    if (key !== 'name') drawGeometry();
  }

  function selectSurfaceGroup(groupId) {
    const selected = groupId || null;
    state.selectedSurface = selected;
    if (selected) switchView('geometry');
    drawGeometry();
    if (selected) pushLog(`${FACES.includes(selected) ? '境界面' : 'CAD 表面'}「${faceLabel(FACES.includes(selected) ? selected : `cad:${selected}`)}」を選択しました。`);
  }

  function addSolidRegion() {
    if (!Array.isArray(state.project.geometry.solids)) state.project.geometry.solids = [];
    const material = currentMaterial();
    const index = state.project.geometry.solids.length;
    state.project.geometry.solids.push(normalizeSolid({ id: uid('solid'), name: `Solid ${index + 1}`, origin: [0, 0, 0], size: [0.01, 0.01, 0.01], material_id: material?.id || 'water' }, index));
    state.mesh = null;
    state.slice = null;
    state.resultControls = null;
    markDirty();
    renderTree(); renderInspector(); drawGeometry();
    pushLog(`固体領域 ${index + 1} を追加しました。`);
  }

  function removeSolidRegion(index) {
    if (!Array.isArray(state.project.geometry.solids) || !state.project.geometry.solids[index]) return;
    const removed = state.project.geometry.solids.splice(index, 1)[0];
    state.mesh = null;
    state.slice = null;
    state.resultControls = null;
    markDirty();
    renderTree(); renderInspector(); drawGeometry();
    pushLog(`固体領域「${removed.name}」を削除しました。`);
  }

  function changeMaterialMode(key, mode) {
    const material = currentMaterial();
    if (!material || !MATERIAL_LABELS[key]) return;
    if (mode === 'table') {
      const current = material[key];
      const value = current?.kind === 'constant' ? numberOr(current.value, 0) : 0;
      material[key] = { kind: 'table', points: [[273.15, value], [373.15, value]] };
    } else {
      const current = material[key];
      const points = current?.points || [];
      const middle = points.length ? points[Math.floor(points.length / 2)][1] : 0;
      material[key] = { kind: 'constant', value: numberOr(middle, 0) };
    }
    markMaterialManuallyEdited(material);
    state.materialTablePages = {};
    markDirty();
    renderInspector();
  }

  function addMaterialRow(key) {
    const material = currentMaterial();
    if (!material) return;
    if (!material[key] || material[key].kind !== 'table') material[key] = { kind: 'table', points: [] };
    const points = material[key].points;
    const last = points[points.length - 1] || [293.15, 0];
    points.push([numberOr(last[0], 293.15) + 25, numberOr(last[1], 0)]);
    markMaterialManuallyEdited(material);
    state.materialTablePages = {};
    markDirty();
    renderInspector();
  }

  function removeMaterialRow(key, index) {
    const material = currentMaterial();
    if (!material || material[key]?.kind !== 'table') return;
    if (material[key].points.length <= 2) {
      toast('テーブルには最低 2 点が必要です。', 'warn');
      return;
    }
    material[key].points.splice(index, 1);
    markMaterialManuallyEdited(material);
    state.materialTablePages = {};
    markDirty();
    renderInspector();
  }

  function setGeometryKind(kind) {
    if (!state.project) return;
    state.project.geometry.kind = kind === 'cad' ? 'cad' : 'box';
    state.selectedSurface = null;
    state.mesh = null;
    state.slice = null; state.resultControls = null;
    markDirty();
    renderTree();
    renderInspector();
    drawAll();
  }

  function addBoundary() {
    const unused = FACES.find((face) => !boundaryForFace(face)) || 'cad';
    if (unused === 'cad' && boundaryForFace('cad')) {
      toast('すべての境界面が定義済みです。', 'warn');
      return;
    }
    const boundary = ensureBoundary(unused);
    boundary.name = FACE_INFO[unused]?.label || 'CAD 表面';
    state.selectedNode = `face:${unused}`;
    renderTree();
    renderInspector();
    toast(`${faceLabel(unused) || '境界'} を追加しました。`);
  }

  function enableBoundary(face) {
    ensureBoundary(face);
    state.selectedNode = `face:${face}`;
    renderTree();
    renderInspector();
    pushLog(`${faceLabel(face)} のカスタム境界を有効化しました。`);
  }

  function removeBoundary(face) {
    const boundary = boundaryForFace(face);
    if (!boundary) return;
    state.project.boundaries = state.project.boundaries.filter((item) => item !== boundary);
    state.selectedNode = `face:${face}`;
    markDirty();
    renderTree();
    renderInspector();
    toast(`${faceLabel(face)} を既定条件へ戻しました。`);
  }

  function switchView(view) {
    state.view = view === 'results' ? 'results' : 'geometry';
    $('#geometryView')?.classList.toggle('hidden', state.view !== 'geometry');
    $('#resultsView')?.classList.toggle('hidden', state.view !== 'results');
    $$('.view-tab').forEach((button) => {
      const active = button.dataset.view === state.view;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    if (state.view === 'results') {
      drawResult();
      drawHistoryCanvas();
    }
  }

  function switchDock(dock) {
    state.activeDock = dock;
    $('.workspace')?.classList.toggle('monitor-expanded', dock === 'history');
    $$('.dock-tab').forEach((button) => {
      const active = button.dataset.dock === dock;
      button.classList.toggle('active', active);
      button.setAttribute('aria-selected', String(active));
    });
    ['log', 'history', 'validation'].forEach((name) => {
      const panel = $(`#${name}Dock`);
      if (panel) panel.hidden = name !== dock;
    });
    if (dock === 'history') renderHistory();
    if (dock === 'validation') renderValidationDock();
  }

  function handleClick(event) {
    const element = event.target.closest('[data-action], [data-view], [data-dock]');
    if (!element) return;
    const action = element.dataset.action;
    if (element.dataset.view) switchView(element.dataset.view);
    if (element.dataset.dock) switchDock(element.dataset.dock);
    if (!action) return;
    if (action === 'select-node') selectNode(element.dataset.node);
    if (action === 'new-project') newProject();
    if (action === 'open-project') openProjectDialog();
    if (action === 'close-project-dialog') closeDialog('projectDialog');
    if (action === 'save-project') saveProject();
    if (action === 'export-project') exportProject();
    if (action === 'import-project') $('#projectFile')?.click();
    if (action === 'select-geometry') selectNode('geometry');
    if (action === 'set-geometry-kind') setGeometryKind(element.dataset.kind);
    if (action === 'import-cad') openCadDialog();
    if (action === 'choose-cad-file') $('#cadFile')?.click();
    if (action === 'close-cad-dialog') closeDialog('cadDialog');
    if (action === 'confirm-cad-import') importCad();
    if (action === 'build-mesh') buildMesh();
    if (action === 'run-simulation') startRun();
    if (action === 'restart-simulation') restartRun();
    if (action === 'stop-simulation') stopRun();
    if (action === 'select-study') selectNode('study');
    if (action === 'open-residual-monitor') {
      switchDock('history');
      $('#historyDock')?.scrollIntoView({ block: 'nearest' });
    }
    if (action === 'fit-view') fitView();
    if (action === 'reset-view') resetView();
    if (action === 'toggle-grid') toggleGrid(element);
    if (action === 'toggle-help') toggleHelp();
    if (action === 'add-boundary') addBoundary();
    if (action === 'enable-boundary') enableBoundary(element.dataset.face);
    if (action === 'remove-boundary') removeBoundary(element.dataset.face);
    if (action === 'add-material-row') addMaterialRow(element.dataset.property);
    if (action === 'remove-material-row') removeMaterialRow(element.dataset.property, Number(element.dataset.index));
    if (action === 'material-table-page') changeMaterialTablePage(element.dataset.property, element.dataset.pageDirection);
    if (action === 'apply-material-preset') applyMaterialPreset(false);
    if (action === 'add-material-preset') applyMaterialPreset(true);
    if (action === 'import-material-csv') startMaterialCsvImport();
    if (action === 'import-material-property-csv') startMaterialCsvImport(element.dataset.property);
    if (action === 'close-material-csv') { invalidateMaterialCsvPreview(); }
    if (action === 'apply-material-csv') applyMaterialCsv();
    if (action === 'add-solid') addSolidRegion();
    if (action === 'remove-solid') removeSolidRegion(Number(element.dataset.solidIndex));
    if (action === 'validate-project') validateAndReport();
    if (action === 'choose-project') openProject(element.dataset.projectId);
    if (action === 'load-slice') loadSlice();
    if (action === 'toggle-result-playback') toggleResultPlayback();
    if (action === 'download-history') downloadRunFile('history.csv');
    if (action === 'download-vtk') downloadRunFile('fields.vtk');
    if (action === 'download-npz') downloadRunFile('fields.npz');
    if (action === 'download-input') downloadRunFile('input.json');
  }

  function newProject() {
    if (state.dirty && !window.confirm('未保存の変更は破棄されます。新規プロジェクトを作成しますか？')) return;
    setProject(DEFAULT_PROJECT);
    state.project.mesh.target_cells = state.project.mesh.cells.reduce((a, b) => a * b, 1);
    state.dirty = true;
    renderTree();
    updateHeader();
    pushLog('新規プロジェクトを作成しました。');
    toast('新規プロジェクトを作成しました。');
  }

  async function saveProject(options = {}) {
    if (!state.project) return null;
    const messages = validateProject();
    const errors = messages.filter((message) => message.level === 'error');
    if (errors.length) {
      renderValidationDock();
      switchDock('validation');
      pushLog(`保存できません: ${errors[0].text}`, 'error');
      toast('入力内容を確認してください。', 'error');
      return null;
    }
    try {
      // Keep UI-only metadata (for example the fetched CAD manifest) out of
      // the strict project schema.  The same canonical payload is used for
      // both create and update so turbulence, CHT solids, and per-face BCs
      // round-trip without leaking view state into project.json.
      const payload = serializableProject();
      const result = state.projectId
        ? await request(`/api/projects/${encodeURIComponent(state.projectId)}`, { method: 'PUT', body: payload })
        : await request('/api/projects', { method: 'POST', body: payload });
      state.projectId = result.id || state.projectId;
      state.project = normalizeProject(result.project || payload);
      state.dirty = false;
      updateHeader();
      renderTree();
      renderInspector();
      await refreshProjects().catch(() => {});
      if (!options.silent) {
        pushLog(`プロジェクトを保存しました${state.projectId ? ` (${state.projectId})` : ''}。`);
        toast('プロジェクトを保存しました。');
      }
      return state.projectId;
    } catch (error) {
      pushLog(`保存に失敗しました: ${error.message}`, 'error');
      toast(`保存に失敗しました: ${error.message}`, 'error');
      return null;
    }
  }

  async function openProjectDialog() {
    const dialog = $('#projectDialog');
    if (!dialog) return;
    try {
      await refreshProjects();
    } catch (error) {
      pushLog(`プロジェクト一覧を取得できません: ${error.message}`, 'error');
      toast(`一覧を取得できません: ${error.message}`, 'error');
    }
    renderProjectList();
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  function renderProjectList() {
    const list = $('#projectList');
    if (!list) return;
    if (!state.projects.length) {
      list.innerHTML = '<div class="empty-state">保存済みプロジェクトはありません。</div>';
      return;
    }
    list.innerHTML = state.projects.map((project) => {
      const id = project.id || project.project_id || '';
      const updated = project.updated_at ? new Date(project.updated_at).toLocaleString('ja-JP') : '更新日時不明';
      return `<button type="button" class="project-row" data-action="choose-project" data-project-id="${esc(id)}"><span class="project-row-icon">∿</span><span class="project-row-copy"><strong>${esc(project.name || '無題のモデル')}</strong><span>${esc(updated)} · ${esc(id)}</span></span><span class="project-row-arrow">›</span></button>`;
    }).join('');
  }

  function closeDialog(id) {
    const dialog = $(`#${id}`);
    if (!dialog) return;
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  }

  async function openProject(id) {
    if (!id) return;
    if (state.dirty && !window.confirm('未保存の変更は破棄されます。プロジェクトを開きますか？')) return;
    try {
      const payload = await request(`/api/projects/${encodeURIComponent(id)}`);
      setProject(payload.project || payload, payload.id || id, payload.runs || []);
      closeDialog('projectDialog');
      pushLog(`プロジェクト「${state.project.name}」を開きました。`);
      toast('プロジェクトを開きました。');
      if (state.runs.length) {
        state.run = normalizeRun(state.runs[0]);
        state.history = normalizeHistory(state.run.history || []);
        state.followLatest = ['queued', 'running', 'stopping'].includes(state.run.status);
        if (!state.history.length && !state.followLatest) await loadHistory(state.run.id);
        renderHistory();
        renderTree();
        renderInspector();
        if (state.followLatest) pollRun();
      }
    } catch (error) {
      pushLog(`プロジェクトを開けません: ${error.message}`, 'error');
      toast(`プロジェクトを開けません: ${error.message}`, 'error');
    }
  }

  async function exportProject() {
    let id = state.projectId;
    if (!id) id = await saveProject({ silent: true });
    if (!id) return;
    try {
      const response = await fetch(`/api/projects/${encodeURIComponent(id)}/export`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = `${(state.project.name || 'xlb-project').replace(/[^\w\-\u3040-\u30ff\u3400-\u9fff]+/g, '_')}.zip`;
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      URL.revokeObjectURL(url);
      pushLog('プロジェクトを ZIP に書き出しました。');
      toast('ZIP を書き出しました。');
    } catch (error) {
      pushLog(`書き出しに失敗しました: ${error.message}`, 'error');
      toast(`書き出しに失敗しました: ${error.message}`, 'error');
    }
  }

  async function importProjectFile(file) {
    if (!file) return;
    try {
      const result = await request('/api/projects/import', { method: 'POST', headers: { 'Content-Type': 'application/zip' }, body: file });
      setProject(result.project || result, result.id || null);
      pushLog(`プロジェクト ZIP を読み込みました${result.id ? ` (${result.id})` : ''}。`);
      toast('プロジェクトを読み込みました。');
      await refreshProjects().catch(() => {});
    } catch (error) {
      pushLog(`ZIP 読み込みに失敗しました: ${error.message}`, 'error');
      toast(`ZIP 読み込みに失敗しました: ${error.message}`, 'error');
    }
  }

  function openCadDialog() {
    const dialog = $('#cadDialog');
    if (!dialog) return;
    state.cadFile = null;
    $('#cadFile').value = '';
    $('#cadFileName').textContent = '';
    $('#cadFileName').classList.add('hidden');
    $('#cadImportStatus').textContent = '';
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  async function importCad() {
    const file = state.cadFile || $('#cadFile')?.files?.[0];
    if (!file) {
      toast('CAD ファイルを選択してください。', 'warn');
      return;
    }
    const unit = $('#cadUnit')?.value || 'mm';
    const status = $('#cadImportStatus');
    status.textContent = 'アップロード中…';
    try {
      const metadata = await request(`/api/import?filename=${encodeURIComponent(file.name)}&unit=${encodeURIComponent(unit)}`, { method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: file });
      const asset = metadata.asset || metadata;
      state.project.geometry.kind = 'cad';
      state.project.geometry.asset_id = asset.id || asset.asset_id || null;
      state.project.geometry.role = 'obstacle';
      state.project.geometry.size = Array.isArray(asset.size) ? asset.size : state.project.geometry.size;
      state.cadAssetMeta = asset;
      state.mesh = null;
      state.slice = null; state.resultControls = null;
      markDirty();
      closeDialog('cadDialog');
      renderTree(); renderInspector(); drawAll();
      pushLog(`CAD「${file.name}」を読み込みました。アセット ID: ${state.project.geometry.asset_id || '—'}`);
      toast('CAD を読み込みました。メッシュを生成してください。');
    } catch (error) {
      status.textContent = error.message;
      pushLog(`CAD の読み込みに失敗しました: ${error.message}`, 'error');
      toast(`CAD の読み込みに失敗しました: ${error.message}`, 'error');
    }
  }

  function serializableProject(project = state.project) {
    if (!project) return null;
    const geometry = {
      kind: project.geometry.kind,
      size: clone(project.geometry.size),
      asset_id: typeof project.geometry.asset_id === 'string' ? project.geometry.asset_id : null,
      role: project.geometry.role
    };
    // Preserve optional CAD computational-domain descriptors from imported
    // projects even though the compact editor does not expose them yet.
    ['origin', 'computational_box', 'domain', 'domain_size', 'domain_origin', 'box_size', 'box_origin'].forEach((key) => {
      if (project.geometry[key] !== undefined) geometry[key] = clone(project.geometry[key]);
    });
    if (project.geometry.solid_material_id) geometry.solid_material_id = String(project.geometry.solid_material_id);
    if (Array.isArray(project.geometry.solids)) geometry.solids = project.geometry.solids.map(serializableSolid);
    const physics = {
      flow: project.physics.flow === true,
      thermal: project.physics.thermal === true,
      material_id: project.physics.material_id,
      initial_temperature: numberOr(project.physics.initial_temperature, 293.15)
    };
    if (project.physics.turbulence) {
      const turbulence = normalizeTurbulence(project.physics.turbulence);
      physics.turbulence = {
        model: turbulence.model,
        smagorinsky_constant: turbulence.smagorinsky_constant,
        turbulent_prandtl: turbulence.turbulent_prandtl
      };
    }
    if (Object.prototype.hasOwnProperty.call(project.physics, 'gravity')) {
      const gravity = normalizeGravity(project.physics.gravity);
      physics.gravity = {
        enabled: gravity.enabled === true,
        mode: gravity.mode === 'buoyancy' ? 'buoyancy' : 'uniform',
        vector: gravity.vector.map((value, index) => numberOr(value, DEFAULT_GRAVITY.vector[index])),
        reference_temperature: numberOr(gravity.reference_temperature, DEFAULT_GRAVITY.reference_temperature)
      };
    }
    return {
      schema_version: 1,
      name: String(project.name || '無題のモデル'),
      geometry,
      materials: project.materials.map((material) => ({
        id: material.id,
        name: material.name,
        density: serializableProperty(material.density),
        viscosity: serializableProperty(material.viscosity),
        heat_capacity: serializableProperty(material.heat_capacity),
        conductivity: serializableProperty(material.conductivity)
      })),
      physics,
      boundaries: project.boundaries.map((boundary) => serializableBoundary(boundary)),
      mesh: project.mesh.target_cells !== undefined ? { target_cells: project.mesh.target_cells } : { cells: project.mesh.cells.map((value) => intOr(value, 1)) },
      study: {
        steps: intOr(project.study.steps, 1),
        output_interval: intOr(project.study.output_interval, 1),
        snapshot_interval: Math.max(0, Math.round(numberOr(project.study.snapshot_interval, 0))),
        dt: numberOr(project.study.dt, 0.001),
        device: ['cpu', 'cuda:0', 'cuda:0-ram'].includes(String(project.study.device || '')) ? String(project.study.device) : 'cpu',
        gpu_batch_cells: Math.round(numberOr(project.study.gpu_batch_cells, 65536)),
        ...(hasOwn(project.study, 'monitor') ? { monitor: (() => {
          const monitor = normalizeMonitor(project.study.monitor);
          return { tolerance: numberOr(monitor.tolerance, DEFAULT_MONITOR.tolerance), consecutive_samples: Math.round(numberOr(monitor.consecutive_samples, DEFAULT_MONITOR.consecutive_samples)) };
        })() } : {})
      }
    };
  }

  function serializableProperty(property) {
    if (property?.kind === 'table') return { kind: 'table', points: (property.points || []).map((point) => [numberOr(point[0], 0), numberOr(point[1], 0)]) };
    return { kind: 'constant', value: numberOr(property?.value, 0) };
  }

  function serializableSolid(solid) {
    return {
      id: String(solid.id), name: String(solid.name), origin: (solid.origin || [0, 0, 0]).map((value) => numberOr(value, 0)), size: (solid.size || [0.01, 0.01, 0.01]).map((value) => numberOr(value, 0.01)), material_id: String(solid.material_id)
    };
  }

  function serializableBoundary(boundary) {
    const result = { id: boundary.id, name: boundary.name, face: boundary.face, flow: { type: boundary.flow?.type || 'wall' }, thermal: { type: boundary.thermal?.type || 'adiabatic' } };
    if (result.flow.type === 'velocity') result.flow.velocity = (boundary.flow.velocity || [0, 0, 0]).map((value) => numberOr(value, 0));
    if (result.flow.type === 'pressure') result.flow.value = numberOr(boundary.flow.value, 0);
    if (result.thermal.type === 'temperature' || result.thermal.type === 'heat_flux') result.thermal.value = numberOr(boundary.thermal.value, 0);
    if (result.thermal.type === 'convection') {
      result.thermal.h = numberOr(boundary.thermal.h, 0);
      result.thermal.ambient_temperature = numberOr(boundary.thermal.ambient_temperature, 293.15);
    }
    return result;
  }

  async function buildMesh() {
    if (!state.project) return;
    const messages = validateProject();
    const errors = messages.filter((message) => message.level === 'error');
    if (errors.length) {
      switchDock('validation');
      pushLog(`メッシュを生成できません: ${errors[0].text}`, 'error');
      toast('メッシュ条件を確認してください。', 'error');
      return;
    }
    const buttons = $$('[data-action="build-mesh"]');
    buttons.forEach((button) => { button.disabled = true; button.dataset.originalText = button.textContent; button.textContent = '生成中…'; });
    try {
      const mesh = await request('/api/mesh', { method: 'POST', body: serializableProject() });
      state.mesh = mesh;
      state.slice = null; state.resultControls = null;
      const warningList = Array.isArray(mesh.warnings) ? mesh.warnings : [];
      const actualWarnings = warningList.filter(warning => !meshAdjustmentNotice(warning));
      pushLog(`メッシュを生成しました: ${mesh.shape ? mesh.shape.join(' × ') : 'shape —'} / fluid ${fmt(mesh.fluid_cells, 0)} cells${actualWarnings.length ? ` / 注意 ${actualWarnings.length} 件` : ''}.`);
      warningList.forEach((warning) => {
        const notice = meshAdjustmentNotice(warning);
        pushLog(notice || String(warning), notice ? 'info' : 'warn');
      });
      renderTree(); renderInspector(); drawAll();
      toast('メッシュプレビューを更新しました。');
    } catch (error) {
      pushLog(`メッシュ生成に失敗しました: ${error.message}`, 'error');
      toast(`メッシュ生成に失敗しました: ${error.message}`, 'error');
    } finally {
      buttons.forEach((button) => { button.disabled = false; button.textContent = button.dataset.originalText || '▦ メッシュ生成'; });
    }
  }

  function normalizeRun(payload) {
    const source = payload && typeof payload === 'object' ? payload : {};
    const progressSample = source.progress && typeof source.progress === 'object' ? source.progress : null;
    const incomingId = source.id || source.run_id;
    const previous = !incomingId || incomingId === state.run?.id ? state.run : null;
    const run = { ...(previous || {}), ...source };
    // Some API versions expose the latest progress sample under a nested
    // `progress` object.  Promote monitor fields so the inspector can render
    // the same live values regardless of response shape.
    [
      'time', 'sample_steps', 'sample_time', 'residual', 'residual_kind',
      'velocity_residual', 'pressure_residual', 'temperature_residual',
      'velocity_change_max', 'pressure_change_max', 'temperature_change_max',
      'monitor_tolerance', 'monitor_required_samples', 'monitor_consecutive_samples',
      'monitor_satisfied', 'monitor_eligible'
    ].forEach((key) => {
      if (run[key] === undefined && progressSample && progressSample[key] !== undefined) run[key] = progressSample[key];
    });
    run.id = run.id || run.run_id;
    let rawProgress = run.progress;
    if (rawProgress && typeof rawProgress === 'object') rawProgress = rawProgress.progress ?? rawProgress.fraction;
    if (rawProgress === undefined || rawProgress === null) rawProgress = run.progress_fraction;
    run.progress = clamp(numberOr(rawProgress, run.status === 'completed' ? 1 : 0), 0, 1);
    if (run.status === 'done' || run.status === 'success') run.status = 'completed';
    if (run.status === 'complete') run.status = 'completed';
    run.status = run.status || 'queued';
    run.step = run.step ?? run.current_step;
    run.residual = run.residual ?? run.last_residual;
    run.temperature_min = run.temperature_min ?? run.temp_min;
    run.temperature_max = run.temperature_max ?? run.temp_max;
    const restartFrom = run.restart_from ?? run.restartFrom;
    if (restartFrom !== undefined && restartFrom !== null && restartFrom !== '') run.restart_from = String(restartFrom);
    if (Array.isArray(run.history)) run.history = normalizeHistory(run.history);
    ['start_step', 'end_step', 'additional_steps'].forEach((key) => {
      const value = Number(run[key]);
      if (Number.isFinite(value)) run[key] = Math.max(0, Math.round(value));
    });
    return run;
  }

  async function restartRun() {
    const sourceRun = state.run;
    if (!sourceRun?.id || restartRunIsActive(sourceRun) || state.restartSubmitting) return;
    const key = restartCacheKey(sourceRun);
    const availability = key && key === state.restartAvailabilityKey ? state.restartAvailability : null;
    if (!availability?.available) return;
    const rawSteps = $('[data-restart-additional]')?.value ?? state.restartAdditionalSteps;
    const additionalSteps = Number(rawSteps);
    if (!Number.isInteger(additionalSteps) || additionalSteps <= 0) {
      state.restartAdditionalSteps = Number.isFinite(additionalSteps) ? additionalSteps : 0;
      pushLog('追加ステップ数は 1 以上の整数にしてください。', 'error');
      toast('追加ステップ数は 1 以上の整数にしてください。', 'error');
      updateRestartAdditionalInput({ value: rawSteps });
      return;
    }
    state.restartAdditionalSteps = additionalSteps;
    state.restartNotice = '元の実行条件（メッシュ・Δt・物性・境界）を再利用します。現在の編集内容は追加計算には使用されません。';
    state.restartSubmitting = true;
    const submitId = state.restartSubmitRequest = (state.restartSubmitRequest || 0) + 1;
    const sourceRunId = String(sourceRun.id);
    renderInspector();
    try {
      const response = await request(`/api/runs/${encodeURIComponent(sourceRunId)}/restart`, {
        method: 'POST',
        body: { additional_steps: additionalSteps }
      });
      if (state.restartSubmitRequest !== submitId || String(state.run?.id || '') !== sourceRunId) return;
      // normalizeRun merges partial polling responses with the selected run;
      // clear that context here so the new run cannot inherit the old id or
      // restart lineage.  Keep its shape as a display fallback only when the
      // standard status payload omits it.
      state.run = null;
      const nextRun = normalizeRun(response);
      if (!nextRun.id) {
        state.run = sourceRun;
        throw new Error('再開実行の ID が応答にありません。');
      }
      if (!nextRun.shape && sourceRun.shape) nextRun.shape = sourceRun.shape;
      state.run = nextRun;
      upsertRun(state.run);
      stopPlayback();
      window.clearTimeout(state.runTimer);
      state.framesRequest += 1;
      state.sliceRequest += 1;
      state.framesLoading = false;
      state.sliceLoading = false;
      state.slice = null;
      state.resultControls = null;
      state.frames = [];
      state.resultFields = [];
      state.selectedFrameStep = null;
      state.followLatest = restartRunIsActive(state.run);
      state.framesEndpointUnavailable = false;
      state.history = [];
      state.resultLineage = {
        sourceRunId,
        startStep: state.run.start_step,
        endStep: state.run.end_step,
        additionalSteps: state.run.additional_steps ?? additionalSteps
      };
      state.restartAvailabilityRequest += 1;
      state.restartAvailability = null;
      state.restartAvailabilityKey = null;
      state.restartAvailabilityLoading = false;
      state.restartSubmitting = false;
      state.selectedNode = 'results';
      switchView('results');
      setRunState(runStatusLabel(state.run.status), restartRunIsActive(state.run));
      updateRunOverlay();
      renderTree(); renderInspector(); renderHistory();
      pushLog(`追加計算を開始しました。run_id: ${state.run.id || '—'}（元 run: ${sourceRunId}）`);
      switchDock('history');
      toast('追加計算を開始しました。');
      if (restartRunIsActive(state.run)) pollRun();
      else await handleTerminalRun();
    } catch (error) {
      if (state.restartSubmitRequest !== submitId || String(state.run?.id || '') !== sourceRunId) return;
      state.restartSubmitting = false;
      renderInspector();
      pushLog(`追加計算を開始できません: ${error.message}`, 'error');
      toast(`追加計算を開始できません: ${error.message}`, 'error');
    } finally {
      // Selection or project changes can invalidate the request while the
      // POST is in flight.  Never leave the restart controls stuck disabled.
      if (state.restartSubmitRequest === submitId && state.restartSubmitting) {
        state.restartSubmitting = false;
        renderInspector();
      }
    }
  }

  async function startRun() {
    if (!state.project || ['queued', 'running', 'stopping'].includes(state.run?.status)) return;
    const messages = validateProject();
    const errors = messages.filter((message) => message.level === 'error');
    if (errors.length) {
      switchDock('validation');
      pushLog(`計算を開始できません: ${errors[0].text}`, 'error');
      toast('計算条件を確認してください。', 'error');
      return;
    }
    const savedId = await saveProject({ silent: true });
    if (!savedId) return;
    try {
      const response = await request('/api/runs', { method: 'POST', body: { project_id: savedId, project: serializableProject() } });
      state.run = normalizeRun(response);
      upsertRun(state.run);
      stopPlayback();
      state.slice = null; state.resultControls = null;
      state.frames = [];
      state.resultFields = [];
      state.selectedFrameStep = null;
      state.followLatest = ['queued', 'running', 'stopping'].includes(state.run.status);
      state.framesEndpointUnavailable = false;
      state.history = [];
      state.restartAvailabilityRequest += 1;
      state.restartSubmitRequest += 1;
      state.restartAvailability = null;
      state.restartAvailabilityKey = null;
      state.restartAvailabilityLoading = false;
      state.restartSubmitting = false;
      state.restartAdditionalSteps = 200;
      state.restartNotice = '';
      state.resultLineage = null;
      state.selectedNode = 'results';
      switchView('results');
      setRunState(runStatusLabel(state.run.status), ['queued', 'running', 'stopping'].includes(state.run.status));
      updateRunOverlay();
      renderTree(); renderInspector(); renderHistory();
      pushLog(`計算を開始しました。run_id: ${state.run.id || '—'}`);
      switchDock('history');
      toast('計算を開始しました。');
      if (['queued', 'running', 'stopping'].includes(state.run.status)) pollRun();
      else await handleTerminalRun();
    } catch (error) {
      pushLog(`計算を開始できません: ${error.message}`, 'error');
      setRunState('Ready', false);
      toast(`計算を開始できません: ${error.message}`, 'error');
    }
  }

  async function stopRun() {
    if (!state.run?.id || !['queued', 'running'].includes(state.run.status)) return;
    try {
      await request(`/api/runs/${encodeURIComponent(state.run.id)}/stop`, { method: 'POST' });
      pushLog('停止要求を送信しました。');
      setRunState('停止要求中', true);
      window.clearTimeout(state.runTimer);
      state.runTimer = window.setTimeout(pollRun, 250);
    } catch (error) {
      pushLog(`停止要求に失敗しました: ${error.message}`, 'error');
      toast(`停止要求に失敗しました: ${error.message}`, 'error');
    }
  }

  function progressHistorySample(payload) {
    const source = payload && typeof payload === 'object' ? payload : {};
    const nested = source.progress && typeof source.progress === 'object' ? source.progress : null;
    const sample = nested ? { ...source, ...nested } : source;
    const step = nullableNumber(sample.step ?? sample.current_step ?? sample.iteration);
    if (step === null) return null;
    const monitorKeys = ['residual_kind', 'velocity_residual', 'pressure_residual', 'temperature_residual', 'velocity_change_max', 'pressure_change_max', 'temperature_change_max', 'monitor_tolerance', 'monitor_required_samples', 'monitor_consecutive_samples', 'monitor_satisfied', 'monitor_eligible'];
    if (!monitorKeys.some((key) => hasOwn(sample, key))) return null;
    return normalizeHistory([{ ...sample, step }])[0] || null;
  }

  function mergeHistorySample(sample) {
    if (!sample) return;
    const existingIndex = state.history.findIndex((row) => row.step === sample.step);
    if (existingIndex >= 0) state.history[existingIndex] = { ...state.history[existingIndex], ...sample };
    else state.history.push(sample);
    state.history.sort((first, second) => first.step - second.step);
    if (state.history.length > 2000) state.history = state.history.slice(-2000);
  }

  async function pollRun() {
    if (!state.run?.id) return;
    window.clearTimeout(state.runTimer);
    try {
      const payload = await request(`/api/runs/${encodeURIComponent(state.run.id)}`);
      state.run = normalizeRun(payload);
      upsertRun(state.run);
      if (Array.isArray(payload.history)) state.history = normalizeHistory(payload.history);
      else mergeHistorySample(progressHistorySample(payload));
      await loadFrames({ silent: true, autoLoad: true });
      updateRunOverlay();
      renderTree(); renderInspector(); renderHistory();
      if (['queued', 'running', 'stopping'].includes(state.run.status)) {
        setRunState(runStatusLabel(state.run.status), true);
        state.runTimer = window.setTimeout(pollRun, 850);
      } else {
        await handleTerminalRun();
      }
    } catch (error) {
      pushLog(`計算状態の取得に失敗しました: ${error.message}`, 'error');
      state.runTimer = window.setTimeout(pollRun, 1800);
    }
  }

  function upsertRun(run) {
    if (!run?.id) return;
    const index = state.runs.findIndex((item) => item.id === run.id);
    if (index >= 0) state.runs[index] = { ...state.runs[index], ...run };
    else state.runs.unshift(run);
  }

  async function selectRun(id) {
    if (!id) return;
    window.clearTimeout(state.runTimer);
    state.restartSubmitRequest += 1;
    state.restartAvailabilityRequest += 1;
    state.restartSubmitting = false;
    state.restartAvailability = null;
    state.restartAvailabilityKey = null;
    state.restartAvailabilityLoading = false;
    state.restartAdditionalSteps = 200;
    state.restartNotice = '';
    state.resultLineage = null;
    try {
      const payload = await request(`/api/runs/${encodeURIComponent(id)}`);
      state.run = normalizeRun(payload);
      upsertRun(state.run);
      stopPlayback();
      state.slice = null; state.resultControls = null;
      state.frames = [];
      state.resultFields = [];
      state.selectedFrameStep = null;
      state.followLatest = ['queued', 'running', 'stopping'].includes(state.run.status);
      state.framesEndpointUnavailable = false;
      state.history = Array.isArray(payload.history) ? normalizeHistory(payload.history) : [];
      state.restartAvailability = null;
      state.restartAvailabilityKey = null;
      state.restartAvailabilityLoading = false;
      state.restartAdditionalSteps = 200;
      state.restartNotice = '';
      state.resultLineage = null;
      state.selectedNode = 'results';
      switchView('results');
      renderTree(); renderInspector();
      if (['queued', 'running', 'stopping'].includes(state.run.status)) {
        setRunState(runStatusLabel(state.run.status), true);
        await loadFrames({ silent: true, autoLoad: true });
        pollRun();
      } else {
        if (!state.history.length) await loadHistory(state.run.id);
        await loadFrames({ silent: true });
        await loadSlice({ silent: true });
        renderHistory(); renderInspector();
      }
      pushLog(`実行 ${String(id).slice(0, 12)} の結果を選択しました。`);
    } catch (error) {
      pushLog(`実行結果を取得できません: ${error.message}`, 'error');
      toast(`実行結果を取得できません: ${error.message}`, 'error');
    }
  }

  async function handleTerminalRun() {
    if (!state.run) return;
    const terminal = state.run.status;
    const wasFollowingLatest = state.followLatest;
    stopPlayback();
    state.followLatest = false;
    await loadFrames({ silent: true });
    if (wasFollowingLatest && ['completed', 'stopped'].includes(terminal)) {
      // Once a run has a final artifact, return to the conventional final
      // result view; the saved frame selector remains available for playback.
      state.selectedFrameStep = null;
      state.resultControls = { ...(state.resultControls || {}), frameStep: null };
    }
    setRunState(runStatusLabel(terminal), false);
    updateRunOverlay();
    state.selectedNode = 'results';
    switchView('results');
    renderTree(); renderInspector();
    if (!state.history.length && state.run.id) await loadHistory(state.run.id);
    renderHistory();
    if (state.run.id && (state.frames.length || ['completed', 'stopped'].includes(terminal))) await loadSlice({ silent: true });
    if (terminal === 'completed') {
      const nativeMonitor = Boolean(state.run.diagnostics?.convergence_monitor);
      pushLog(`計算が完了しました。${state.run.converged === true ? (nativeMonitor ? '定常性の基準を満たしました。' : '収束条件を満たしました。') : state.run.converged === false ? (nativeMonitor ? '指定ステップでは定常性の基準を満たしていません。' : '指定ステップで未収束です。') : ''}`);
      toast('計算が完了しました。');
    } else if (terminal === 'stopped') {
      pushLog('計算を停止しました。保存済みの途中結果がある場合のみ取得できます。', 'warn');
      toast('計算を停止しました。', 'warn');
    } else if (terminal === 'failed') {
      pushLog(`計算に失敗しました: ${state.run.error || state.run.message || 'サーバーログを確認してください。'}`, 'error');
      toast('計算に失敗しました。', 'error');
    }
    renderInspector();
  }

  function updateRunOverlay() {
    const overlay = $('#runOverlay');
    if (!overlay || !state.run) return;
    const active = ['queued', 'running', 'stopping'].includes(state.run.status);
    const terminalStatus = ['completed', 'stopped', 'failed', 'interrupted', 'cancelled', 'canceled'].includes(state.run.status);
    overlay.classList.toggle('hidden', !active && !terminalStatus);
    $('#runStatusLabel').textContent = runStatusLabel(state.run.status);
    $('#runProgressLabel').textContent = `${Math.round((state.run.progress || 0) * 100)}%`;
    $('#runProgressBar').style.width = `${Math.round((state.run.progress || 0) * 100)}%`;
    $('#runStepMetric').textContent = `step ${state.run.step ?? '—'}`;
    $('#runResidualMetric').textContent = `最大相対変化 ${monitorValue(nullableNumber(state.run.residual))}`;
    const min = state.run.temperature_min;
    const max = state.run.temperature_max;
    $('#runTemperatureMetric').textContent = min !== undefined || max !== undefined ? `T ${fmt(min, 2)}–${fmt(max, 2)} K` : 'T —';
    if (!active) window.setTimeout(() => { if (state.run && !['queued', 'running', 'stopping'].includes(state.run.status)) overlay.classList.add('hidden'); }, 2600);
  }

  function normalizeHistory(history) {
    if (!Array.isArray(history)) return [];
    return history.map((item) => {
      if (Array.isArray(item)) return {
        step: numberOr(item[0], 0),
        residual: nullableNumber(item[1]),
        temperature_min: nullableNumber(item[2]),
        temperature_max: nullableNumber(item[3]),
        temperature_residual: null,
        velocity_residual: null,
        pressure_residual: null,
        residual_kind: null,
        monitor_tolerance: null,
        monitor_required_samples: null,
        monitor_consecutive_samples: null,
        monitor_satisfied: null,
        monitor_eligible: null
      };
      const source = item && typeof item === 'object' ? item : {};
      const read = (names, fallback = null) => {
        for (const name of names) {
          if (hasOwn(source, name)) return source[name];
        }
        return fallback;
      };
      const normalized = {
        ...source,
        step: numberOr(read(['step', 'iteration', 'time_step'], 0), 0),
        time: nullableNumber(read(['time', 'time_s', 't'])),
        sample_steps: nullableNumber(read(['sample_steps', 'sample_step', 'interval_steps'])),
        sample_time: nullableNumber(read(['sample_time', 'sample_time_s', 'interval_time'])),
        residual: nullableNumber(read(['residual', 'error', 'l2'])),
        velocity_residual: nullableNumber(read(['velocity_residual', 'velocity_change_relative'])),
        pressure_residual: nullableNumber(read(['pressure_residual', 'pressure_change_relative'])),
        temperature_residual: nullableNumber(read(['temperature_residual', 'temp_residual', 'temperature_change_relative'])),
        velocity_change_max: nullableNumber(read(['velocity_change_max', 'velocity_change'])),
        pressure_change_max: nullableNumber(read(['pressure_change_max', 'pressure_change'])),
        temperature_change_max: nullableNumber(read(['temperature_change_max', 'temp_change_max', 'temperature_change'])),
        temperature_min: nullableNumber(read(['temperature_min', 'temp_min'])),
        temperature_max: nullableNumber(read(['temperature_max', 'temp_max'])),
        monitor_tolerance: nullableNumber(read(['monitor_tolerance', 'monitor_tol'])),
        monitor_required_samples: nullableNumber(read(['monitor_required_samples', 'required_samples'])),
        monitor_consecutive_samples: nullableNumber(read(['monitor_consecutive_samples', 'consecutive_samples'])),
        monitor_satisfied: nullableBoolean(read(['monitor_satisfied', 'satisfied'])),
        monitor_eligible: nullableBoolean(read(['monitor_eligible', 'eligible'])),
        residual_kind: (() => {
          const value = read(['residual_kind', 'monitor_kind']);
          return value === null || value === undefined || String(value).trim() === '' ? null : String(value);
        })()
      };
      return normalized;
    }).filter((item) => Number.isFinite(item.step));
  }

  async function loadHistory(runId) {
    if (!runId) return [];
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/files/history.csv`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const text = await response.text();
      state.history = parseHistoryCsv(text);
      return state.history;
    } catch (error) {
      if (!state.history.length) pushLog(`収束履歴を取得できません: ${error.message}`, 'warn');
      return state.history;
    }
  }

  function parseHistoryCsv(text) {
    const rows = String(text || '').replace(/^\uFEFF/, '').trim().split(/\r?\n/).filter(Boolean);
    if (rows.length < 2) return [];
    const parseCsvLine = (line) => {
      const columns = [];
      let current = '';
      let quoted = false;
      for (let index = 0; index < line.length; index += 1) {
        const character = line[index];
        if (character === '"') {
          if (quoted && line[index + 1] === '"') { current += '"'; index += 1; }
          else quoted = !quoted;
        } else if (character === ',' && !quoted) {
          columns.push(current.trim()); current = '';
        } else current += character;
      }
      columns.push(current.trim());
      return columns;
    };
    const headers = parseCsvLine(rows[0]).map((header) => header.trim().toLowerCase());
    const findColumn = (names) => {
      return headers.findIndex((header) => names.includes(header));
    };
    const columns = {
      step: findColumn(['step', 'iteration', 'time_step']),
      time: findColumn(['time', 'time_s', 't']),
      sample_steps: findColumn(['sample_steps', 'sample_step', 'interval_steps']),
      sample_time: findColumn(['sample_time', 'sample_time_s', 'interval_time']),
      residual: findColumn(['residual', 'error', 'l2']),
      velocity_residual: findColumn(['velocity_residual', 'velocity_change_relative']),
      pressure_residual: findColumn(['pressure_residual', 'pressure_change_relative']),
      temperature_residual: findColumn(['temperature_residual', 'temp_residual', 'temperature_change_relative']),
      velocity_change_max: findColumn(['velocity_change_max', 'velocity_change']),
      pressure_change_max: findColumn(['pressure_change_max', 'pressure_change']),
      temperature_change_max: findColumn(['temperature_change_max', 'temp_change_max', 'temperature_change']),
      temperature_min: findColumn(['temperature_min', 'temp_min']),
      temperature_max: findColumn(['temperature_max', 'temp_max']),
      monitor_tolerance: findColumn(['monitor_tolerance', 'monitor_tol']),
      monitor_required_samples: findColumn(['monitor_required_samples', 'required_samples']),
      monitor_consecutive_samples: findColumn(['monitor_consecutive_samples', 'consecutive_samples']),
      monitor_satisfied: findColumn(['monitor_satisfied', 'satisfied']),
      monitor_eligible: findColumn(['monitor_eligible', 'eligible']),
      residual_kind: findColumn(['residual_kind', 'monitor_kind'])
    };
    const valueAt = (row, key) => columns[key] >= 0 ? row[columns[key]] : null;
    return rows.slice(1).map((line) => {
      const row = parseCsvLine(line);
      return normalizeHistory([{
        step: valueAt(row, 'step'), time: valueAt(row, 'time'), sample_steps: valueAt(row, 'sample_steps'), sample_time: valueAt(row, 'sample_time'),
        residual: valueAt(row, 'residual'), velocity_residual: valueAt(row, 'velocity_residual'), pressure_residual: valueAt(row, 'pressure_residual'), temperature_residual: valueAt(row, 'temperature_residual'),
        velocity_change_max: valueAt(row, 'velocity_change_max'), pressure_change_max: valueAt(row, 'pressure_change_max'), temperature_change_max: valueAt(row, 'temperature_change_max'),
        temperature_min: valueAt(row, 'temperature_min'), temperature_max: valueAt(row, 'temperature_max'), monitor_tolerance: valueAt(row, 'monitor_tolerance'),
        monitor_required_samples: valueAt(row, 'monitor_required_samples'), monitor_consecutive_samples: valueAt(row, 'monitor_consecutive_samples'), monitor_satisfied: valueAt(row, 'monitor_satisfied'),
        monitor_eligible: valueAt(row, 'monitor_eligible'), residual_kind: valueAt(row, 'residual_kind')
      }])[0];
    }).filter(Boolean);
  }

  function renderHistory() {
    drawHistoryCanvas();
    const stats = $('#historyStats');
    if (!stats) return;
    const status = monitorState(state.run);
    if (!state.history.length) {
      const message = status.active
        ? '計算中：出力間隔ごとの残差モニターサンプルを待っています。'
        : 'この実行には残差モニター履歴がありません。';
      stats.innerHTML = `<div class="empty-state monitor-empty" style="min-height:80px;grid-column:1/-1">${esc(message)}</div>`;
      return;
    }
    const latest = status.latest || state.history[state.history.length - 1];
    const latestTime = nullableNumber(latest?.time);
    if (status.legacy) {
      const legacyLast = status.legacyRows[status.legacyRows.length - 1];
      stats.innerHTML = `
        <div class="monitor-dock-status neutral"><span>旧形式履歴</span><strong>統合 residual（legacy）</strong></div>
        <div class="monitor-dock-note">この実行には場ごとの残差モニター項目がありません。統合 residual を参考値として表示しています。</div>
        <div class="history-stat"><span>記録点</span><strong>${esc(fmt(state.history.length, 0))}</strong></div>
        <div class="history-stat"><span>最終 step</span><strong>${esc(fmt(legacyLast?.step, 0))}</strong></div>
        <div class="history-stat"><span>最終 residual</span><strong>${esc(fmtSci(legacyLast?.residual))}</strong></div>
        <div class="history-stat"><span>判定</span><strong>参考値</strong></div>`;
      return;
    }
    const fields = MONITOR_COMPONENTS.map((series) => {
      const residual = nullableNumber(latest?.[series.key]);
      const change = nullableNumber(latest?.[series.changeKey]);
      return `<div class="monitor-history-value"><span>${series.label}</span><strong>${esc(monitorValue(residual))}</strong><small>Δ ${esc(monitorValue(change, series.unit))}</small></div>`;
    }).join('');
    stats.innerHTML = `
      <div class="monitor-dock-status ${monitorStatusClass(status)}"><span>状態</span><strong>${esc(monitorStatusText(status))}</strong></div>
      <div class="monitor-dock-note">定常性の目安（前回記録からの相対変化） · 同じ閾値を有効な場に適用</div>
      <div class="monitor-history-values">${fields}</div>
      <div class="history-stat"><span>記録点</span><strong>${esc(fmt(state.history.length, 0))}</strong></div>
      <div class="history-stat"><span>最新 step / time</span><strong>${esc(fmt(latest?.step, 0))}${latestTime === null ? '' : ` / ${esc(fmt(latestTime, 6))} s`}</strong></div>
      <div class="history-stat"><span>許容値</span><strong>${esc(fmtSci(status.settings.tolerance))}</strong></div>
      <div class="history-stat"><span>連続サンプル</span><strong>${esc(fmt(status.consecutive, 0))} / ${esc(fmt(status.settings.requiredSamples, 0))}</strong></div>`;
  }

  function latestFrame() {
    return state.frames.length ? state.frames[state.frames.length - 1] : null;
  }

  function frameByStep(step) {
    if (step === null || step === undefined) return null;
    return state.frames.find((frame) => frame.step === Number(step)) || null;
  }

  async function loadFrames(options = {}) {
    const runId = state.run?.id;
    if (!runId) {
      state.frames = [];
      state.resultFields = [];
      return { frames: [], fields: [] };
    }
    const requestId = state.framesRequest = (state.framesRequest || 0) + 1;
    state.framesLoading = true;
    if (!options.silent) renderInspector();
    try {
      const payload = await request(`/api/runs/${encodeURIComponent(runId)}/frames`);
      if (state.run?.id !== runId || state.framesRequest !== requestId) return { frames: state.frames, fields: state.resultFields };
      const frames = (Array.isArray(payload?.frames) ? payload.frames : []).map(normalizeFrame).filter(Boolean);
      const unique = new Map();
      frames.forEach((frame) => unique.set(frame.step, frame));
      state.frames = Array.from(unique.values()).sort((first, second) => first.step - second.step);
      const rawFields = Array.isArray(payload?.fields) ? payload.fields : (Array.isArray(payload?.available_fields) ? payload.available_fields : []);
      state.resultFields = rawFields.map(normalizeResultField).filter(Boolean).sort((first, second) => {
        const a = RESULT_FIELD_ORDER.indexOf(first.id); const b = RESULT_FIELD_ORDER.indexOf(second.id);
        return (a < 0 ? 999 : a) - (b < 0 ? 999 : b);
      });
      state.framesEndpointUnavailable = false;
      const latest = latestFrame();
      const selectedStillExists = state.selectedFrameStep === null || frameByStep(state.selectedFrameStep);
      if (!selectedStillExists) state.selectedFrameStep = null;
      const wasLatest = state.resultControls?.frameStep === latest?.step;
      if (state.followLatest && latest) {
        state.selectedFrameStep = latest.step;
        state.resultControls = { ...(state.resultControls || {}), frameStep: latest.step };
      } else if (latest && ['failed', 'interrupted', 'cancelled', 'canceled'].includes(state.run.status) && state.selectedFrameStep === null) {
        // Interrupted runs may have no final fields artifact; their newest
        // saved frame is still a valid result to inspect.
        state.selectedFrameStep = latest.step;
        state.resultControls = { ...(state.resultControls || {}), frameStep: latest.step };
      }
      renderInspector();
      if (options.autoLoad && state.followLatest && latest && (!state.slice || !wasLatest || state.slice.step !== latest.step)) await loadSlice({ silent: true });
      return { frames: state.frames, fields: state.resultFields };
    } catch (error) {
      if (!state.framesEndpointUnavailable) {
        state.framesEndpointUnavailable = true;
        pushLog(`保存ステップ一覧を取得できません: ${error.message}（最終結果の表示は継続します）`, 'warn');
      }
      return { frames: state.frames, fields: state.resultFields };
    } finally {
      if (state.run?.id === runId && state.framesRequest === requestId) {
        state.framesLoading = false;
        renderInspector();
      }
    }
  }

  function stopPlayback() {
    window.clearInterval(state.playbackTimer);
    state.playbackTimer = null;
    state.playbackActive = false;
  }

  function advanceFrame() {
    if (state.sliceLoading) return;
    if (state.frames.length < 2) { stopPlayback(); renderInspector(); return; }
    const current = state.frames.findIndex((frame) => frame.step === Number(state.selectedFrameStep));
    const next = state.frames[(current + 1 + state.frames.length) % state.frames.length];
    state.selectedFrameStep = next.step;
    state.followLatest = false;
    state.resultControls = { ...(state.resultControls || {}), frameStep: next.step };
    renderInspector();
    loadSlice({ silent: true });
  }

  function toggleResultPlayback() {
    if (state.playbackActive) {
      stopPlayback();
      renderInspector();
      return;
    }
    if (state.frames.length < 2) {
      toast('再生できる保存ステップがありません。', 'warn');
      return;
    }
    state.followLatest = false;
    state.playbackActive = true;
    state.playbackTimer = window.setInterval(advanceFrame, 720);
    advanceFrame();
    renderInspector();
  }

  function selectFrameStep(value, options = {}) {
    const raw = String(value ?? '').trim();
    const step = raw === '' ? null : Number(raw);
    state.selectedFrameStep = Number.isFinite(step) ? Math.round(step) : null;
    if (!options.keepFollowing) state.followLatest = false;
    state.resultControls = { ...(state.resultControls || {}), frameStep: state.selectedFrameStep };
    if (options.stopPlayback !== false) stopPlayback();
    renderInspector();
    if (options.load !== false) loadSlice({ silent: Boolean(options.silent) });
  }

  function toggleFollowLatest(enabled) {
    state.followLatest = Boolean(enabled);
    if (state.followLatest) {
      const latest = latestFrame();
      if (latest) {
        state.selectedFrameStep = latest.step;
        state.resultControls = { ...(state.resultControls || {}), frameStep: latest.step };
      }
    }
    renderInspector();
    if (state.followLatest && latestFrame()) loadSlice({ silent: true });
  }

  function refreshResultSlice() {
    // Invalidate an in-flight request before clearing the canvas so changing
    // field, plane, or index can never leave an older field painted as if it
    // were the current selection.
    state.sliceRequest = (state.sliceRequest || 0) + 1;
    state.sliceLoading = false;
    state.slice = null;
    drawResult();
    renderInspector();
    const status = state.run?.status;
    const canReadResult = Boolean(state.run?.id && (state.frames.length > 0 || ['completed', 'stopped'].includes(status)));
    if (canReadResult) loadSlice({ silent: true });
  }

  async function loadSlice(options = {}) {
    const activeRun = ['queued', 'running', 'stopping'].includes(state.run?.status);
    const canReadResult = state.frames.length > 0 || ['completed', 'stopped'].includes(state.run?.status);
    if (!state.run?.id || !canReadResult) {
      if (!options.silent) toast('保存された計算結果がありません。', 'warn');
      return;
    }
    const field = $('[data-result-field]')?.value || state.slice?.field || 'temperature';
    const axis = $('[data-result-axis]')?.value || state.slice?.axis || 'y';
    const shape = state.run.shape || state.mesh?.shape || state.project.mesh.cells;
    const axisLength = shape[['x', 'y', 'z'].indexOf(axis)] || 1;
    const index = clamp(Math.round(numberOr($('[data-result-index]')?.value, Math.floor((axisLength - 1) / 2))), 0, Math.max(0, axisLength - 1));
    let step = state.selectedFrameStep;
    if (step === null || step === undefined) step = state.resultControls?.frameStep ?? null;
    if (state.followLatest && latestFrame()) step = latestFrame().step;
    if (activeRun && step === null) {
      state.slice = null;
      drawResult();
      return;
    }
    const runId = state.run.id;
    const requestId = state.sliceRequest = (state.sliceRequest || 0) + 1;
    state.resultControls = {field,axis,index,frameStep: step};
    state.sliceLoading = true;
    renderInspector();
    try {
      const stepQuery = step !== null && step !== undefined && Number.isFinite(Number(step)) ? `&step=${Math.round(Number(step))}` : '';
      const payload = await request(`/api/runs/${encodeURIComponent(runId)}/slice?field=${encodeURIComponent(field)}&axis=${encodeURIComponent(axis)}&index=${index}${stepQuery}`);
      if (state.run?.id !== runId || state.sliceRequest !== requestId) return;
      if (!payload || !Array.isArray(payload.values)) throw new Error('スライス配列がありません');
      state.slice = { ...payload, field, axis, index, step: payload.step ?? step, values: payload.values };
      if (payload.shape && !state.run.shape) state.run.shape = payload.shape;
      switchView('results');
      if (!options.silent) {
        pushLog(`${FIELD_INFO[field]?.label || field} の ${axis.toUpperCase()} 断面を取得しました。min ${fmtSci(payload.min)} / max ${fmtSci(payload.max)} ${payload.unit || ''}`);
        toast('結果スライスを表示しました。');
      }
    } catch (error) {
      pushLog(`スライス取得に失敗しました: ${error.message}`, options.silent ? 'warn' : 'error');
      if (!options.silent) toast(`スライス取得に失敗しました: ${error.message}`, 'error');
    } finally {
      if (state.run?.id === runId && state.sliceRequest === requestId) {
        state.sliceLoading = false;
        renderInspector();
        drawResult();
      }
    }
  }

  async function downloadRunFile(filename) {
    if (!state.run?.id) return;
    try {
      const response = await fetch(`/api/runs/${encodeURIComponent(state.run.id)}/files/${encodeURIComponent(filename)}`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; document.body.appendChild(anchor); anchor.click(); anchor.remove(); URL.revokeObjectURL(url);
      pushLog(`${filename} をダウンロードしました。`);
    } catch (error) {
      pushLog(`${filename} のダウンロードに失敗しました: ${error.message}`, 'error');
      toast(`${filename} を取得できません: ${error.message}`, 'error');
    }
  }

  function validateAndReport() {
    const messages = validateProject();
    state.validation = messages;
    renderValidationDock();
    switchDock('validation');
    const errors = messages.filter((message) => message.level === 'error');
    if (errors.length) {
      pushLog(`検証完了: ${errors.length} 件のエラーがあります。`, 'error');
      toast(`${errors.length} 件のエラーを修正してください。`, 'error');
    } else if (messages.length) {
      pushLog(`検証完了: ${messages.length} 件の注意があります。`, 'warn');
      toast('検証は完了しました。注意事項を確認してください。', 'warn');
    } else {
      pushLog('検証完了: プロジェクト設定は有効です。');
      toast('検証 OK');
    }
  }

  function resizeCanvas(canvas) {
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width || canvas.clientWidth || 500));
    const height = Math.max(1, Math.round(rect.height || canvas.clientHeight || 300));
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const targetWidth = Math.round(width * ratio);
    const targetHeight = Math.round(height * ratio);
    if (canvas.width !== targetWidth || canvas.height !== targetHeight) { canvas.width = targetWidth; canvas.height = targetHeight; }
    const context = canvas.getContext('2d');
    if (context) context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
  }

  function extractBounds(rawBounds) {
    const asPoint = (value) => {
      const point = pointToArray(value);
      return point && point.every((component) => Number.isFinite(component)) ? point : null;
    };
    if (Array.isArray(rawBounds) && rawBounds.length >= 6) {
      const min = rawBounds.slice(0, 3).map(Number);
      const max = rawBounds.slice(3, 6).map(Number);
      if (min.every(Number.isFinite) && max.every(Number.isFinite)) return { min, max };
    }
    if (Array.isArray(rawBounds) && rawBounds.length === 2) {
      const min = asPoint(rawBounds[0]);
      const max = asPoint(rawBounds[1]);
      if (min && max) return { min, max };
    }
    if (rawBounds && typeof rawBounds === 'object') {
      const min = asPoint(rawBounds.min ?? rawBounds.minimum ?? rawBounds.lower);
      const max = asPoint(rawBounds.max ?? rawBounds.maximum ?? rawBounds.upper);
      if (min && max) return { min, max };
    }
    return null;
  }

  function modelBounds() {
    const geometry = state.project?.geometry;
    if (!geometry) return { min: [-1, -1, -1], max: [1, 1, 1] };
    if (geometry.kind === 'box' && Array.isArray(geometry.size)) return { min: [0, 0, 0], max: geometry.size.map((value) => Math.max(numberOr(value, 1), 1e-9)) };
    const meta = state.cadAssetMeta;
    const bounds = extractBounds(meta?.bounds || meta?.bounding_box || meta?.bounds_m);
    if (bounds) return bounds;
    const vertices = meta?.vertices;
    if (Array.isArray(vertices) && vertices.length) return calculateBounds(vertices);
    return { min: [0, 0, 0], max: [0.06, 0.02, 0.02] };
  }

  function calculateBounds(points) {
    const parsed = points.map(pointToArray).filter(Boolean);
    if (!parsed.length) return { min: [0, 0, 0], max: [0.06, 0.02, 0.02] };
    return { min: [0, 1, 2].map((index) => Math.min(...parsed.map((point) => point[index]))), max: [0, 1, 2].map((index) => Math.max(...parsed.map((point) => point[index]))) };
  }

  function pointToArray(point) {
    if (Array.isArray(point) && point.length >= 3) return point.slice(0, 3).map(Number);
    if (point && typeof point === 'object' && ['x', 'y', 'z'].every((key) => point[key] !== undefined)) return [Number(point.x), Number(point.y), Number(point.z)];
    return null;
  }

  function geometryPoints() {
    const preview = state.mesh?.preview || state.mesh;
    const cadSurface = state.cadAssetMeta;
    const previewSurfacePoints = preview?.vertices || preview?.surface_vertices;
    const previewSurfaceFaces = preview?.faces || preview?.surface_faces;
    const previewMeshPoints = preview?.points;
    if (state.project?.geometry?.kind === 'cad') {
      // Mesh previews contain cell-centre samples (`points`) and separate
      // surface vertices/faces.  Face indices always refer to the latter.
      const surfacePoints = Array.isArray(cadSurface?.vertices) && cadSurface.vertices.length
        ? cadSurface.vertices : (Array.isArray(previewSurfacePoints) ? previewSurfacePoints : []);
      const surfaceFaces = Array.isArray(cadSurface?.faces) && cadSurface.faces.length
        ? cadSurface.faces : (Array.isArray(previewSurfaceFaces) ? previewSurfaceFaces : []);
      if (surfacePoints.length) return {
        points: surfacePoints.map(pointToArray).filter(Boolean),
        faces: surfaceFaces,
        meshPoints: Array.isArray(previewMeshPoints) ? previewMeshPoints.map(pointToArray).filter(Boolean) : [],
        source: 'cad'
      };
      if (Array.isArray(previewMeshPoints) && previewMeshPoints.length) return { points: previewMeshPoints.map(pointToArray).filter(Boolean), faces: [], source: 'mesh', meshPoints: [] };
    }
    if (state.project?.geometry?.kind === 'box') {
      const bounds = modelBounds();
      const [x0, y0, z0] = bounds.min; const [x1, y1, z1] = bounds.max;
      return {
        points: [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],
        faces: [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]],
        faceGroups: BOX_FACE_GROUPS.slice(),
        meshPoints: Array.isArray(previewMeshPoints) ? previewMeshPoints.map(pointToArray).filter(Boolean) : [],
        source: 'box'
      };
    }
    if (preview) {
      const points = preview.points || preview.vertices;
      if (Array.isArray(points) && points.length) return { points: points.map(pointToArray).filter(Boolean), faces: Array.isArray(preview.faces) ? preview.faces : [], source: 'mesh', meshPoints: [] };
    }
    const bounds = modelBounds();
    const [x0, y0, z0] = bounds.min; const [x1, y1, z1] = bounds.max;
    return { points: [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0], [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], faces: [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]], faceGroups: BOX_FACE_GROUPS.slice(), source: 'box', meshPoints: [] };
  }

  function triangleIndicesForSurface(groupId, count) {
    if (BOX_FACE_GROUPS.includes(String(groupId))) return new Set([BOX_FACE_GROUPS.indexOf(String(groupId))].filter((index) => index >= 0 && index < count));
    const group = surfaceGroupInfo(groupId);
    const metadata = state.cadAssetMeta || state.mesh?.preview || state.mesh || {};
    const set = new Set();
    if (!group) return set;
    const arrays = [group.triangle_indices, group.triangleIndices, group.face_indices, group.faceIndices, group.triangles].find((value) => Array.isArray(value));
    if (arrays) arrays.forEach((value) => { if (Number.isInteger(Number(value))) set.add(Number(value)); });
    const start = numberOr(group.triangle_start ?? group.start, NaN); const triangleCount = numberOr(group.triangle_count ?? group.count, NaN);
    if (Number.isFinite(start) && Number.isFinite(triangleCount)) for (let index = Math.max(0, start); index < Math.min(count, start + triangleCount); index += 1) set.add(index);
    const triangleGroups = metadata.triangle_groups || metadata.triangleGroups;
    if (Array.isArray(triangleGroups)) triangleGroups.forEach((item, index) => {
      const itemId = typeof item === 'string' ? item : item?.id ?? item?.patch_id ?? item?.group_id ?? item?.name;
      if (String(itemId) === String(groupId)) set.add(index);
    });
    return set;
  }

  function surfaceGroupForTriangle(index, source = '') {
    if (source === 'box' && BOX_FACE_GROUPS[index]) return BOX_FACE_GROUPS[index];
    const groups = surfaceGroups();
    const metadata = state.cadAssetMeta || state.mesh?.preview || state.mesh || {};
    const triangleGroups = metadata.triangle_groups || metadata.triangleGroups;
    if (Array.isArray(triangleGroups)) {
      const item = triangleGroups[index]; const id = typeof item === 'string' ? item : item?.id ?? item?.patch_id ?? item?.group_id ?? item?.name;
      if (id !== undefined && groups.some((group) => surfaceGroupId(group) === String(id))) return String(id);
    }
    for (const group of groups) if (triangleIndicesForSurface(surfaceGroupId(group), index + 1).has(index)) return surfaceGroupId(group);
    return null;
  }

  function rotate3d(point, center) {
    const x = point[0] - center[0]; const y = point[1] - center[1]; const z = point[2] - center[2];
    const cy = Math.cos(state.orbit.yaw); const sy = Math.sin(state.orbit.yaw);
    const cp = Math.cos(state.orbit.pitch); const sp = Math.sin(state.orbit.pitch);
    const x1 = x * cy - z * sy; const z1 = x * sy + z * cy;
    return [x1, y * cp - z1 * sp, y * sp + z1 * cp];
  }

  function drawGeometry() {
    const canvas = $('#geometryCanvas');
    const info = resizeCanvas(canvas);
    if (!info?.context || !state.project) return;
    const { context: ctx, width, height } = info;
    ctx.clearRect(0, 0, width, height);
    const background = ctx.createLinearGradient(0, 0, 0, height);
    background.addColorStop(0, '#091d28'); background.addColorStop(1, '#061118');
    ctx.fillStyle = background; ctx.fillRect(0, 0, width, height);
    if (state.showGrid) drawViewportGrid(ctx, width, height);
    const geometryData = geometryPoints();
    const { points, faces, meshPoints = [], faceGroups = [], source = '' } = geometryData;
    if (!points.length) return;
    const bounds = calculateBounds(points);
    const center = [0, 1, 2].map((index) => (bounds.min[index] + bounds.max[index]) / 2);
    const span = Math.max(...[0, 1, 2].map((index) => Math.abs(bounds.max[index] - bounds.min[index])), 1e-8);
    const scale = Math.min(width, height) * .60 / span * state.orbit.zoom;
    const projected = points.map((point) => { const rotated = rotate3d(point, center); return [width / 2 + rotated[0] * scale, height / 2 - rotated[1] * scale, rotated[2]]; });
    const faceList = faces.length ? faces : [];
    const visibleFaces = faceList.map((face, faceIndex) => {
      const indices = Array.isArray(face) ? face : (face?.indices || face?.vertices || []);
      const valid = indices.map((index) => projected[index]).filter(Boolean);
      const depth = valid.reduce((sum, point) => sum + point[2], 0) / Math.max(valid.length, 1);
      return { indices, valid, depth, faceIndex };
    }).filter((face) => face.valid.length >= 3).sort((a, b) => a.depth - b.depth);
    const selectedTriangles = state.selectedSurface && source === 'box' ? triangleIndicesForSurface(state.selectedSurface, faceList.length) : new Set();
    state.projectedFaces = visibleFaces.map((face) => {
      const centroid = face.valid.reduce((result, point) => [result[0] + point[0], result[1] + point[1]], [0, 0]).map((value) => value / face.valid.length);
      return { x: centroid[0], y: centroid[1], faceIndex: face.faceIndex, groupId: faceGroups[face.faceIndex] || surfaceGroupForTriangle(face.faceIndex, source), vertices: face.valid.map((point) => point.slice()) };
    });
    visibleFaces.forEach((face, index) => {
      ctx.beginPath(); ctx.moveTo(face.valid[0][0], face.valid[0][1]); face.valid.slice(1).forEach((point) => ctx.lineTo(point[0], point[1])); ctx.closePath();
      const opacity = state.mesh ? .13 + (index % 3) * .025 : .18;
      const groupId = faceGroups[face.faceIndex] || surfaceGroupForTriangle(face.faceIndex, source);
      const highlighted = selectedTriangles.has(face.faceIndex) || (Boolean(groupId) && groupId === state.selectedSurface);
      ctx.fillStyle = highlighted ? 'rgba(247, 193, 82, .62)' : state.mesh ? `rgba(50, 209, 195, ${opacity})` : `rgba(79, 185, 233, ${opacity})`;
      ctx.fill(); ctx.strokeStyle = highlighted ? 'rgba(255, 220, 123, .95)' : state.mesh ? 'rgba(77, 204, 212, .58)' : 'rgba(104, 184, 217, .8)'; ctx.lineWidth = highlighted ? 1.6 : 1; ctx.stroke();
    });
    const projectedMesh = meshPoints.map((point) => { const rotated = rotate3d(point, center); return [width / 2 + rotated[0] * scale, height / 2 - rotated[1] * scale, rotated[2]]; });
    if (state.mesh && projectedMesh.length <= 4500) {
      ctx.fillStyle = 'rgba(161, 243, 230, .65)';
      projectedMesh.forEach((point) => { ctx.beginPath(); ctx.arc(point[0], point[1], 1.35, 0, Math.PI * 2); ctx.fill(); });
    }
    if (state.project.geometry.kind === 'box' && Array.isArray(state.project.geometry.solids) && state.project.geometry.solids.length) drawSolidRegions(ctx, state.project.geometry.solids, center, scale, width, height);
    drawAxisTriad(ctx, width - 74, height - 55);
    const geometry = state.project.geometry;
    $('#geometryOverlayTitle').textContent = geometry.role === 'obstacle' ? '障害物 / 固体' : '流体領域';
    const dimensions = modelBounds();
    const extents = [0, 1, 2].map((index) => (dimensions.max[index] - dimensions.min[index]) * 1000);
    $('#geometryOverlaySub').textContent = geometry.kind === 'cad' ? `CAD / ${extents.map((value) => `${fmt(value, 2)} mm`).join(' × ')}` : `ボックス / ${extents.map((value) => `${fmt(value, 0)}`).join(' × ')} mm`;
    const badge = $('#meshBadge');
    badge.classList.toggle('hidden', !state.mesh);
    if (state.mesh) $('#meshBadgeText').textContent = `${fmt(state.mesh.fluid_cells, 0)} fluid cells`;
    if (state.selectedSurface) {
      const group = surfaceGroupInfo(state.selectedSurface);
      const selectedLabel = FACES.includes(state.selectedSurface) ? faceLabel(state.selectedSurface) : (group?.name || faceLabel(`cad:${state.selectedSurface}`));
      const triangleCount = group?.triangle_count ?? group?.triangleCount;
      $('#geometryOverlaySub').textContent = triangleCount !== undefined ? `${selectedLabel} · ${fmt(triangleCount, 0)} triangles` : `${selectedLabel} · 境界面`;
    }
  }

  function drawSolidRegions(ctx, solids, center, scale, width, height) {
    const edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
    solids.forEach((solid) => {
      const origin = solid.origin || [0, 0, 0]; const end = origin.map((value, index) => value + (solid.size?.[index] || 0));
      const corners = [[origin[0], origin[1], origin[2]], [end[0], origin[1], origin[2]], [end[0], end[1], origin[2]], [origin[0], end[1], origin[2]], [origin[0], origin[1], end[2]], [end[0], origin[1], end[2]], [end[0], end[1], end[2]], [origin[0], end[1], end[2]]].map((point) => { const rotated = rotate3d(point, center); return [width / 2 + rotated[0] * scale, height / 2 - rotated[1] * scale, rotated[2]]; });
      ctx.save(); ctx.fillStyle = 'rgba(234, 183, 101, .17)'; ctx.strokeStyle = 'rgba(234, 183, 101, .82)'; ctx.lineWidth = 1;
      [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6]].forEach((face) => { ctx.beginPath(); ctx.moveTo(corners[face[0]][0], corners[face[0]][1]); face.slice(1).forEach((index) => ctx.lineTo(corners[index][0], corners[index][1])); ctx.closePath(); ctx.fill(); ctx.stroke(); });
      ctx.fillStyle = '#f1c77a'; ctx.font = '10px ' + getComputedStyle(document.body).fontFamily; ctx.fillText(solid.name || 'solid', corners[0][0] + 4, corners[0][1] - 4); ctx.restore();
    });
  }

  function drawViewportGrid(ctx, width, height) {
    const horizon = height * .62;
    ctx.save(); ctx.strokeStyle = 'rgba(91, 158, 169, .12)'; ctx.lineWidth = 1;
    for (let index = -10; index <= 10; index += 1) {
      const x = width / 2 + index * Math.min(width, height) * .038;
      ctx.beginPath(); ctx.moveTo(x, horizon - 32); ctx.lineTo(width / 2 + index * Math.min(width, height) * .11, height); ctx.stroke();
    }
    for (let index = 0; index < 8; index += 1) {
      const y = horizon + index * index * 6;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
    }
    ctx.restore();
  }

  function drawAxisTriad(ctx, x, y) {
    const length = 27;
    ctx.save(); ctx.lineWidth = 1.5; ctx.font = '10px ' + getComputedStyle(document.body).fontFamily;
    [[1, 0, '#e77d7b', 'X'], [0, -1, '#5cd29d', 'Y'], [-.7, .5, '#5ca8ed', 'Z']].forEach(([dx, dy, color, label]) => { ctx.strokeStyle = color; ctx.fillStyle = color; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + dx * length, y + dy * length); ctx.stroke(); ctx.fillText(label, x + dx * (length + 5), y + dy * (length + 5)); });
    ctx.restore();
  }

  function resetView() { state.orbit = { yaw: -0.58, pitch: 0.34, zoom: 1 }; drawGeometry(); }
  function fitView() { state.orbit.zoom = 1; drawGeometry(); }
  function toggleGrid(button) { state.showGrid = !state.showGrid; button.setAttribute('aria-pressed', String(state.showGrid)); drawGeometry(); }
  function toggleHelp() { $('#helpPopover')?.classList.toggle('hidden'); }

  function fieldColor(value, min, max) {
    if (!Number.isFinite(value)) return [0, 0, 0, 0];
    const ratio = clamp((value - min) / (max - min || 1), 0, 1);
    const stops = [[0, [12, 31, 74]], [.22, [32, 105, 178]], [.48, [36, 190, 202]], [.72, [91, 207, 146]], [1, [240, 205, 88]]];
    let left = stops[0]; let right = stops[stops.length - 1];
    for (let index = 1; index < stops.length; index += 1) { if (ratio <= stops[index][0]) { left = stops[index - 1]; right = stops[index]; break; } }
    const amount = (ratio - left[0]) / (right[0] - left[0] || 1);
    return [0, 1, 2].map((index) => Math.round(left[1][index] + (right[1][index] - left[1][index]) * amount)).concat(255);
  }

  function drawResult() {
    const canvas = $('#resultCanvas');
    const info = resizeCanvas(canvas);
    if (!info?.context) return;
    const { context: ctx, width, height } = info;
    ctx.clearRect(0, 0, width, height);
    const background = ctx.createLinearGradient(0, 0, 0, height); background.addColorStop(0, '#091d28'); background.addColorStop(1, '#061118'); ctx.fillStyle = background; ctx.fillRect(0, 0, width, height);
    const slice = state.slice;
    const empty = $('#resultEmpty');
    const overlay = $('#resultOverlay');
    if (!slice || !Array.isArray(slice.values) || !slice.values.length) {
      empty?.classList.remove('hidden'); overlay?.classList.add('hidden');
      const message = $('#resultEmptyMessage');
      if (message) {
        const active = ['queued', 'running', 'stopping'].includes(state.run?.status);
        message.textContent = active
          ? (state.frames.length ? '保存ステップを選択すると、計算中の最新フィールドを表示します。' : '計算中です。保存間隔に達したスナップショットが到着すると表示できます。')
          : state.run ? 'この実行には表示できる保存フィールドがありません。' : '計算を開始すると、ここに実データのフィールドが表示されます。';
      }
      return;
    }
    empty?.classList.add('hidden'); overlay?.classList.remove('hidden');
    const values = slice.values;
    // The API returns values[horizontal][vertical].  Convert to canvas rows
    // with the physical low side at the bottom of the plot.
    const horizontal = values.length;
    const vertical = Array.isArray(values[0]) ? values[0].length : 0;
    if (!horizontal || !vertical) return;
    const plotBox = { left: 54, top: 52, right: Math.max(120, width - 88), bottom: Math.max(92, height - 37) };
    const availableWidth = Math.max(10, plotBox.right - plotBox.left); const availableHeight = Math.max(10, plotBox.bottom - plotBox.top);
    const extent = normalizeSliceExtent(slice.extent, slice.axis, horizontal, vertical);
    const physicalWidth = Math.abs(extent[2] - extent[0]) || horizontal;
    const physicalHeight = Math.abs(extent[3] - extent[1]) || vertical;
    const physicalRatio = physicalWidth / physicalHeight;
    let plotWidth = availableWidth; let plotHeight = plotWidth / physicalRatio;
    if (plotHeight > availableHeight) { plotHeight = availableHeight; plotWidth = plotHeight * physicalRatio; }
    const plot = { left: plotBox.left + (availableWidth - plotWidth) / 2, top: plotBox.top + (availableHeight - plotHeight) / 2, right: plotBox.left + (availableWidth - plotWidth) / 2 + plotWidth, bottom: plotBox.top + (availableHeight - plotHeight) / 2 + plotHeight };
    const validValues = values.flat().filter((value) => value !== null && value !== undefined && Number.isFinite(Number(value))).map(Number);
    const min = Number.isFinite(Number(slice.min)) ? Number(slice.min) : (validValues.length ? Math.min(...validValues) : 0);
    const max = Number.isFinite(Number(slice.max)) ? Number(slice.max) : (validValues.length ? Math.max(...validValues) : 1);
    const image = ctx.createImageData(horizontal, vertical);
    for (let column = 0; column < horizontal; column += 1) {
      for (let physicalRow = 0; physicalRow < vertical; physicalRow += 1) {
        const raw = values[column]?.[physicalRow];
        const offset = ((vertical - 1 - physicalRow) * horizontal + column) * 4;
        if (raw === null || raw === undefined || !Number.isFinite(Number(raw))) { image.data[offset + 3] = 0; continue; }
        const rgba = fieldColor(Number(raw), min, max); image.data[offset] = rgba[0]; image.data[offset + 1] = rgba[1]; image.data[offset + 2] = rgba[2]; image.data[offset + 3] = rgba[3];
      }
    }
    const offscreen = document.createElement('canvas'); offscreen.width = horizontal; offscreen.height = vertical; offscreen.getContext('2d').putImageData(image, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(offscreen, plot.left, plot.top, plotWidth, plotHeight);
    ctx.strokeStyle = 'rgba(179, 231, 226, .35)'; ctx.lineWidth = 1; ctx.strokeRect(plot.left, plot.top, plotWidth, plotHeight);
    const horizontalLabel = slice.axis === 'x' ? 'Y' : 'X'; const verticalLabel = slice.axis === 'z' ? 'Y' : 'Z';
    ctx.save(); ctx.fillStyle = '#87a9ad'; ctx.font = '10px ' + getComputedStyle(document.body).fontFamily; ctx.textAlign = 'center'; ctx.fillText(horizontalLabel, plot.left + plotWidth / 2, plot.bottom + 25); ctx.translate(plot.left - 37, plot.top + plotHeight / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(verticalLabel, 0, 0); ctx.restore();
    drawColorbar(ctx, plot.right + 17, plot.top, 12, plotHeight, min, max, slice.unit || '');
    const infoData = state.resultFields.find((item) => item.id === slice.field) || FIELD_INFO[slice.field] || { label: slice.field, title: slice.field, unit: slice.unit || '' };
    $('#resultFieldLabel').textContent = String(infoData.label).toUpperCase();
    $('#resultSliceTitle').textContent = infoData.title;
    const frame = frameByStep(slice.step);
    const frameLabel = slice.step !== null && slice.step !== undefined ? ` · step ${slice.step}${frame?.time !== null && frame?.time !== undefined ? ` · t ${fmt(frame.time, 6)} s` : ''}` : '';
    $('#resultSliceSub').textContent = `${slice.axis.toUpperCase()} 断面 · index ${slice.index} · ${slice.shape ? slice.shape.join(' × ') : `${horizontal} × ${vertical}`}${frameLabel}`;
  }

  function normalizeSliceExtent(extent, axis, horizontal, vertical) {
    if (Array.isArray(extent) && extent.length >= 4 && extent.slice(0, 4).every((value) => Number.isFinite(Number(value)))) return extent.slice(0, 4).map(Number);
    if (Array.isArray(extent) && extent.length === 2 && extent.every((part) => Array.isArray(part) && part.length >= 2)) return [Number(extent[0][0]), Number(extent[1][0]), Number(extent[0][1]), Number(extent[1][1])];
    const xSpan = axis === 'x' ? vertical : horizontal; const ySpan = axis === 'z' ? vertical : vertical;
    return [0, 0, xSpan, ySpan];
  }

  function drawColorbar(ctx, x, y, width, height, min, max, unit) {
    const gradient = ctx.createLinearGradient(0, y + height, 0, y); gradient.addColorStop(0, '#0c1f4a'); gradient.addColorStop(.22, '#2069b2'); gradient.addColorStop(.48, '#24beca'); gradient.addColorStop(.72, '#5bcf92'); gradient.addColorStop(1, '#f0cd58'); ctx.fillStyle = gradient; ctx.fillRect(x, y, width, height); ctx.strokeStyle = 'rgba(179, 231, 226, .35)'; ctx.strokeRect(x, y, width, height);
    ctx.fillStyle = '#9ab8ba'; ctx.font = '9px ' + getComputedStyle(document.body).fontFamily; ctx.textAlign = 'left'; ctx.fillText(fmtSci(max), x + width + 6, y + 8); ctx.fillText(fmtSci(min), x + width + 6, y + height); ctx.fillText(unit, x - 1, y - 7);
  }

  function drawHistoryCanvas() {
    const canvas = $('#historyCanvas');
    const info = resizeCanvas(canvas);
    if (!info?.context) return;
    const { context: ctx, width, height } = info;
    ctx.clearRect(0, 0, width, height); ctx.fillStyle = '#07151d'; ctx.fillRect(0, 0, width, height);
    const status = monitorState(state.run);
    const plot = { left: 58, top: 30, right: Math.max(100, width - 18), bottom: Math.max(60, height - 27) };
    const fontFamily = getComputedStyle(document.body).fontFamily;
    const emptyText = status.active
      ? '計算中：出力間隔ごとのモニターサンプルを待っています'
      : 'この実行には残差モニター履歴がありません';
    if (!state.history.length) {
      ctx.fillStyle = '#709095'; ctx.font = `11px ${fontFamily}`; ctx.textAlign = 'center';
      ctx.fillText(emptyText, (plot.left + plot.right) / 2, (plot.top + plot.bottom) / 2);
      return;
    }

    const minStep = Math.min(...state.history.map((item) => item.step));
    const maxStep = Math.max(...state.history.map((item) => item.step), minStep + 1);
    const xFor = (row) => plot.left + (row.step - minStep) / (maxStep - minStep) * (plot.right - plot.left);
    const nativeSeries = MONITOR_COMPONENTS.filter((series) => status.nativeRows.some((row) => Number.isFinite(row[series.key])));
    if (!nativeSeries.length) {
      const legacy = status.legacyRows;
      if (!legacy.length) {
        ctx.fillStyle = '#709095'; ctx.font = `10px ${fontFamily}`; ctx.textAlign = 'center';
        ctx.fillText('モニター値は最初の完全なサンプル後に表示されます', (plot.left + plot.right) / 2, (plot.top + plot.bottom) / 2);
        return;
      }
      const values = legacy.map((row) => row.residual).filter((value) => Number.isFinite(value) && value >= 0);
      if (!values.length) return;
      drawLogMonitorSeries(ctx, plot, state.history, [{ key: 'residual', label: '統合 residual（legacy）', color: '#49d5c8' }], null, xFor, fontFamily);
      return;
    }
    drawLogMonitorSeries(ctx, plot, status.nativeRows, nativeSeries, status.settings.tolerance, xFor, fontFamily);
  }

  function drawLogMonitorSeries(ctx, plot, history, seriesList, tolerance, xFor, fontFamily) {
    const values = [];
    seriesList.forEach((series) => history.forEach((row) => {
      const value = nullableNumber(row[series.key]);
      if (value !== null && value >= 0) values.push(value);
    }));
    if (tolerance !== null && tolerance !== undefined && tolerance > 0) values.push(tolerance);
    if (!values.length) return;
    const positive = values.filter((value) => value > 0);
    const maxValue = Math.max(...values, 1e-12);
    const smallestPositive = positive.length ? Math.min(...positive) : maxValue;
    const floor = Math.max(1e-12, Math.min(smallestPositive * 0.25, Math.max(maxValue, tolerance || 0) * 1e-4));
    const upper = Math.max(maxValue * 1.35, (tolerance || 0) * 6, floor * 100);
    const minLog = Math.log10(floor);
    const maxLog = Math.log10(upper);
    const yFor = (value) => {
      const plotted = value === 0 ? floor : clamp(value, floor, upper);
      return plot.bottom - (Math.log10(plotted) - minLog) / (maxLog - minLog || 1) * (plot.bottom - plot.top);
    };
    ctx.font = `9px ${fontFamily}`;
    ctx.textAlign = 'right'; ctx.fillStyle = '#78979b';
    for (let index = 0; index < 4; index += 1) {
      const fraction = index / 3;
      const y = plot.top + (plot.bottom - plot.top) * fraction;
      const value = Math.pow(10, maxLog - (maxLog - minLog) * fraction);
      ctx.strokeStyle = 'rgba(91, 158, 169, .18)'; ctx.lineWidth = 1; ctx.setLineDash([]);
      ctx.beginPath(); ctx.moveTo(plot.left, y); ctx.lineTo(plot.right, y); ctx.stroke();
      ctx.fillText(fmtSci(value), plot.left - 7, y + 3);
    }
    ctx.strokeStyle = 'rgba(196, 224, 225, .25)'; ctx.setLineDash([2, 3]); ctx.beginPath(); ctx.moveTo(plot.left, plot.bottom); ctx.lineTo(plot.right, plot.bottom); ctx.stroke(); ctx.setLineDash([]);
    ctx.textAlign = 'left'; ctx.fillStyle = '#6f9094'; ctx.fillText('0 / plot floor', plot.left + 3, plot.bottom - 4);

    if (tolerance !== null && tolerance !== undefined && tolerance > 0) {
      const thresholdY = yFor(tolerance);
      ctx.strokeStyle = '#eab765'; ctx.lineWidth = 1.2; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(plot.left, thresholdY); ctx.lineTo(plot.right, thresholdY); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = '#eab765'; ctx.textAlign = 'right'; ctx.fillText(`閾値 ${fmtSci(tolerance)}`, plot.right, thresholdY - 4);
    }

    seriesList.forEach((series) => {
      ctx.strokeStyle = series.color; ctx.fillStyle = series.color; ctx.lineWidth = 1.8; ctx.setLineDash([]);
      let previous = null;
      history.forEach((row) => {
        const raw = nullableNumber(row[series.key]);
        if (raw === null || raw < 0) { previous = null; return; }
        const point = { x: xFor(row), y: yFor(raw), value: raw, eligible: row.monitor_eligible };
        if (previous) { ctx.beginPath(); ctx.moveTo(previous.x, previous.y); ctx.lineTo(point.x, point.y); ctx.stroke(); }
        ctx.globalAlpha = point.eligible === false ? .58 : 1;
        ctx.beginPath(); ctx.arc(point.x, point.y, raw === 0 ? 3 : 2, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = 1;
        previous = point;
      });
    });

    ctx.font = `9px ${fontFamily}`; ctx.textAlign = 'left';
    let legendX = plot.left;
    seriesList.forEach((series) => {
      ctx.fillStyle = series.color; ctx.fillRect(legendX, 12, 15, 2); ctx.fillText(series.label, legendX + 20, 15); legendX += Math.max(72, ctx.measureText(series.label).width + 34);
    });
    if (tolerance !== null && tolerance !== undefined && tolerance > 0 && seriesList.length) {
      ctx.fillStyle = '#eab765'; ctx.fillRect(legendX, 12, 15, 2); ctx.fillText('閾値', legendX + 20, 15);
    }
    ctx.fillStyle = '#719094'; ctx.textAlign = 'left'; ctx.fillText(`step ${fmt(Math.min(...history.map((row) => row.step)), 0)}`, plot.left, plot.bottom + 19);
    ctx.textAlign = 'right'; ctx.fillText(`step ${fmt(Math.max(...history.map((row) => row.step)), 0)}`, plot.right, plot.bottom + 19);
  }

  function drawAll() { drawGeometry(); drawResult(); if (state.activeDock === 'history') drawHistoryCanvas(); }

  function bindGeometryControls() {
    const view = $('#geometryView');
    if (!view) return;
    view.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      state.drag = { x: event.clientX, y: event.clientY, yaw: state.orbit.yaw, pitch: state.orbit.pitch, moved: false, pointerId: event.pointerId };
      view.classList.add('dragging');
      view.setPointerCapture?.(event.pointerId);
    });
    view.addEventListener('pointermove', (event) => {
      if (!state.drag || state.drag.pointerId !== event.pointerId) return;
      const deltaX = event.clientX - state.drag.x; const deltaY = event.clientY - state.drag.y;
      if (Math.hypot(deltaX, deltaY) > 6) state.drag.moved = true;
      state.orbit.yaw = state.drag.yaw + deltaX * .01;
      state.orbit.pitch = clamp(state.drag.pitch + deltaY * .01, -.95, .95);
      drawGeometry();
    });
    const endDrag = (event) => {
      if (!state.drag || state.drag.pointerId !== event.pointerId) return;
      const drag = state.drag;
      if (Math.hypot(event.clientX - drag.x, event.clientY - drag.y) > 6) drag.moved = true;
      state.drag = null; view.classList.remove('dragging'); view.releasePointerCapture?.(event.pointerId);
      if (drag && !drag.moved && event.type !== 'pointercancel') pickSurface(event.clientX, event.clientY, view);
    };
    view.addEventListener('pointerup', endDrag); view.addEventListener('pointercancel', endDrag); view.addEventListener('pointerleave', (event) => { if (state.drag && event.buttons === 0) endDrag(event); });
    view.addEventListener('wheel', (event) => { event.preventDefault(); state.orbit.zoom = clamp(state.orbit.zoom * (event.deltaY < 0 ? 1.08 : .92), .35, 3.2); drawGeometry(); }, { passive: false });
  }

  function pointInTriangleDepth(x, y, first, second, third) {
    const denominator = (second[1] - third[1]) * (first[0] - third[0]) + (third[0] - second[0]) * (first[1] - third[1]);
    if (Math.abs(denominator) < 1e-9) return null;
    const firstWeight = ((second[1] - third[1]) * (x - third[0]) + (third[0] - second[0]) * (y - third[1])) / denominator;
    const secondWeight = ((third[1] - first[1]) * (x - third[0]) + (first[0] - third[0]) * (y - third[1])) / denominator;
    const thirdWeight = 1 - firstWeight - secondWeight;
    const epsilon = 1e-7;
    if (firstWeight < -epsilon || secondWeight < -epsilon || thirdWeight < -epsilon) return null;
    return firstWeight * first[2] + secondWeight * second[2] + thirdWeight * third[2];
  }

  function projectedFaceDepthAtPoint(face, x, y) {
    const vertices = Array.isArray(face.vertices) ? face.vertices : [];
    if (vertices.length < 3) return null;
    let depth = null;
    // CAD faces are triangles; the fan also supports the box preview's
    // quadrilateral faces without reintroducing a centroid hit radius.
    for (let index = 1; index < vertices.length - 1; index += 1) {
      const candidate = pointInTriangleDepth(x, y, vertices[0], vertices[index], vertices[index + 1]);
      if (candidate !== null && (depth === null || candidate > depth)) depth = candidate;
    }
    return depth;
  }

  function pickSurface(clientX, clientY, view) {
    if (!state.projectedFaces.length) return;
    const rect = view.getBoundingClientRect(); const x = clientX - rect.left; const y = clientY - rect.top;
    let picked = null;
    state.projectedFaces.forEach((face) => {
      if (!face.groupId) return;
      const depth = projectedFaceDepthAtPoint(face, x, y);
      // Faces are drawn in ascending depth order, so the largest interpolated
      // depth is the frontmost surface when projected faces overlap.
      if (depth !== null && (!picked || depth >= picked.depth)) picked = { ...face, depth };
    });
    if (picked?.groupId) {
      state.selectedSurface = picked.groupId;
      state.selectedNode = `face:cad:${picked.groupId}`;
      if (BOX_FACE_GROUPS.includes(picked.groupId)) state.selectedNode = `face:${picked.groupId}`;
      switchView('geometry');
      renderTree();
      renderInspector();
      drawGeometry();
      pushLog(`${BOX_FACE_GROUPS.includes(picked.groupId) ? '境界面' : 'CAD 表面'}「${faceLabel(BOX_FACE_GROUPS.includes(picked.groupId) ? picked.groupId : `cad:${picked.groupId}`)}」を選択しました。`);
    }
  }

  function bindCadDrop() {
    const input = $('#cadFile'); const zone = $('#cadDropZone');
    if (!input || !zone) return;
    input.addEventListener('change', () => setCadFile(input.files?.[0]));
    ['dragenter', 'dragover'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.add('dragover'); }));
    ['dragleave', 'drop'].forEach((eventName) => zone.addEventListener(eventName, (event) => { event.preventDefault(); zone.classList.remove('dragover'); }));
    zone.addEventListener('drop', (event) => setCadFile(event.dataTransfer?.files?.[0]));
  }

  function setCadFile(file) {
    if (!file) return;
    const allowed = /\.(stl|obj|step?|stp|iges?|igs|brep)$/i;
    if (!allowed.test(file.name)) { toast('対応形式は STL / OBJ / STEP / IGES / BREP です。', 'warn'); return; }
    state.cadFile = file;
    $('#cadFileName').textContent = `${file.name} · ${Math.ceil(file.size / 1024)} KB`;
    $('#cadFileName').classList.remove('hidden');
  }

  function bindGlobalEvents() {
    document.addEventListener('click', handleClick);
    document.addEventListener('input', handleInput);
    document.addEventListener('change', handleChange);
    $('#projectFile')?.addEventListener('change', (event) => { importProjectFile(event.target.files?.[0]); event.target.value = ''; });
    $('#materialCsvFile')?.addEventListener('change', (event) => { handleMaterialCsvFile(event.target.files?.[0]); event.target.value = ''; });
    bindGeometryControls(); bindCadDrop();
    window.addEventListener('resize', drawAll);
    if ('ResizeObserver' in window) {
      const observer = new ResizeObserver(() => drawAll());
      ['#geometryCanvas', '#resultCanvas', '#historyCanvas'].forEach((selector) => { const element = $(selector); if (element) observer.observe(element.parentElement || element); });
    }
    $('#projectDialog')?.addEventListener('cancel', () => closeDialog('projectDialog'));
    $('#cadDialog')?.addEventListener('cancel', () => closeDialog('cadDialog'));
    $('#materialCsvDialog')?.addEventListener('cancel', () => invalidateMaterialCsvPreview(false));
  }

  function boot() {
    bindGlobalEvents();
    renderLogs(); renderValidationDock();
    loadInitial();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', boot);
  else boot();
})();
