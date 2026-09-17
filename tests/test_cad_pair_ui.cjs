// Exercise paired CAD normalization, serialization, and preview layering
// without starting browser I/O.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const marker = "  if (document.readyState === 'loading')";
assert(source.includes(marker));
const instrumented = source.replace(marker,
  '  globalThis.cadPairTest = { normalizeProject, serializableProject, geometryPoints, state };\n' + marker);
const context = {
  document: { readyState: 'loading', addEventListener() {} },
  window: {},
  structuredClone
};
vm.runInNewContext(instrumented, context);
const api = context.cadPairTest;

const project = api.normalizeProject({
  name: 'paired CAD',
  geometry: {
    kind: 'cad', role: 'fluid', size: [0.12, 0.01, 0.01],
    asset_id: 'fluid-asset', solid_asset_id: 'solid-asset', solid_material_id: 'copper'
  },
  materials: [{ id: 'water', name: 'Water' }, { id: 'copper', name: 'Copper' }],
  physics: { material_id: 'water' },
  boundaries: [],
  mesh: { cells: [12, 4, 4] },
  study: { steps: 10, output_interval: 1, dt: 0.001, device: 'cpu', gpu_batch_cells: 1024 }
});
api.state.project = project;
const payload = api.serializableProject(project);
assert.equal(payload.geometry.asset_id, 'fluid-asset');
assert.equal(payload.geometry.solid_asset_id, 'solid-asset');
assert.equal(payload.geometry.solid_material_id, 'copper');

api.state.cadAssetMeta = {
  id: 'fluid-asset', vertices: [[0, 0, 0], [0.12, 0, 0], [0, 0.01, 0]], faces: [[0, 1, 2]]
};
api.state.cadSolidAssetMeta = {
  id: 'solid-asset', vertices: [[0, 0, 0], [0.12, 0, 0], [0, 0.06, 0]], faces: [[0, 1, 2]]
};
const preview = api.geometryPoints();
assert.equal(preview.layers.length, 2);
assert.deepEqual(Array.from(preview.layers, (layer) => layer.role), ['fluid', 'solid']);
console.log('CAD pair UI: solid_asset_id round-trip and two preview layers OK');
