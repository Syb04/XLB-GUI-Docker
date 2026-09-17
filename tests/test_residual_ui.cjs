// Exercise actual history parsing/selection code without starting browser I/O.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const marker = "  if (document.readyState === 'loading')";
assert(source.includes(marker));
const instrumented = source.replace(marker,
  '  globalThis.monitorTest = { normalizeRun, normalizeHistory, parseHistoryCsv, isNativeMonitorRow, monitorState, state };\n' + marker);
const context = { document: { readyState: 'loading', addEventListener() {} }, structuredClone };
vm.runInNewContext(instrumented, context);
const api = context.monitorTest;
for (const rows of [api.normalizeHistory([[0, 0, 300, 300]]), api.parseHistoryCsv('step,residual,temperature_residual,temperature_min,temperature_max\n0,0,0,300,300\n3,0.2,0.001,300,301')]) {
  assert(rows.every(row => !api.isNativeMonitorRow(row)));
  api.state.history = rows;
  assert.equal(api.monitorState({}).legacy, true);
  assert.equal(rows[rows.length-1].time ?? null, null);
}
const rows = api.parseHistoryCsv('step,time,residual,velocity_residual,pressure_residual,temperature_residual,velocity_change_max,monitor_tolerance,monitor_required_samples,monitor_consecutive_samples,monitor_satisfied,monitor_eligible,residual_kind\n0,0,,,,,,0.001,3,0,False,False,sample_change_linf_v1\n2,0.02,0,0,0,0,0,0.001,3,3,True,True,sample_change_linf_v1');
assert.equal(rows[0].residual, null);
assert.equal(rows[0].velocity_residual, null);
assert.equal(rows[1].velocity_change_max, 0);
assert.equal(rows[1].monitor_satisfied, true);
api.state.history = rows;
api.state.project = {study: {monitor: { tolerance: 1e-10, consecutive_samples: 100 }}};
const state = api.monitorState({});
assert.equal(state.settings.tolerance, .001);
assert.equal(state.settings.requiredSamples, 3);
assert.equal(state.satisfied, true);
api.state.run = {id: 'new', monitor_satisfied: true, monitor_tolerance: 0.001, residual_kind: 'sample_change_linf_v1'};
assert.equal(api.normalizeRun({id:'legacy', status:'completed', residual:0.2}).monitor_satisfied, undefined);
console.log('Residual UI: legacy CSV, baseline/null, zero, booleans and immutable run settings OK');
