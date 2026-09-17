"""Exercise real HTTP runs on CPU and CUDA and save reviewable verification evidence.

Usage: python scripts/verify_gpu.py --url http://127.0.0.1:8766
Requires NumPy on the client; server must have XLB and CUDA-enabled JAX.
"""
from __future__ import annotations

import argparse
import copy
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', default='docs/gpu-verification.json')
    args = parser.parse_args()

    def request(path, body=None, raw=False):
        payload = json.dumps(body).encode() if body is not None else None
        with urlopen(Request(args.url + path, data=payload,
                             headers={'Content-Type': 'application/json'}), timeout=60) as response:
            return response.read() if raw else json.load(response)

    project = request('/api/default')
    project['name'] = 'LES・共役熱伝達 GPU検証'
    project['materials'].append({
        'id': 'steel', 'name': 'Steel',
        'density': {'kind': 'constant', 'value': 7800},
        'viscosity': {'kind': 'constant', 'value': .001},
        'heat_capacity': {'kind': 'constant', 'value': 500},
        'conductivity': {'kind': 'table', 'points': [[273.15, 14.], [373.15, 16.]]},
    })
    project['geometry']['solids'] = [{
        'id': 'heated-block', 'name': '加熱固体', 'origin': [.01, 0, .016],
        'size': [.04, .02, .004], 'material_id': 'steel',
    }]
    project['physics']['turbulence'] = {
        'model': 'smagorinsky', 'smagorinsky_constant': .17, 'turbulent_prandtl': .9,
    }
    project['study'].update(steps=100, output_interval=20, snapshot_interval=0, dt=.005)
    for boundary in project['boundaries']:
        if boundary['face'] == 'zmax':
            boundary['thermal'] = {'type': 'temperature', 'value': 333.15}
    mesh = request('/api/mesh', project)
    assert mesh['solid_cells'] > 0 and mesh['fluid_cells'] > 0, mesh
    fields, runs = {}, {}
    for device in ('cpu', 'cuda:0'):
        current = copy.deepcopy(project)
        current['name'] += ' / ' + device
        current['study']['device'] = device
        run = request('/api/runs', {'project': current})
        print(device, run['id'], flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            run = request('/api/runs/' + run['id'])
            if run['status'] not in ('queued', 'running', 'stopping'):
                break
            time.sleep(1)
        assert run['status'] == 'completed', run
        actual_device = run['diagnostics']['device_actual'].lower()
        assert ('cuda' in actual_device or 'gpu' in actual_device) if device == 'cuda:0' else 'cpu' in actual_device, actual_device
        if device == 'cuda:0':
            assert run['diagnostics']['compute_backend'] == 'device', run['diagnostics']
            assert run['diagnostics']['full_field_transfers'] == 1, run['diagnostics']
            assert run['diagnostics']['thermal_backend'] == 'JAX finite-volume on device', run['diagnostics']
        runs[device] = run
        data = request('/api/runs/' + run['id'] + '/files/fields.npz', raw=True)
        with np.load(io.BytesIO(data), allow_pickle=False) as stored:
            fields[device] = {key: stored[key] for key in stored.files}
        assert np.isfinite(fields[device]['temperature']).all()
        assert fields[device]['solid_mask'].any()
        solid_t = fields[device]['temperature'][fields[device]['solid_mask']]
        assert solid_t.max() > project['physics']['initial_temperature'] + .01
        assert fields[device]['eddy_viscosity'].max() > 0
        print(device, 'completed', run['diagnostics'], flush=True)
    comparison = {}
    for field in ('temperature', 'velocity', 'pressure', 'eddy_viscosity'):
        a, b = fields['cpu'][field], fields['cuda:0'][field]
        comparison[field] = {'max_absolute_difference': float(np.max(np.abs(a - b)))}
        np.testing.assert_allclose(a, b, rtol=0 if field == 'temperature' else 3e-3, atol={'temperature': .001, 'velocity': 2e-5,
                                                      'pressure': .05, 'eddy_viscosity': 1e-7}[field])
    summary = {'purpose': 'Execution and CPU/CUDA agreement; not turbulence benchmark validation',
               'mesh': {k: v for k, v in mesh.items() if k != 'preview'},
               'runs': runs, 'comparison': comparison}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = request('/api/projects/' + runs['cuda:0']['project_id'] + '/export', raw=True)
    example = destination.parent.parent / 'examples' / 'les_conjugate_gpu.xlb.zip'
    example.parent.mkdir(parents=True, exist_ok=True)
    example.write_bytes(archive)
    print(json.dumps(comparison, indent=2), flush=True)


if __name__ == '__main__':
    main()
