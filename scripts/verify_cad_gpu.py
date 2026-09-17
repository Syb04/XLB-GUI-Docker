"""GPU execution on a rotated CAD fluid volume with individual inlet/outlet/heated patches."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
import numpy as np
import trimesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', default='docs/cad-gpu-verification.json')
    args = parser.parse_args()

    def request(path, data=None, raw=False):
        if data is not None and not isinstance(data, bytes):
            data = json.dumps(data).encode()
        with urlopen(Request(args.url + path, data=data), timeout=120) as response:
            return response.read() if raw else json.load(response)

    angle = np.deg2rad(23.)
    direction = np.array([np.cos(angle), np.sin(angle), 0.])
    cad = trimesh.creation.box(extents=[.06, .02, .02])
    cad.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
    asset = request('/api/import?filename=rotated-channel.stl&unit=m', cad.export(file_type='stl'))
    groups = asset['surface_groups']
    assert len(groups) == 6, groups
    inlet = min(groups, key=lambda group: np.dot(group['normal'], direction))
    outlet = max(groups, key=lambda group: np.dot(group['normal'], direction))
    heated = max(groups, key=lambda group: group['normal'][2])
    project = request('/api/default')
    project['name'] = '斜めCAD・面別境界 GPU検証'
    project['geometry'].update(kind='cad', asset_id=asset['id'], role='fluid')
    project['mesh']['cells'] = [24, 18, 10]
    project['study'].update(steps=60, output_interval=10, dt=.005, device='cuda:0')
    project['physics']['turbulence'] = {'model': 'smagorinsky', 'smagorinsky_constant': .17,
                                      'turbulent_prandtl': .9}
    project['boundaries'] = [
        {'id': 'inlet', 'name': '斜め入口', 'face': 'cad:' + inlet['id'],
         'flow': {'type': 'velocity', 'velocity': (direction * .01).tolist()},
         'thermal': {'type': 'temperature', 'value': 293.15}},
        {'id': 'outlet', 'name': '斜め出口', 'face': 'cad:' + outlet['id'],
         'flow': {'type': 'pressure', 'value': 0.}, 'thermal': {'type': 'adiabatic'}},
        {'id': 'heater', 'name': '加熱CAD面', 'face': 'cad:' + heated['id'],
         'flow': {'type': 'wall'}, 'thermal': {'type': 'heat_flux', 'value': 1000.}},
    ]
    mesh = request('/api/mesh', project)
    for boundary in project['boundaries']:
        assert mesh['boundary_link_counts'][boundary['face']] > 0
    run = request('/api/runs', {'project': project})
    print('CAD GPU run', run['id'], flush=True)
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        run = request('/api/runs/' + run['id'])
        if run['status'] not in ('queued', 'running', 'stopping'):
            break
        time.sleep(1)
    assert run['status'] == 'completed', run
    assert any(label in run['diagnostics']['device_actual'].lower() for label in ('cuda', 'gpu')), run['diagnostics']
    with np.load(io.BytesIO(request('/api/runs/' + run['id'] + '/files/fields.npz', raw=True))) as fields:
        assert np.linalg.norm(fields['velocity'], axis=0).max() > .005
        assert fields['temperature'][fields['fluid_mask']].max() > 293.151
        stats = {'speed_max': float(np.linalg.norm(fields['velocity'], axis=0).max()),
                 'temperature_max': float(fields['temperature'][fields['fluid_mask']].max())}
    evidence = {'rotation_degrees': 23, 'asset_id': asset['id'], 'surface_groups': groups,
                'mesh': {k: v for k, v in mesh.items() if k != 'preview'}, 'run': run, 'fields': stats}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    archive = request('/api/projects/' + run['project_id'] + '/export', raw=True)
    Path('examples/rotated_cad_gpu.xlb.zip').write_bytes(archive)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
