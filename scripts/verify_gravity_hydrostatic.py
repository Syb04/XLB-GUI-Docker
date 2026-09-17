"""Closed-box uniform gravity should approach dp/dz = rho*g_z."""
import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
from workbench.schema import default_project, validate_project
from workbench.geometry import build_mesh
from workbench.solver import simulate


def main():
    os.environ['XLB_COMPUTE_BACKEND'] = 'device'
    p = default_project()
    p['geometry']['size'] = [.04, .02, .04]
    p['mesh'] = {'cells': [8, 4, 8]}
    p['boundaries'] = []
    p['physics'].update(thermal=False, initial_temperature=300., gravity={'enabled': True, 'mode': 'uniform', 'vector': [0., 0., -.1], 'reference_temperature': 300.})
    p['materials'][0]['density'] = {'kind': 'constant', 'value': 1000.}
    p['materials'][0]['viscosity'] = {'kind': 'constant', 'value': 1.}
    p['study'].update(device='cpu', steps=1500, output_interval=300, snapshot_interval=0, dt=.001)
    p = validate_project(p)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mesh = build_mesh(p, root)
        result = simulate(p, mesh, root, on_progress=lambda row: print(row['step'], flush=True))
        with np.load(root / 'fields.npz') as fields:
            pressure = fields['pressure'].mean(axis=(0, 1))
            z = (np.arange(8) + .5) * .005
            slope = float(np.polyfit(z, pressure, 1)[0])
            maximum_speed = float(np.linalg.norm(fields['velocity'], axis=0).max())
            relative_error = abs((slope - (-100.)) / 100.)
            assert relative_error < .03, (slope, pressure)
            assert maximum_speed < 1e-5, maximum_speed
            evidence = {'steps': 1500, 'expected_dp_dz_pa_m': -100., 'measured_dp_dz_pa_m': slope,
                        'relative_error': relative_error, 'maximum_speed_m_s': maximum_speed,
                        'z_m': z.tolist(), 'pressure_pa': pressure.tolist(), 'diagnostics': result['diagnostics']}
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
