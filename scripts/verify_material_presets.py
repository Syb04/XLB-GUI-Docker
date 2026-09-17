"""Exercise bundled presets and their CSV equivalents on a real CUDA device."""
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from workbench.geometry import build_mesh
from workbench.material_csv import parse_material_csv
from workbench.schema import default_project, validate_project
from workbench.solver import simulate


def main():
    os.environ['XLB_COMPUTE_BACKEND'] = 'device'
    root = Path(__file__).resolve().parents[1]
    catalog = json.loads((root / 'workbench/data/material_presets.json').read_text())['presets']
    evidence = []
    with tempfile.TemporaryDirectory() as temporary:
        for preset in catalog:
            if not preset['id'].endswith('-temperature'):
                continue
            fluid = preset['id'].split('-')[0]
            imported = parse_material_csv((root / f'static/examples/{fluid}-temperature.csv').read_text())
            material = {**preset['material'], 'id': 'fluid'}
            for key, prop in imported['properties'].items():
                assert prop == material[key], f'{fluid} CSV and catalog differ: {key}'
            project = default_project()
            project['name'] = f'{fluid} temperature-dependent property verification'
            project['geometry']['size'] = [.04, .02, .02]
            project['mesh'] = {'cells': [8, 4, 4]}
            project['materials'] = [material]
            project['physics']['material_id'] = 'fluid'
            project['study'].update(steps=3, output_interval=1, snapshot_interval=0, dt=.001, device='cuda:0')
            project = validate_project(project)
            mesh = build_mesh(project, Path(temporary) / 'assets')
            run = Path(temporary) / fluid
            result = simulate(project, mesh, run)
            assert result['status'] == 'completed'
            assert result['diagnostics']['device_actual'] == 'cuda:0'
            with np.load(run / 'fields.npz') as fields:
                for field in ('temperature', 'velocity', 'pressure'):
                    assert np.isfinite(fields[field]).all()
            evidence.append({'preset': preset['id'], 'csv_rows': imported['rows'],
                             'steps': result['steps'], 'status': result['status'],
                             'diagnostics': result['diagnostics']})
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps([{'preset': entry['preset'], 'steps': entry['steps'], 'status': entry['status'],
                       'device': entry['diagnostics']['device_actual']} for entry in evidence], indent=2))


if __name__ == '__main__':
    main()
