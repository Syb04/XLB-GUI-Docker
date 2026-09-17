"""Compare actual CUDA continuous and split LES/CHT evolution and checkpoints.

Run inside the CUDA image with --gpus all; writes only the requested evidence.
"""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from scripts.verify_transient_solver import make_project, make_mesh
from workbench.solver import simulate
from workbench.restart import read_restart_metadata


def main():
    os.environ['XLB_COMPUTE_BACKEND'] = 'device'
    evidence = {'hardware': 'cuda:0', 'cases': []}
    with tempfile.TemporaryDirectory(prefix='restart-gpu-') as temp:
        root = Path(temp)
        for mode, flow, thermal in [('les_cht', True, True), ('thermal_only', False, True), ('flow_only', True, False)]:
            project = make_project(device='cuda:0', flow=flow, thermal=thermal)
            project['name'] = 'Restart CUDA verification ' + mode
            mesh = make_mesh(project, root / 'assets')
            continuous = root / mode / 'continuous'
            first = root / mode / 'first'
            resumed = root / mode / 'resumed'
            uninterrupted = simulate(project, mesh, continuous)
            initial = copy.deepcopy(project)
            initial['study']['steps'] = 3
            simulate(initial, mesh, first)
            continued = simulate(project, mesh, resumed, restart_from=first / 'restart.npz')
            assert uninterrupted['status'] == continued['status'] == 'completed'
            assert uninterrupted['steps'] == continued['steps'] == 7
            assert 'cuda' in str(continued['diagnostics']['device_actual']).lower()
            errors = {}
            for artifact, fields in [('restart.npz', ('f', 'temperature')),
                                     ('fields.npz', ('velocity', 'pressure', 'temperature', 'eddy_viscosity'))]:
                with np.load(continuous / artifact) as expected, np.load(resumed / artifact) as actual:
                    for field in fields:
                        # A split should preserve the exact same state on the
                        # same hardware/backend, including FP64 thermal state.
                        np.testing.assert_array_equal(actual[field], expected[field])
                        errors[artifact + ':' + field] = float(np.max(np.abs(actual[field] - expected[field])))
            frames = json.loads((resumed / 'frames.json').read_text())['frames']
            assert [entry['step'] for entry in frames] == [3, 4, 6, 7]
            metadata = read_restart_metadata(resumed / 'restart.npz')
            evidence['cases'].append({'mode': mode, 'continuous_steps': 7, 'split_steps': [3, 4],
                                      'checkpoint_step': metadata['step'], 'end_time_s': metadata['step'] * metadata['dt'],
                                      'max_absolute_errors': errors, 'diagnostics': continued['diagnostics']})
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
