"""Real CUDA checks of bounded collision, coupled fields and RAM restart."""
import copy
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

from scripts.verify_transient_solver import make_project, make_mesh, COMPARISON_TOLERANCES
from workbench.solver import simulate


def main():
    os.environ['XLB_COMPUTE_BACKEND'] = 'auto'
    evidence = {'cases': []}
    with tempfile.TemporaryDirectory(prefix='ram-gpu-') as tmp:
        root = Path(tmp)
        for name, flow, thermal in [('les_cht', True, True), ('thermal_only', False, True), ('flow_only', True, False)]:
            p = make_project(device='cpu', flow=flow, thermal=thermal)
            p['mesh']['cells'] = [24, 12, 12]
            p['study']['dt'] = 0.0003
            mesh = make_mesh(p, root / 'assets')
            simulate(p, mesh, root / name / 'cpu')
            p['study'].update(device='cuda:0-ram', gpu_batch_cells=1024)
            started = time.perf_counter()
            ram = simulate(p, mesh, root / name / 'ram')
            elapsed = time.perf_counter() - started
            assert ram['status'] == 'completed'
            assert 'cpu' in ram['diagnostics']['state_device'].lower()
            if flow:
                assert 'cuda' in ram['diagnostics']['collision_device'].lower()
            errors = {}
            with np.load(root / name / 'cpu' / 'fields.npz') as cpu, np.load(root / name / 'ram' / 'fields.npz') as actual:
                for field, tolerances in COMPARISON_TOLERANCES.items():
                    np.testing.assert_allclose(actual[field], cpu[field], **tolerances)
                    errors[field] = float(np.max(np.abs(actual[field] - cpu[field])))
            first = copy.deepcopy(p)
            first['study']['steps'] = 3
            simulate(first, mesh, root / name / 'first')
            resumed = simulate(p, mesh, root / name / 'resumed', restart_from=root / name / 'first' / 'restart.npz')
            for artifact, fields in [('restart.npz', ('f', 'temperature')), ('fields.npz', tuple(COMPARISON_TOLERANCES))]:
                with np.load(root / name / 'ram' / artifact) as expected, np.load(root / name / 'resumed' / artifact) as actual:
                    for field in fields:
                        np.testing.assert_array_equal(actual[field], expected[field])
            frames = json.loads((root / name / 'resumed' / 'frames.json').read_text())['frames']
            assert [frame['step'] for frame in frames] == [3, 4, 6, 7]
            ram_frames = json.loads((root / name / 'ram' / 'frames.json').read_text())['frames']
            cpu_frames = json.loads((root / name / 'cpu' / 'frames.json').read_text())['frames']
            assert [f['step'] for f in ram_frames] == [f['step'] for f in cpu_frames]
            for frame in ram_frames:
                with np.load(root / name / 'ram' / 'frames' / frame['file']) as actual, np.load(root / name / 'cpu' / 'frames' / frame['file']) as expected:
                    for field, tolerances in COMPARISON_TOLERANCES.items():
                        np.testing.assert_allclose(actual[field], expected[field], **tolerances)
            p['study']['gpu_batch_cells'] = 2048
            simulate(p, mesh, root / name / 'resized', restart_from=root / name / 'first' / 'restart.npz')
            with np.load(root / name / 'ram' / 'fields.npz') as expected, np.load(root / name / 'resized' / 'fields.npz') as actual:
                for field, tolerances in COMPARISON_TOLERANCES.items():
                    np.testing.assert_allclose(actual[field], expected[field], **tolerances)
            evidence['cases'].append({'case': name, 'cells': int(mesh['fluid_mask'].size), 'seconds': elapsed,
                                      'cpu_max_absolute_differences': errors, 'restart_exact': True,
                                      'changed_batch_cells': 2048, 'changed_batch_agrees_within_tolerance': True,
                                      'diagnostics': ram['diagnostics']})
        evidence['peak_host_rss_mib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    Path(sys.argv[1]).write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
