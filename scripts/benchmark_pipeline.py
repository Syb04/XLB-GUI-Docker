"""Run a repeatable coupled LES/CHT case inside a selected Docker image.

Compare cold first progress interval, subsequent intervals, and end-to-end
time separately; full outputs are retained for independent numerical checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

from workbench.schema import default_project, validate_project
from workbench.geometry import build_mesh
from workbench.solver import simulate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cells', type=int, default=50, help='Y/Z cells; X is 3 times this')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    p = default_project()
    p['name'] = 'GPU pipeline LES/CHT benchmark'
    p['mesh']['cells'] = [args.cells * 3, args.cells, args.cells]
    p['materials'].append({
        'id': 'steel', 'name': 'Steel',
        'density': {'kind': 'constant', 'value': 7800},
        'viscosity': {'kind': 'constant', 'value': .001},
        'heat_capacity': {'kind': 'constant', 'value': 500},
        'conductivity': {'kind': 'table', 'points': [[273.15, 14.], [373.15, 16.]]},
    })
    p['geometry']['solids'] = [{
        'id': 'heated-block', 'name': 'Heated solid', 'origin': [.01, 0, .016],
        'size': [.04, .02, .004], 'material_id': 'steel',
    }]
    p['physics']['turbulence'] = {'model': 'smagorinsky',
                                'smagorinsky_constant': .17, 'turbulent_prandtl': .9}
    p['study'].update(steps=args.steps, output_interval=args.interval,
                      dt=.05 / args.cells, device='cuda:0')
    # Keep the historical compute-only comparison, including older images
    # whose schema does not yet accept snapshot_interval.
    p['study'].pop('snapshot_interval', None)
    for bc in p['boundaries']:
        if bc['face'] == 'zmax':
            bc['thermal'] = {'type': 'temperature', 'value': 333.15}
    p = validate_project(p)
    (root / 'input.json').write_text(json.dumps(p, indent=2), encoding='utf-8')
    t0 = time.perf_counter()
    mesh = build_mesh(p, root / 'assets')
    mesh_seconds = time.perf_counter() - t0
    samples = []
    t0 = time.perf_counter()

    def report(sample):
        sample = dict(sample, elapsed_seconds=time.perf_counter() - t0)
        samples.append(sample)
        print(json.dumps(sample), flush=True)

    result = simulate(p, mesh, root, on_progress=report)
    elapsed = time.perf_counter() - t0
    assert result['status'] == 'completed', result
    assert 'cuda' in result['diagnostics']['device_actual'].lower(), result
    interval_times = [(b['elapsed_seconds'] - a['elapsed_seconds']) / (b['step'] - a['step'])
                      for a, b in zip(samples[1:], samples[2:]) if b['step'] > a['step']]
    source = Path(__file__).resolve().parents[1] / 'workbench'
    # The benchmark scripts may be mounted separately from the installed code.
    import workbench.solver
    source = Path(workbench.solver.__file__).parent
    summary = {'cells': p['mesh']['cells'], 'total_cells': int(mesh['fluid_mask'].size),
               'mesh_seconds': mesh_seconds, 'simulate_including_output_seconds': elapsed,
               'first_progress_interval_seconds': samples[1]['elapsed_seconds'] - samples[0]['elapsed_seconds'],
               'warm_seconds_per_step': sum(interval_times) / len(interval_times) if interval_times else None,
               'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
               'samples': samples, 'result': result,
               'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source.glob('*.py')}}
    (root / 'benchmark.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('samples','result','source_sha256')}), flush=True)


if __name__ == '__main__':
    main()
