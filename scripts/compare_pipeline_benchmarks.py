"""Compare retained before/after fields and produce a compact benchmark report."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads((path / 'benchmark.json').read_text())
                     for path in (args.before, args.after)]
    inputs = [json.loads((path / 'input.json').read_text()) for path in (args.before, args.after)]
    assert inputs[0] == inputs[1], 'Benchmarks used different inputs'
    comparisons = {}
    with np.load(args.before / 'fields.npz') as a, np.load(args.after / 'fields.npz') as b:
        for name in ('fluid_mask', 'solid_mask', 'thermal_mask', 'material_index'):
            np.testing.assert_array_equal(a[name], b[name])
        for name, tolerance in [('temperature', .001), ('velocity', 2e-5),
                                ('pressure', .05), ('eddy_viscosity', 1e-7)]:
            assert np.isfinite(b[name]).all(), name
            np.testing.assert_allclose(a[name], b[name], rtol=0 if name == 'temperature' else .003,
                                       atol=tolerance, err_msg=name)
            comparisons[name] = {'max_absolute_difference': float(np.max(np.abs(a[name] - b[name])))}
    output = {
        'purpose': 'Same RTX3060/input; warm progress intervals include scalar checks and output summaries. Not turbulence accuracy validation.',
        'total_cells': before['total_cells'],
        'warm_step_speedup': before['warm_seconds_per_step'] / after['warm_seconds_per_step'],
        'end_to_end_speedup': before['simulate_including_output_seconds'] / after['simulate_including_output_seconds'],
        'fields': comparisons,
        'before': before,
        'after': after,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key:value for key,value in output.items() if key not in ('before', 'after')}, indent=2))


if __name__ == '__main__':
    main()
