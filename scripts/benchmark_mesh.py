"""Representative 3M-cell box mesh/preview benchmark; does not run the solver.

Run inside Linux/Docker: python -m scripts.benchmark_mesh
"""
import json
import resource
import tempfile
import time
from workbench.geometry import build_mesh
from workbench.schema import default_project
from workbench.server import mesh_preview


def main():
    project = default_project()
    project['mesh']['cells'] = [300, 100, 100]
    start = time.perf_counter()
    with tempfile.TemporaryDirectory() as assets:
        mesh = build_mesh(project, assets)
    mesh_seconds = time.perf_counter() - start
    start = time.perf_counter()
    preview = mesh_preview(mesh)
    preview_seconds = time.perf_counter() - start
    assert preview['fluid_cells'] == 3_000_000
    assert len(preview['preview']['points']) <= 6000
    print(json.dumps({
        'case': '300 x 100 x 100 box; mesh only, no solver',
        'mesh_seconds': mesh_seconds, 'preview_seconds': preview_seconds,
        'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        'boundary_array_logical_mib': sum(v.nbytes for v in mesh['boundary_links'].values()) / 1024**2,
        'total_cells': preview['total_cells'],
        'preview_points': len(preview['preview']['points']),
        'limits': preview['limits'],
    }, indent=2), flush=True)


if __name__ == '__main__':
    main()
