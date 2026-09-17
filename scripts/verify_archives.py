"""Restore the portable examples into isolated temporary stores and verify artifacts."""
import tempfile
from pathlib import Path
import numpy as np
from workbench.server import Store, read_json
from workbench.geometry import load_asset_metadata


def main():
    for filename in ('heated_channel.xlb.zip', 'les_conjugate_gpu.xlb.zip', 'rotated_cad_gpu.xlb.zip'):
        source = Path('examples') / filename
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(temporary)
            try:
                restored = store.import_archive(source.read_bytes())
                project = restored['project']
                runs = store.project(restored['id'])['runs']
                assert runs and all(run['status'] == 'completed' for run in runs)
                for run in runs:
                    directory = store.runs / run['id']
                    snapshot = read_json(directory / 'input.json')
                    if snapshot['geometry'].get('asset_id'):
                        asset = load_asset_metadata(store.assets, snapshot['geometry']['asset_id'])
                        selectors = {'cad:' + group['id'] for group in asset['surface_groups']}
                        assert all(boundary['face'] in selectors for boundary in snapshot['boundaries']
                                   if boundary['face'].startswith('cad:'))
                    with np.load(directory / 'fields.npz', allow_pickle=False) as data:
                        assert np.isfinite(data['temperature']).all()
                        if 'conjugate' in filename:
                            assert data['solid_mask'].sum() == 400
                            assert data['eddy_viscosity'].max() > 0
                    assert (directory / 'provenance.json').exists()
                print(filename, 'restored:', len(runs), 'run(s)', flush=True)
            finally:
                store.pool.shutdown(wait=True)


if __name__ == '__main__':
    main()
