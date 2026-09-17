import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np

from workbench.schema import default_project
from workbench.server import Store, atomic_json, make_server, slice_result


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.server = make_server(port=0,data=self.temp.name)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        for event in list(self.server.store.stops.values()):
            event.set()
        self.server.shutdown()
        self.server.server_close()
        self.server.store.pool.shutdown(wait=True)
        self.thread.join()
        self.temp.cleanup()

    def request(self,path,data=None,method=None):
        body = json.dumps(data).encode() if data is not None else None
        with urlopen(Request(self.base+path,data=body,method=method,
                             headers={'Content-Type':'application/json'}),timeout=20) as response:
            return json.load(response)

    def test_project_save_update_and_archive(self):
        project = self.request('/api/default')
        record = self.request('/api/projects',project)
        project['name'] = '熱流体 round-trip'
        self.request('/api/projects/'+record['id'],project,'PUT')
        self.assertEqual(self.request('/api/projects')['projects'][0]['name'],project['name'])
        with urlopen(self.base+'/api/projects/'+record['id']+'/export') as response:
            archive = response.read()
        with urlopen(Request(self.base+'/api/projects/import',data=archive,
                             headers={'Content-Type':'application/zip'})) as response:
            restored = json.load(response)
        self.assertNotEqual(record['id'],restored['id'])
        self.assertEqual(restored['project'],project)

    def test_total_mesh_estimate_save_and_build(self):
        project = default_project()
        project['mesh'] = {'target_cells': 3500}
        estimate = self.request('/api/mesh/estimate', project)
        self.assertEqual(estimate['shape'], [33, 11, 11])
        self.assertEqual(estimate['total_cells'], 3993)
        saved = self.request('/api/projects', project)['project']
        self.assertEqual(saved['mesh']['target_cells'], 3500)
        mesh = self.request('/api/mesh', saved)
        self.assertEqual(mesh['shape'], estimate['shape'])
        np.testing.assert_allclose(mesh['spacing'], estimate['spacing'])
        project['mesh'] = {'target_cells': 3_000_000}
        self.assertEqual(self.request('/api/mesh/estimate', project)['shape'], [300, 100, 100])
        project['mesh'] = {'target_cells': 1.5}
        with self.assertRaises(HTTPError) as error:
            self.request('/api/mesh/estimate', project)
        self.assertEqual(error.exception.code, 400)

    def test_material_catalog_and_csv_survive_project_archive(self):
        from workbench.schema import evaluate_property
        catalog = self.request('/api/materials/presets')['presets']
        self.assertEqual({entry['id'] for entry in catalog},
                         {'water-temperature', 'water-constant', 'air-temperature', 'air-constant'})
        for preset in catalog:
            with self.subTest(preset=preset['id']):
                project = default_project()
                project['materials'] = [{**preset['material'], 'id': 'fluid'}]
                project['physics']['material_id'] = 'fluid'
                stored = self.request('/api/projects', project)
                self.assertEqual(stored['project']['materials'][0], project['materials'][0])
                material = stored['project']['materials'][0]
                self.assertEqual(preset['pressure_pa'], 101325)
                self.assertTrue(preset['sources'][0]['url'].startswith('https://coolprop.org/'))
                density = evaluate_property(material['density'], 293.15)
                viscosity = evaluate_property(material['viscosity'], 293.15)
                if preset['id'].startswith('water'):
                    self.assertTrue(997 < density < 999)
                    self.assertTrue(.00099 < viscosity < .00102)
                else:
                    self.assertTrue(1.20 < density < 1.21)
                    self.assertTrue(1.80e-5 < viscosity < 1.84e-5)
        parsed = self.request('/api/materials/csv', {
            'text': 'temperature_C,density,viscosity\n40,992.2,0.000653\n20,998.2,0.001002\n'})
        self.assertEqual(parsed['rows'], 2)
        self.assertEqual(parsed['properties']['density']['points'], [[293.15, 998.2], [313.15, 992.2]])
        project = default_project()
        project['materials'][0].update(parsed['properties'])
        original = self.request('/api/projects', project)
        archive = self.server.store.archive(original['id'])
        with urlopen(Request(self.base + '/api/projects/import', data=archive,
                             headers={'Content-Type': 'application/zip'})) as response:
            imported = json.load(response)
        self.assertEqual(imported['project']['materials'], original['project']['materials'])

    def test_material_csv_api_rejects_bad_content_without_saving(self):
        before = self.request('/api/projects')
        for body in ({'text': 'temperature_C,density\n20,1\n20,2'},
                     {'text': 'temperature_K,value\n290,1\n310,2'},
                     {'text': 123}, {'text': 'x', 'project_id': 'a' * 32}):
            with self.subTest(body=body), self.assertRaises(HTTPError) as error:
                self.request('/api/materials/csv', body)
            self.assertEqual(error.exception.code, 400)
        self.assertEqual(self.request('/api/projects'), before)
        parsed = self.request('/api/materials/csv', {
            'text': '\ufefftemperature_K,value\r\n290,1.8e-5\r\n310,1.9e-5\r\n', 'property_key': 'viscosity'})
        self.assertEqual(set(parsed['properties']), {'viscosity'})
        self.assertEqual(parsed['properties']['viscosity']['points'], [[290., 1.8e-5], [310., 1.9e-5]])

    def test_reject_traversal_and_invalid_input(self):
        with self.assertRaises(HTTPError) as error:
            self.request('/api/projects/invalid')
        self.assertEqual(error.exception.code,400)
        p = default_project()
        p['mesh']['cells'] = [-1,10,10]
        with self.assertRaises(HTTPError) as error:
            self.request('/api/projects',p)
        self.assertEqual(error.exception.code,400)
        archive = io.BytesIO()
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('../escape.txt','bad')
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.base+'/api/projects/import',data=archive.getvalue()))
        self.assertEqual(error.exception.code,400)
        self.assertFalse((Path(self.temp.name).parent/'escape.txt').exists())

    def test_mesh_and_slice_orientation(self):
        mesh = self.request('/api/mesh',default_project())
        self.assertEqual(mesh['total_cells'],int(np.prod(mesh['shape'])))
        directory = Path(self.temp.name)/'slice'
        directory.mkdir()
        values = np.arange(60,dtype=float).reshape((3,4,5))
        mask = np.ones((3,4,5),dtype=bool)
        mask[1,2,3] = False
        np.savez(directory/'fields.npz',temperature=values,pressure=values*2,
                 velocity=np.ones((3,3,4,5)),fluid_mask=mask,origin=[0,0,0],spacing=[.1,.1,.1])
        result = slice_result(directory,{'axis':['x'],'index':['1']})
        self.assertEqual(result['values'][0][0],20)
        self.assertIsNone(result['values'][2][3])
        self.assertEqual(result['shape'],[3,4,5])

    def test_restart_marks_unfinished_run_interrupted(self):
        rid = 'a'*32
        atomic_json(Path(self.temp.name)/'runs'/rid/'status.json',{'id':rid,'status':'running'})
        other = Store(self.temp.name)
        self.assertEqual(json.loads((other.runs/rid/'status.json').read_text())['status'],'interrupted')
        other.pool.shutdown()

    def test_conjugate_temperature_slice_keeps_solid_but_flow_excludes_it(self):
        directory = Path(self.temp.name) / 'conjugate-slice'
        directory.mkdir()
        fluid = np.ones((3, 4, 5), dtype=bool)
        fluid[1, 2, 3] = False
        solid = ~fluid
        np.savez(directory / 'fields.npz', temperature=np.full(fluid.shape, 310.),
                 pressure=np.ones(fluid.shape), velocity=np.ones((3, *fluid.shape)),
                 eddy_viscosity=np.full(fluid.shape, .001), fluid_mask=fluid,
                 solid_mask=solid, thermal_mask=fluid | solid, origin=[0, 0, 0], spacing=[.1]*3)
        query = {'axis': ['x'], 'index': ['1']}
        temperature = slice_result(directory, query)
        self.assertEqual(temperature['values'][2][3], 310.)
        self.assertTrue(temperature['solid_mask'][2][3])
        speed = slice_result(directory, {**query, 'field': ['speed']})
        self.assertIsNone(speed['values'][2][3])
        eddy = slice_result(directory, {**query, 'field': ['eddy_viscosity']})
        self.assertEqual(eddy['values'][0][0], .001)
        self.assertEqual(eddy['unit'], 'm²/s')

    def test_legacy_cad_asset_endpoint_exposes_selectable_patches(self):
        import trimesh
        from workbench.geometry import import_cad
        source = Path(self.temp.name) / 'legacy.stl'
        trimesh.creation.box(extents=[.03, .01, .01]).export(source)
        asset = import_cad(source, self.server.store.assets)
        legacy = dict(asset)
        legacy.pop('surface_groups')
        legacy.pop('triangle_groups')
        metadata = self.server.store.assets / asset['id'] / 'metadata.json'
        atomic_json(metadata, legacy)
        fetched = self.request('/api/assets/' + asset['id'])
        self.assertEqual(len(fetched['surface_groups']), 6)
        self.assertEqual(len(fetched['triangle_groups']), len(fetched['faces']))
        self.assertNotIn('surface_groups', json.loads(metadata.read_text()))

    def test_archive_preserves_historical_cad_and_fields(self):
        store = self.server.store
        aid, rid = 'b'*32, 'c'*32
        asset = store.assets/aid
        asset.mkdir()
        atomic_json(asset/'metadata.json',{'id':aid,'original_name':'original.stl'})
        np.savez(asset/'surface.npz',vertices=np.zeros((3,3)),faces=np.array([[0,1,2]]))
        (asset/'original.stl').write_bytes(b'original CAD source')
        record = store.save(default_project())
        directory = store.runs/rid
        directory.mkdir()
        snapshot = default_project()
        snapshot['geometry'].update(kind='cad',asset_id=aid)
        atomic_json(directory/'input.json',snapshot)
        atomic_json(directory/'status.json',{'id':rid,'project_id':record['id'],'status':'completed','created_at':'2026-09-16'})
        np.savez(directory/'fields.npz',temperature=np.ones((3,3,3))*310)
        restored = store.import_archive(store.archive(record['id']))
        runs = store.project(restored['id'])['runs']
        self.assertEqual(len(runs),1)
        restored_snapshot = json.loads((store.runs/runs[0]['id']/'input.json').read_text())
        restored_aid = restored_snapshot['geometry']['asset_id']
        self.assertNotEqual(restored_aid,aid)
        self.assertEqual((store.assets/restored_aid/'original.stl').read_bytes(),b'original CAD source')
        with np.load(store.runs/runs[0]['id']/'fields.npz') as data:
            self.assertEqual(float(data['temperature'].mean()),310)

    def test_http_run_slice_and_stop(self):
        p = default_project()
        p['study'].update(steps=8,output_interval=4)
        run = self.request('/api/runs',{'project':p})
        def wait_for_terminal(rid):
            deadline = time.monotonic()+60
            while time.monotonic() < deadline:
                state = self.request('/api/runs/'+rid)
                if state['status'] not in ('queued','running','stopping'):
                    return state
                time.sleep(.2)
            self.fail('run did not terminate in 60 seconds')
        state = wait_for_terminal(run['id'])
        self.assertEqual(state['status'],'completed',state.get('error'))
        self.assertEqual(state['shape'],p['mesh']['cells'])
        self.assertEqual(state['history'][0]['residual_kind'], 'sample_change_linf_v1')
        self.assertIsNone(state['history'][0]['residual'])
        self.assertTrue(all(key in state['history'][-1] for key in ('velocity_residual', 'pressure_residual', 'temperature_residual',
                                                                  'monitor_satisfied', 'velocity_change_max', 'pressure_change_max')))
        self.assertEqual(state['diagnostics']['convergence_monitor']['tolerance'], p['study']['monitor']['tolerance'])
        self.assertIn('XLB',state['diagnostics']['backend'])
        result = self.request('/api/runs/'+run['id']+'/slice?field=temperature&axis=y&index=4')
        self.assertGreater(result['max'],result['min'])
        with urlopen(self.base+'/api/runs/'+run['id']+'/files/fields.vtk') as response:
            self.assertTrue(response.read().startswith(b'# vtk DataFile'))
        p['study']['steps'] = 100000
        long_run = self.request('/api/runs',{'project':p,'project_id':run['project_id']})
        self.request('/api/runs/'+long_run['id']+'/stop',{},'POST')
        state = wait_for_terminal(long_run['id'])
        self.assertEqual(state['status'],'stopped')

    def test_step_import_in_http_worker(self):
        try:
            import gmsh
        except (ImportError,OSError):
            self.skipTest('Gmsh native CAD dependencies unavailable')
        source = Path(self.temp.name)/'threaded.step'
        gmsh.initialize(interruptible=False)
        try:
            gmsh.option.setNumber('General.Terminal',0)
            gmsh.model.add('thread-test')
            gmsh.model.occ.addBox(0,0,0,30,10,10)
            gmsh.model.occ.synchronize()
            gmsh.write(str(source))
        finally:
            gmsh.finalize()
        with urlopen(Request(self.base+'/api/import?filename=threaded.step&unit=mm',data=source.read_bytes(),
                             headers={'Content-Type':'application/octet-stream'})) as response:
            asset = json.load(response)
        self.assertTrue(asset['watertight'])
        np.testing.assert_allclose(asset['bounds'],[[0,0,0],[.03,.01,.01]],atol=1e-12)
        p = default_project()
        p['geometry'].update(kind='cad',asset_id=asset['id'])
        mesh = self.request('/api/mesh',p)
        self.assertGreater(mesh['fluid_cells'],0)


if __name__ == '__main__':
    unittest.main()
