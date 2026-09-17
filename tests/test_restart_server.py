"""Continuation through HTTP, including immutable inputs and portable archives."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from workbench.server import atomic_json, make_server
from tests.test_solver import _project


@unittest.skipUnless(importlib.util.find_spec('xlb') and importlib.util.find_spec('jax'), 'XLB/JAX required')
class RestartServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {'XLB_COMPUTE_BACKEND': 'device'})
        self.environment.start()
        self.server = make_server(port=0, data=self.temp.name)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_port)

    def tearDown(self):
        for event in list(self.server.store.stops.values()):
            event.set()
        self.server.shutdown()
        self.server.server_close()
        self.server.store.pool.shutdown(wait=True)
        self.thread.join()
        self.environment.stop()
        self.temp.cleanup()

    def request(self, path, data=None, method=None):
        body = json.dumps(data).encode() if data is not None else None
        with urlopen(Request(self.base + path, data=body, method=method,
                             headers={'Content-Type': 'application/json'}), timeout=30) as response:
            return json.load(response)

    def finish(self, state):
        # Poll the local status file to avoid filling test output with HTTP logs.
        status = self.server.store.runs / state['id'] / 'status.json'
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            state = json.loads(status.read_text())
            if state['status'] not in ('queued', 'running', 'stopping'):
                self.assertEqual(state['status'], 'completed', state)
                return self.request('/api/runs/' + state['id'])
            time.sleep(.1)
        self.fail('Simulation did not finish in 90 seconds')

    def test_continuation_keeps_source_and_project_edits_and_survives_archive(self):
        project = _project(steps=2)
        project['study'].update(output_interval=1, snapshot_interval=1)
        source = self.finish(self.request('/api/runs', {'project': project}))
        rid, pid = source['id'], source['project_id']
        directory = self.server.store.runs / rid
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in directory.iterdir() if p.is_file()}
        info = self.request('/api/runs/' + rid + '/restart')
        self.assertEqual(info, {'available': True, 'step': 2, 'dt': .001, 'time': .002})
        edited = copy.deepcopy(project)
        edited['name'] = 'New boundary settings for a different experiment'
        edited['boundaries'][0]['flow']['velocity'][0] = .02
        saved = self.request('/api/projects/' + pid, edited, 'PUT')
        child = self.finish(self.request('/api/runs/' + rid + '/restart', {'additional_steps': 3}))
        self.assertNotEqual(child['id'], rid)
        self.assertEqual(child['restart_from'], rid)
        self.assertEqual(child['steps'], 5)
        self.assertEqual(child['start_step'], 2)
        self.assertEqual(child['end_step'], 5)
        self.assertEqual(child['additional_steps'], 3)
        self.assertEqual(self.request('/api/projects/' + pid)['project'], saved['project'])
        child_input = self.request('/api/runs/' + child['id'] + '/files/input.json')
        self.assertEqual(child_input['boundaries'][0]['flow']['velocity'][0], .01)
        self.assertEqual(child_input['study']['steps'], 5)
        frames = self.request('/api/runs/' + child['id'] + '/frames')['frames']
        self.assertEqual([frame['step'] for frame in frames], [2, 3, 4, 5])
        self.assertEqual(frames[-1]['time'], .005)
        self.assertEqual(source_hashes, {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in directory.iterdir() if p.is_file()})
        with urlopen(self.base + '/api/projects/' + pid + '/export') as response:
            archive = response.read()
        with urlopen(Request(self.base + '/api/projects/import', data=archive,
                             headers={'Content-Type': 'application/zip'})) as response:
            imported = json.load(response)
        imported_runs = self.request('/api/projects/' + imported['id'])['runs']
        imported_child = next(run for run in imported_runs if run.get('start_step') == 2)
        imported_source = next(run for run in imported_runs if run.get('start_step') != 2)
        self.assertEqual(imported_child['restart_from'], imported_source['id'])
        self.assertTrue(self.request('/api/runs/' + imported_child['id'] + '/restart')['available'])
        grandchild = self.finish(self.request('/api/runs/' + imported_child['id'] + '/restart', {'additional_steps': 2}))
        self.assertEqual(grandchild['steps'], 7)
        self.assertEqual(grandchild['project_id'], imported['id'])

    def test_invalid_requests_and_legacy_run_are_explicitly_rejected(self):
        rid = 'a' * 32
        directory = self.server.store.runs / rid
        atomic_json(directory / 'status.json', {'id': rid, 'status': 'completed', 'steps': 2})
        info = self.request('/api/runs/' + rid + '/restart')
        self.assertFalse(info['available'])
        self.assertIn('再開用データがありません', info['reason'])
        for data in ({'additional_steps': 2}, {'additional_steps': 0}, {'additional_steps': True},
                     {'additional_steps': 1.5}, {'additional_steps': -1}, {},
                     {'additional_steps': 2, 'project': _project()}):
            with self.subTest(data=data), self.assertRaises(HTTPError) as error:
                self.request('/api/runs/' + rid + '/restart', data)
            self.assertEqual(error.exception.code, 400)
        atomic_json(directory / 'status.json', {'id': rid, 'status': 'running'})
        self.assertFalse(self.request('/api/runs/' + rid + '/restart')['available'])
        atomic_json(directory / 'status.json', {'id': rid, 'status': 'completed', 'steps': 2})
        for name in ('restart.npz', 'mesh.npz', 'input.json'):
            (directory / name).write_bytes(b'invalid archive')
        info = self.request('/api/runs/' + rid + '/restart')
        self.assertFalse(info['available'])
        self.assertIn('読み込めません', info['reason'])
