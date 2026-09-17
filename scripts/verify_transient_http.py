"""Verify published live frames, derived plots and portable frame archives over HTTP."""
import argparse
import io
import json
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8766')
    parser.add_argument('--output', default='docs/transient-http-verification.json')
    args = parser.parse_args()

    def request(path, body=None, raw=False, content_type='application/json'):
        payload = json.dumps(body).encode() if body is not None and not isinstance(body, bytes) else body
        with urlopen(Request(args.url + path, data=payload, headers={'Content-Type': content_type}), timeout=60) as response:
            return response.read() if raw else json.load(response)

    project = request('/api/default')
    project['name'] = '時系列・物理量 GPU検証'
    project['study'].update(steps=80, output_interval=5, snapshot_interval=2, device='cuda:0', dt=.005)
    project['physics']['turbulence'] = {'model': 'smagorinsky', 'smagorinsky_constant': .17, 'turbulent_prandtl': .9}
    run = request('/api/runs', {'project': project})
    rid = run['id']
    live = None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        run = request('/api/runs/' + rid)
        frames = request('/api/runs/' + rid + '/frames')
        available = [frame for frame in frames['frames'] if frame['step'] > 0]
        if live is None and run['status'] == 'running' and available:
            step = available[-1]['step']
            result = request(f'/api/runs/{rid}/slice?step={step}&field=velocity_x&axis=y&index=4')
            assert result['step'] == step, result
            live = {'step': step, 'status': run['status'], 'time': result['time']}
        if run['status'] not in ('queued', 'running', 'stopping'):
            break
        time.sleep(.05)
    assert run['status'] == 'completed', run
    assert live is not None, 'No live intermediate frame was observed'
    assert [f['step'] for f in frames['frames']] == list(range(0, 81, 2)), frames
    fields = {}
    for field in frames['fields']:
        result = request(f"/api/runs/{rid}/slice?step=20&field={field['id']}&axis=y&index=4")
        assert result['step'] == 20 and result['time'] == .1, result
        assert result['values'], result
        fields[field['id']] = {'unit': result['unit'], 'min': result['min'], 'max': result['max']}
    try:
        request(f'/api/runs/{rid}/slice?step=3&field=speed')
        raise AssertionError('An unsaved frame was accepted')
    except HTTPError as error:
        assert error.code == 404, error
    archive = request('/api/projects/' + run['project_id'] + '/export', raw=True)
    with zipfile.ZipFile(io.BytesIO(archive)) as stored:
        assert sum(name.endswith('.npz') and '/frames/step_' in name for name in stored.namelist()) == 41
    restored = request('/api/projects/import', archive, content_type='application/zip')
    restored_project = request('/api/projects/' + restored['id'])
    restored_rid = restored_project['runs'][0]['id']
    restored_frames = request('/api/runs/' + restored_rid + '/frames')
    assert restored_frames['frames'] == frames['frames']
    original_slice = request(f'/api/runs/{rid}/slice?step=20&field=temperature&axis=y&index=4')
    restored_slice = request(f'/api/runs/{restored_rid}/slice?step=20&field=temperature&axis=y&index=4')
    assert original_slice == restored_slice
    proof = {'url': args.url, 'run_id': rid, 'restored_run_id': restored_rid, 'live_frame': live,
             'frames': frames['frames'], 'fields': fields, 'diagnostics': run['diagnostics'],
             'archive_bytes': len(archive), 'archive_restore': 'passed',
             'provenance': request(f'/api/runs/{rid}/files/provenance.json')}
    Path(args.output).write_text(json.dumps(proof, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'run': rid, 'live': live, 'frame_count': len(frames['frames']),
                      'field_count': len(fields), 'archive_restore': 'passed'}), flush=True)


if __name__ == '__main__':
    main()
