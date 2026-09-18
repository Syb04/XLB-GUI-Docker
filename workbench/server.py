"""Local HTTP workbench, immutable run snapshots and portable archives."""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import io
import hashlib
import importlib.metadata
import json
import mimetypes
import os
import re
import shutil
import signal
import tempfile
import threading
import traceback
from urllib.parse import parse_qs, urlparse
import uuid
import zipfile

import numpy as np
from .schema import default_project, validate_project
from .geometry import build_mesh, estimate_mesh, import_cad, load_asset_metadata
from .limits import configured_limits
from .results import frames_result, slice_result as _slice_result

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"^[a-f0-9]{32}$")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp-' + uuid.uuid4().hex)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def identifier(value):
    if not ID.fullmatch(value):
        raise ValueError('Invalid identifier')
    return value


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.projects = self.root / 'projects'
        self.assets = self.root / 'assets'
        self.runs = self.root / 'runs'
        for path in (self.projects, self.assets, self.runs):
            path.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='xlb-worker')
        self.stops = {}
        # Unfinished workers have no published final checkpoint; they cannot
        # be resumed automatically after a process restart.
        for path in self.runs.glob('*/status.json'):
            state = read_json(path)
            if state.get('status') in ('queued', 'running', 'stopping'):
                state.update(status='interrupted', error='Server restarted; start a new run.', finished_at=now())
                atomic_json(path, state)

    def save(self, project, pid=None):
        project = validate_project(project)
        pid = identifier(pid) if pid else uuid.uuid4().hex
        for field in ('asset_id', 'solid_asset_id'):
            if project['geometry'].get(field):
                asset = identifier(project['geometry'][field])
                if not (self.assets / asset / 'metadata.json').exists():
                    raise ValueError('CAD asset does not exist; import the CAD file first.')
        record = {'id': pid, 'project': project, 'updated_at': now()}
        with self.lock:
            atomic_json(self.projects / pid / 'project.json', record)
        return record

    def project(self, pid):
        record = read_json(self.projects / identifier(pid) / 'project.json')
        record['runs'] = sorted([read_json(p) for p in self.runs.glob('*/status.json')
                                 if read_json(p).get('project_id') == pid],
                                key=lambda x: x['created_at'], reverse=True)
        return record

    def restart_info(self, rid):
        directory = self.runs / identifier(rid)
        state = read_json(directory / 'status.json')
        if state.get('status') not in ('completed', 'stopped'):
            return {'available': False, 'reason': '完了または停止した計算から再開できます。'}
        for name in ('restart.npz', 'mesh.npz', 'input.json'):
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                return {'available': False, 'reason': '再開データの保存先が不正です。'}
            if not path.is_file():
                return {'available': False, 'reason': 'この実行には再開用データがありません。機能追加後に実行した計算から再開できます。'}
        try:
            from .restart import read_restart_metadata
            metadata = read_restart_metadata(directory / 'restart.npz')
            step, dt = metadata['step'], metadata['dt']
            if state.get('steps') != step:
                raise ValueError('checkpoint step does not match completed run')
            if read_json(directory / 'input.json')['study']['dt'] != dt:
                raise ValueError('checkpoint dt does not match run input')
        except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            return {'available': False, 'reason': f'再開データを読み込めません: {exc}'}
        return {'available': True, 'step': step, 'dt': dt, 'time': step * dt}

    def restart_run(self, rid, additional_steps):
        rid = identifier(rid)
        if isinstance(additional_steps, bool) or not isinstance(additional_steps, int) or additional_steps < 1:
            raise ValueError('additional_steps must be a positive integer')
        info = self.restart_info(rid)
        if not info['available']:
            raise ValueError(info['reason'])
        end_step = info['step'] + additional_steps
        if end_step > 999_999_999:
            raise ValueError('累積ステップ数は999999999以下にしてください。')
        directory = self.runs / rid
        source = read_json(directory / 'status.json')
        project = read_json(directory / 'input.json')
        project['study']['steps'] = end_step
        project['study']['output_interval'] = min(project['study']['output_interval'], end_step)
        return self.start(project, source['project_id'], restart_source=rid, start_step=info['step'])

    def start(self, project, pid=None, *, restart_source=None, start_step=0):
        if pid and not (self.projects / identifier(pid) / 'project.json').exists():
            raise ValueError('Unknown project')
        # A continuation uses the immutable source input, while edits in the
        # saved project remain untouched. The new run owns its own artifacts.
        record = {'id': pid, 'project': validate_project(project)} if restart_source else self.save(project, pid)
        rid = uuid.uuid4().hex
        directory = self.runs / rid
        directory.mkdir()
        atomic_json(directory / 'input.json', record['project'])
        sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted((ROOT/'workbench').glob('*.py'))}
        versions = {}
        for package in ('xlb','jax','jaxlib','numpy','scipy','trimesh','gmsh','warp-lang'):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        atomic_json(directory/'provenance.json',{'created_at':now(),'source_sha256':sources,
                                               'packages':versions,'limits':configured_limits(),
                                               'compute_backend':os.environ.get('XLB_COMPUTE_BACKEND','auto')})
        state = {'id': rid, 'project_id': record['id'], 'name': record['project']['name'],
                 'status': 'queued', 'created_at': now(), 'progress': 0, 'history': []}
        if restart_source:
            state.update(restart_from=restart_source, start_step=start_step,
                         end_step=project['study']['steps'], additional_steps=project['study']['steps'] - start_step)
        atomic_json(directory / 'status.json', state)
        stop = threading.Event()
        self.stops[rid] = stop
        self.pool.submit(self._execute, rid, record['project'], stop, restart_source)
        return state

    def _execute(self, rid, project, stop, restart_source=None):
        directory = self.runs / rid
        def update(changes):
            with self.lock:
                state = read_json(directory / 'status.json')
                state.update(changes)
                atomic_json(directory / 'status.json', state)
        try:
            if stop.is_set():
                update({'status': 'stopped', 'finished_at': now()})
                return
            update({'status': 'running', 'started_at': now(), 'phase': 'meshing'})
            restart_path = None
            if restart_source:
                source_directory = self.runs / identifier(restart_source)
                restart_path = source_directory / 'restart.npz'
                source_mesh = source_directory / 'mesh.npz'
                if any(not path.resolve().is_relative_to(source_directory.resolve()) for path in (restart_path, source_mesh)):
                    raise ValueError('Unsafe restart data path')
                with np.load(source_mesh, allow_pickle=False) as archive:
                    mesh = {key: archive[key] for key in archive.files if not key.startswith('boundary_links:')}
                    mesh['boundary_links'] = {key[len('boundary_links:'):]: archive[key] for key in archive.files
                                              if key.startswith('boundary_links:')}
                mesh['warnings'] = read_json(source_directory / 'status.json').get('mesh_warnings', [])
            else:
                mesh = build_mesh(project, self.assets)
            mesh_arrays = {k: v for k, v in mesh.items()
                           if isinstance(v, np.ndarray) or k in ('origin','spacing','shape')}
            # Keep individual CAD lattice contacts in the portable mesh snapshot.
            for selector, links in mesh.get('boundary_links', {}).items():
                mesh_arrays['boundary_links:' + selector] = np.asarray(links, dtype=bool)
            np.savez_compressed(directory / 'mesh.npz', **mesh_arrays)
            update({'phase': 'solving', 'shape': list(np.asarray(mesh['fluid_mask']).shape),
                    'origin': np.asarray(mesh['origin']).tolist(), 'spacing': np.asarray(mesh['spacing']).tolist(),
                    'fluid_cells': int(np.count_nonzero(mesh['fluid_mask'])),
                    'solid_cells': int(np.count_nonzero(mesh.get('solid_mask', []))),
                    'mesh_warnings': mesh.get('warnings', [])})
            from .solver import simulate
            history = deque(maxlen=2000)
            def progress(sample):
                sample = json.loads(json.dumps(sample, default=lambda v: v.item() if hasattr(v, 'item') else str(v)))
                history.append(sample)
                update({**sample, 'history': list(history)})
            restart_options = {'restart_from': restart_path} if restart_path is not None else {}
            result = simulate(project, mesh, directory, on_progress=progress, should_stop=stop.is_set, **restart_options)
            atomic_json(directory / 'result.json', result)
            update({**result, 'finished_at': now(), 'phase': 'finished',
                    'files': [p.relative_to(directory).as_posix() for p in directory.rglob('*')
                              if p.is_file() and p.name != 'status.json']})
        except Exception as exc:
            (directory / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
            update({'status': 'failed', 'error': str(exc), 'finished_at': now(), 'phase': 'failed'})
        finally:
            self.stops.pop(rid, None)

    def stop(self, rid):
        identifier(rid)
        with self.lock:
            state = read_json(self.runs / rid / 'status.json')
            event = self.stops.get(rid)
            if event and state['status'] in ('queued', 'running', 'stopping'):
                event.set()
                state['status'] = 'stopping'
                atomic_json(self.runs / rid / 'status.json', state)
        return state

    def archive(self, pid):
        record = self.project(pid)
        if any(r['status'] in ('queued','running','stopping') for r in record['runs']):
            raise ValueError('Stop or finish active runs before exporting this project.')
        output = io.BytesIO()
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('manifest.json', json.dumps({'format':'xlb-workbench','version':1,'project_id':pid}))
            z.writestr('project.json', json.dumps(record['project'], ensure_ascii=False))
            asset_ids = {
                record['project']['geometry'].get('asset_id'),
                record['project']['geometry'].get('solid_asset_id'),
            }
            for run in record['runs']:
                directory = self.runs / run['id']
                run_geometry = read_json(directory/'input.json')['geometry']
                asset_ids.update({run_geometry.get('asset_id'), run_geometry.get('solid_asset_id')})
                # Read the atomically published manifest before walking the
                # run. Unpublished/stale files below frames are excluded so
                # an export cannot expose a half-written transient frame.
                published_frames = set()
                manifest = directory / 'frames.json'
                manifest_bytes = None
                if manifest.exists():
                    try:
                        # Keep the exact bytes used to select the frame files;
                        # a later atomic manifest replacement must not pair a
                        # new manifest with this older file set.
                        manifest_bytes = manifest.read_bytes()
                        value = json.loads(manifest_bytes.decode('utf-8'))
                        for item in value.get('frames', []):
                            filename = item.get('file') if isinstance(item, dict) else None
                            if isinstance(filename, str) and re.fullmatch(r'step_\d{9}\.npz', filename):
                                frame_path = directory / 'frames' / filename
                                if frame_path.is_file():
                                    published_frames.add(Path('frames') / filename)
                    except (OSError, ValueError, TypeError, AttributeError):
                        # Old runs without a usable transient manifest remain
                        # exportable; their final artifacts retain compatibility.
                        published_frames.clear()
                if (directory / 'frames' / 'static.npz').is_file():
                    published_frames.add(Path('frames') / 'static.npz')
                for path in directory.rglob('*'):
                    if not path.is_file():
                        continue
                    if not path.resolve().is_relative_to(directory.resolve()):
                        continue
                    relative = path.relative_to(directory)
                    if relative == Path('frames.json') or '.tmp' in relative.name:
                        continue
                    if relative.parts and relative.parts[0] == 'frames':
                        if relative not in published_frames:
                            continue
                    try:
                        z.write(path, 'runs/' + run['id'] + '/' + relative.as_posix())
                    except FileNotFoundError:
                        # A concurrently cleaned transient file was not part
                        # of this archive snapshot.
                        continue
                if manifest_bytes is not None:
                    z.writestr('runs/' + run['id'] + '/frames.json', manifest_bytes)
            for aid in asset_ids - {None}:
                identifier(aid)
                for path in (self.assets / aid).rglob('*'):
                    if path.is_file() and path.resolve().is_relative_to((self.assets / aid).resolve()):
                        z.write(path, 'assets/' + aid + '/' + path.relative_to(self.assets / aid).as_posix())
        return output.getvalue()

    def import_archive(self, data):
        # Stage and validate the entire archive before committing any records.
        with tempfile.TemporaryDirectory(dir=self.root) as temp:
            staging = Path(temp)
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                infos = z.infolist()
                if len(infos) > 10000 or sum(i.file_size for i in infos) > 1024**3:
                    raise ValueError('Archive exceeds 1 GiB or 10000 files.')
                for info in infos:
                    path = Path(info.filename)
                    if path.is_absolute() or '..' in path.parts or '\\' in info.filename or ':' in info.filename:
                        raise ValueError('Unsafe archive path')
                    if not (staging / path).resolve().is_relative_to(staging.resolve()):
                        raise ValueError('Unsafe archive path')
                z.extractall(staging)
            manifest = read_json(staging / 'manifest.json')
            if manifest.get('format') != 'xlb-workbench' or manifest.get('version') != 1:
                raise ValueError('Unsupported project archive')
            project = validate_project(read_json(staging / 'project.json'))
            mapping = {}
            staged_assets = staging / 'assets'
            if staged_assets.exists():
                for directory in staged_assets.iterdir():
                    old = identifier(directory.name)
                    metadata = read_json(directory / 'metadata.json')
                    with np.load(directory / 'surface.npz', allow_pickle=False) as surface:
                        for value in surface.values():
                            if not np.isfinite(value).all():
                                raise ValueError('Invalid CAD surface data')
                    new = uuid.uuid4().hex
                    mapping[old] = new
                    metadata['id'] = new
                    atomic_json(directory / 'metadata.json', metadata)
            def remap_project(p):
                for field in ('asset_id', 'solid_asset_id'):
                    aid = p['geometry'].get(field)
                    if aid:
                        if aid not in mapping:
                            raise ValueError('Archive is missing a referenced CAD asset')
                        p['geometry'][field] = mapping[aid]
                return p
            project = remap_project(project)
            pid = uuid.uuid4().hex
            run_copies = []
            run_mapping = {}
            if (staging / 'runs').exists():
                for directory in (staging / 'runs').iterdir():
                    identifier(directory.name)
                    snapshot = remap_project(validate_project(read_json(directory / 'input.json')))
                    state = read_json(directory / 'status.json')
                    rid = uuid.uuid4().hex
                    run_mapping[directory.name] = rid
                    state.update(id=rid, project_id=pid)
                    if state.get('status') in ('queued','running','stopping'):
                        state['status'] = 'interrupted'
                    atomic_json(directory / 'input.json', snapshot)
                    atomic_json(directory / 'status.json', state)
                    run_copies.append((directory, rid))
            for directory, _ in run_copies:
                state_path = directory / 'status.json'
                state = read_json(state_path)
                source = state.get('restart_from')
                if source:
                    if source in run_mapping:
                        state['restart_from'] = run_mapping[source]
                    else:
                        state['restart_from_original'] = state.pop('restart_from')
                    atomic_json(state_path, state)
            def move_tree(source, target):
                """Move staged trees without requiring Windows copystat rights."""
                try:
                    source.rename(target)
                except OSError:
                    # Archives may be staged on another filesystem.  The
                    # copyfile fallback avoids shutil.copytree's metadata
                    # operation, which is denied on some bind mounts.
                    shutil.copytree(source, target, copy_function=shutil.copyfile)
            for old, new in mapping.items():
                move_tree(staged_assets / old, self.assets / new)
            record = self.save(project, pid)
            for directory, rid in run_copies:
                move_tree(directory, self.runs / rid)
            return record


def sampled_cells(mask, max_points=6000):
    """Sample occupied cells without a 24-byte-per-cell full coordinate array."""
    count = int(np.count_nonzero(mask))
    stride = max(1, (count + max_points - 1) // max_points)
    flat = mask.reshape(-1)
    samples = []
    seen = 0
    for start in range(0, flat.size, 65536):
        occupied = np.flatnonzero(flat[start:start + 65536])
        samples.append(occupied[(-seen) % stride::stride] + start)
        seen += len(occupied)
    selected = np.concatenate(samples) if samples else np.empty(0, dtype=np.int64)
    return count, np.column_stack(np.unravel_index(selected, mask.shape))


def mesh_preview(mesh):
    mask = np.asarray(mesh['fluid_mask'], dtype=bool)
    fluid_count, cells = sampled_cells(mask)
    origin = np.asarray(mesh['origin'])
    spacing = np.asarray(mesh['spacing'])
    preview = {'points': (origin + (cells + .5) * spacing).tolist()}
    for key in ('vertices','faces'):
        value = mesh.get(key, mesh.get('surface_' + key, []))
        preview[key] = np.asarray(value).tolist()
    for key in ('surface_groups', 'triangle_groups'):
        if key in mesh:
            value = mesh[key]
            preview[key] = value.tolist() if isinstance(value, np.ndarray) else value
    solid = np.asarray(mesh.get('solid_mask', np.zeros_like(mask)), dtype=bool)
    solid_count, solid_cells = sampled_cells(solid)
    preview['solid_points'] = (origin + (solid_cells + .5) * spacing).tolist()
    response = {'shape': list(mask.shape), 'fluid_cells': fluid_count, 'total_cells': int(mask.size),
            'solid_cells': solid_count, 'limits': configured_limits(),
            'boundary_link_counts': {key: int(np.count_nonzero(value))
                                     for key, value in mesh.get('boundary_links', {}).items()},
            'origin': origin.tolist(), 'spacing': spacing.tolist(), 'preview': preview,
            'warnings': mesh.get('warnings', [])}
    if isinstance(mesh.get('geometry_metadata'), dict):
        response['geometry_metadata'] = mesh['geometry_metadata']
    return response


def slice_result(directory, query):
    field = query.get('field', ['temperature'])[0]
    axis = query.get('axis', ['z'])[0]
    if field not in ('temperature','speed','pressure','eddy_viscosity') or axis not in ('x','y','z'):
        raise ValueError('Unknown field or slice axis')
    with np.load(directory / 'fields.npz', allow_pickle=False) as data:
        if field != 'speed' and field not in data:
            raise ValueError('This run does not contain the requested field')
        values = np.linalg.norm(data['velocity'], axis=0) if field == 'speed' else data[field]
        idx = 'xyz'.index(axis)
        index = int(query.get('index', [values.shape[idx] // 2])[0])
        if not 0 <= index < values.shape[idx]:
            raise ValueError('Slice index outside mesh')
        plane = np.take(values, index, axis=idx)
        active = data['thermal_mask'] if field == 'temperature' and 'thermal_mask' in data else data['fluid_mask']
        mask = np.take(active, index, axis=idx)
        solid = np.take(data['solid_mask'], index, axis=idx) if 'solid_mask' in data else np.zeros_like(mask)
        plane = np.where(mask, plane, np.nan)
        finite = plane[np.isfinite(plane)]
        result = [[float(v) if np.isfinite(v) else None for v in row] for row in plane]
        other = [a for a in range(3) if a != idx]
        origin, spacing = data['origin'], data['spacing']
        return {'values': result, 'min': float(finite.min()) if finite.size else 0,
                'max': float(finite.max()) if finite.size else 0,
                'solid_mask': solid.tolist(),
                'unit': {'temperature':'K','speed':'m/s','pressure':'Pa','eddy_viscosity':'m²/s'}[field],
                'field': field, 'axis': axis, 'index': index, 'shape': list(values.shape),
                'extent': [float(origin[a]) for a in other] +
                          [float(origin[a] + values.shape[a]*spacing[a]) for a in other]}


class Handler(BaseHTTPRequestHandler):
    server_version = 'XLBWorkbench/0.1'

    def log_message(self, fmt, *args):
        print('%s %s' % (self.log_date_time_string(), fmt % args), flush=True)

    def send_data(self, data, status=200, content_type='application/json; charset=utf-8', filename=None):
        if isinstance(data, (dict,list)):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-store')
        if filename:
            self.send_header('Content-Disposition', f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def body(self, raw=False):
        length = int(self.headers.get('Content-Length', '0'))
        limit = 256*1024*1024 if raw else 2*1024*1024
        if length < 0 or length > limit:
            raise ValueError('Upload exceeds size limit')
        body = self.rfile.read(length)
        return body if raw else json.loads(body)

    def do_GET(self):
        self.dispatch('GET')

    def do_POST(self):
        self.dispatch('POST')

    def do_PUT(self):
        self.dispatch('PUT')

    def dispatch(self, method):
        try:
            self.route(method)
        except FileNotFoundError:
            self.send_data({'error':'Requested project, asset or result does not exist.'},404)
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            self.send_data({'error':str(exc)},400)
        except Exception as exc:
            traceback.print_exc()
            self.send_data({'error':str(exc)},500)

    def route(self, method):
        parsed = urlparse(self.path)
        parts = parsed.path.strip('/').split('/')
        query = parse_qs(parsed.query)
        store = self.server.store
        if method in ('POST','PUT'):
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                self.send_data({'error':'Cross-origin writes are not permitted.'},403)
                return
        if parts[0] != 'api':
            if method != 'GET':
                raise ValueError('Method not supported')
            relative = parsed.path.lstrip('/') or 'index.html'
            if relative.startswith('static/'):
                relative = relative[len('static/'):]
            path = (ROOT / 'static' / relative).resolve()
            if not path.is_relative_to((ROOT / 'static').resolve()):
                raise ValueError('Invalid path')
            self.send_data(path.read_bytes(), content_type=mimetypes.guess_type(path)[0] or 'application/octet-stream')
        elif parts == ['api','health'] and method == 'GET':
            packages = {}
            for name in ('xlb','warp-lang','numpy','trimesh','gmsh'):
                try:
                    packages[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    packages[name] = None
            self.send_data({'status':'ok', 'version':'0.1.0', 'packages':packages,
                            'limits':configured_limits()})
        elif parts == ['api','default'] and method == 'GET':
            self.send_data(default_project())
        elif parts == ['api', 'materials', 'presets'] and method == 'GET':
            self.send_data(read_json(ROOT / 'workbench' / 'data' / 'material_presets.json'))
        elif parts == ['api', 'materials', 'csv'] and method == 'POST':
            from .material_csv import parse_material_csv
            data = self.body()
            if not isinstance(data, dict) or 'text' not in data or set(data) - {'text', 'property_key'}:
                raise ValueError('CSV読み込みにはtextと任意のproperty_keyを指定してください。')
            self.send_data(parse_material_csv(data['text'], property_key=data.get('property_key')))
        elif parts == ['api','projects']:
            if method == 'GET':
                records = [read_json(p) for p in store.projects.glob('*/project.json')]
                self.send_data({'projects': sorted([{'id':p['id'],'name':p['project']['name'],'updated_at':p['updated_at']} for p in records], key=lambda p:p['updated_at'], reverse=True)})
            elif method == 'POST':
                self.send_data(store.save(self.body()),201)
            else:
                raise ValueError('Method not supported')
        elif parts == ['api','projects','import'] and method == 'POST':
            self.send_data(store.import_archive(self.body(raw=True)),201)
        elif len(parts) == 3 and parts[:2] == ['api','projects']:
            if method == 'GET':
                self.send_data(store.project(parts[2]))
            elif method == 'PUT':
                if not (store.projects / identifier(parts[2]) / 'project.json').exists():
                    raise FileNotFoundError()
                self.send_data(store.save(self.body(),parts[2]))
            else:
                raise ValueError('Method not supported')
        elif len(parts) == 4 and parts[:2] == ['api','projects'] and parts[3] == 'export' and method == 'GET':
            self.send_data(store.archive(parts[2]),content_type='application/zip',filename='xlb-project.zip')
        elif parts == ['api','import'] and method == 'POST':
            filename = query.get('filename',['geometry.stl'])[0]
            if Path(filename).name != filename or '\\' in filename:
                raise ValueError('Invalid filename')
            with tempfile.TemporaryDirectory(dir=store.root) as temp:
                path = Path(temp) / filename
                path.write_bytes(self.body(raw=True))
                with store.lock:
                    metadata = import_cad(path,store.assets,query.get('unit',['m'])[0])
            self.send_data(metadata,201)
        elif len(parts) == 3 and parts[:2] == ['api','assets'] and method == 'GET':
            merge_angle = query.get('surface_merge_angle', [12.0])[0]
            self.send_data(load_asset_metadata(store.assets, identifier(parts[2]), float(merge_angle)))
        elif parts == ['api','mesh','estimate'] and method == 'POST':
            self.send_data(estimate_mesh(self.body(),store.assets))
        elif parts == ['api','mesh'] and method == 'POST':
            self.send_data(mesh_preview(build_mesh(validate_project(self.body()),store.assets)))
        elif parts == ['api','runs'] and method == 'POST':
            data = self.body()
            self.send_data(store.start(data['project'],data.get('project_id')),202)
        elif len(parts) >= 3 and parts[:2] == ['api','runs']:
            directory = store.runs / identifier(parts[2])
            if not directory.is_dir():
                raise FileNotFoundError()
            if len(parts) == 3 and method == 'GET':
                self.send_data(read_json(directory / 'status.json'))
            elif len(parts) == 4 and parts[3] == 'restart':
                if method == 'GET':
                    self.send_data(store.restart_info(parts[2]))
                elif method == 'POST':
                    data = self.body()
                    if not isinstance(data, dict) or set(data) != {'additional_steps'}:
                        raise ValueError('Specify only additional_steps for a restart')
                    self.send_data(store.restart_run(parts[2], data['additional_steps']), 202)
                else:
                    raise ValueError('Method not supported')
            elif len(parts) == 4 and parts[3] == 'stop' and method == 'POST':
                self.send_data(store.stop(parts[2]))
            elif len(parts) == 4 and parts[3] == 'frames' and method == 'GET':
                self.send_data(frames_result(directory))
            elif len(parts) == 4 and parts[3] == 'slice' and method == 'GET':
                self.send_data(slice_result(directory,query))
            elif len(parts) == 5 and parts[3] == 'files' and method == 'GET':
                if parts[4] not in ('input.json','status.json','result.json','provenance.json','fields.npz','fields.vtk','history.csv','error.log','mesh.npz','frames.json','restart.npz'):
                    raise ValueError('Unknown artifact')
                self.send_data((directory / parts[4]).read_bytes(),content_type='application/octet-stream',filename=parts[4])
            else:
                raise ValueError('Unknown endpoint')
        else:
            self.send_data({'error':'Unknown endpoint'},404)


# Preserve the historical ``workbench.server.slice_result`` import while the
# implementation lives with the transient result readers.
slice_result = _slice_result


def make_server(host='127.0.0.1', port=8766, data=None):
    server = ThreadingHTTPServer((host,port),Handler)
    server.store = Store(data or os.environ.get('XLB_DATA_DIR',str(ROOT / 'data')))
    return server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8766)
    parser.add_argument('--data', default=None)
    args = parser.parse_args()
    server = make_server(args.host,args.port,args.data)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    print(f'XLB Workbench http://{args.host}:{args.port}',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        for event in list(server.store.stops.values()):
            event.set()
        server.store.pool.shutdown(wait=True,cancel_futures=True)
        server.server_close()


if __name__ == '__main__':
    main()
