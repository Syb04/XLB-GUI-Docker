"""Publish immutable transient fields without exposing partially written files."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import numpy as np


class SnapshotWriter:
    def __init__(self, directory, interval, dt, **static):
        self.directory = Path(directory)
        self.interval = int(interval)
        if self.interval < 0:
            raise ValueError('study.snapshot_interval must be non-negative')
        self.dt = float(dt)
        self.frames = []
        self.bytes_written = 0
        if self.interval:
            (self.directory / 'frames').mkdir(parents=True, exist_ok=True)
            self._write_npz(self.directory / 'frames' / 'static.npz', static)
            self._publish()

    def due(self, step):
        return bool(self.interval and step % self.interval == 0)

    def _write_npz(self, path, arrays):
        size = sum(np.asarray(value).nbytes for value in arrays.values())
        if shutil.disk_usage(path.parent).free < size + 64 * 1024**2:
            raise OSError('Insufficient disk space for transient fields; increase snapshot_interval or disable snapshots')
        temporary = path.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                # Uncompressed arrays keep checkpoint CPU cost bounded. ZIP
                # project export may compress them later, outside the solver.
                np.savez(stream, **arrays)
            os.replace(temporary, path)
            self.bytes_written += path.stat().st_size
        finally:
            temporary.unlink(missing_ok=True)

    def _publish(self):
        temporary = self.directory / 'frames.json.tmp'
        temporary.write_text(json.dumps({'frames': self.frames}), encoding='utf-8')
        os.replace(temporary, self.directory / 'frames.json')

    def write(self, step, *, velocity, pressure, temperature, eddy_viscosity):
        if not self.interval or any(frame['step'] == step for frame in self.frames):
            return
        filename = f'step_{step:09d}.npz'
        arrays = {key: np.asarray(value, dtype=np.float32) for key, value in
                  dict(velocity=velocity, pressure=pressure, temperature=temperature,
                       eddy_viscosity=eddy_viscosity).items()}
        self._write_npz(self.directory / 'frames' / filename, arrays)
        self.frames.append({'step': int(step), 'time': step * self.dt, 'file': filename})
        self._publish()
